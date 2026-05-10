"""
MolDA: unified model combining LLaDA + (optional) GNN + Q-Former.

Stage 1: LLaDA-only (string_only), LoRA + embed + head trainable
Stage 2: + GNN(frozen, pretrained) + Q-Former(trainable bridge), LLM/LoRA frozen
Stage 3: full multimodal (TBD)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.llada_wrapper import LLaDAWrapper
from src.training.loss import MaskedDiffusionLoss, MASK_TOKEN_ID

logger = logging.getLogger(__name__)

STAGE2_TRAINABLE_KEYS = ("qformer", "query_tokens", "opt_proj", "ln_graph")

# Stage 3 V-MolPO: LoRA + embed + Q-Former (optional) trainable, base LLM frozen.
# tune_gnn=false (default) 면 GNN/Q-Former 도 frozen, LoRA + embed 만.
STAGE3_LORA_TRAINABLE_KEYS = (
    "lora_", "modules_to_save", "wte", "embed", "lm_head",
)
STAGE3_QFORMER_TRAINABLE_KEYS = STAGE2_TRAINABLE_KEYS  # tune_gnn=true 일 때 추가


class MolDA(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stage = cfg.stage
        self.mol_representation = cfg.model.mol_representation

        # Core: LLaDA backbone
        self.llada = LLaDAWrapper(cfg)

        # Loss
        self.loss_fn = MaskedDiffusionLoss(
            mask_token_id=MASK_TOKEN_ID,
            log_nan=cfg.logging.get("log_nan_details", True),
            nan_log_dir=cfg.logging.get("nan_log_dir", "./nan_logs"),
            eos_token_id=self.llada.tokenizer.eos_token_id,
            normalization=cfg.training.get("loss_normalization", "global"),
        )

        # Stage 2+: GNN + Q-Former
        self.mol_token_id = None
        if self.stage >= 2:
            from src.model.gnn import GINETokenGT
            from src.model.qformer import QFormer
            self.gnn = GINETokenGT(cfg)
            llm_hidden_dim = self.llada.model.config.hidden_size
            self.qformer = QFormer(cfg, llm_hidden_dim=llm_hidden_dim)
            # <mol> placeholder token id — LLaDAWrapper가 tokenizer.mol_token_id로 캐시
            self.mol_token_id = getattr(
                self.llada.tokenizer, "mol_token_id",
                self.llada.tokenizer.convert_tokens_to_ids("<mol>"),
            )
            unk = self.llada.tokenizer.unk_token_id
            assert self.mol_token_id is not None and self.mol_token_id != unk, (
                "<mol> token must be in tokenizer vocab for stage>=2"
            )

        # Stage 3 V-MolPO: reference policy + per-task EMA anchor
        self.ref_model = None
        self.task_anchor_ema = None
        molpo_cfg = cfg.get("molpo", None)
        self.molpo_enabled = bool(
            self.stage == 3 and molpo_cfg and molpo_cfg.get("enabled", False)
        )
        if self.molpo_enabled:
            from src.model.ref_llada_wrapper import RefMolDA
            from src.training.v_molpo_loss import TaskAnchorEMA

            ref_ckpt = molpo_cfg.get("ref_ckpt_path")
            assert ref_ckpt and ref_ckpt != "???", (
                "molpo.ref_ckpt_path must be set when molpo.enabled=true. "
                "Pass via CLI: pretrained_ckpt_path=... molpo.ref_ckpt_path=..."
            )
            self.ref_model = RefMolDA(cfg, ref_ckpt_path=str(ref_ckpt))
            # Don't register ref_model.parameters() with optimizer
            # (already done via requires_grad=False in RefMolDA._freeze_all)

            ema_alpha = float(molpo_cfg.get("ema_alpha", 0.99))
            self.task_anchor_ema = TaskAnchorEMA(alpha=ema_alpha)
            logger.info(
                f"V-MolPO enabled: ref_ckpt={ref_ckpt}, "
                f"n_t={molpo_cfg.get('n_t', 2)}, "
                f"antithetic={molpo_cfg.get('antithetic', True)}, "
                f"beta={molpo_cfg.get('beta', 0.1)}"
            )

    @property
    def tokenizer(self):
        return self.llada.tokenizer

    def forward(self, batch: dict) -> dict:
        """Training forward pass with masked diffusion loss.

        Args:
            batch: dict with input_ids [B,L], labels [B,L], prompt_lengths [B], tasks [B]
                For V-MolPO: batch also has molpo_batch_size, molpo_batch_division.

        Returns:
            dict with "loss", "answer_length_mean", and (V-MolPO) v_molpo sub-metrics
        """
        # V-MolPO branch: stage==3 + molpo enabled + collator emitted MolPO batch
        if self.molpo_enabled and "molpo_batch_size" in batch:
            return self._molpo_forward(batch)

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # 1. Forward process: mask answer tokens
        noisy_ids, mask_indices, p_mask = self.loss_fn.make_noisy(input_ids, labels)

        # 2. Model forward — string+graph 모드면 graph token을 inputs_embeds에 injection
        attention_mask = batch.get("attention_mask", None)
        use_graph = (
            self.stage >= 2
            and self.mol_representation == "string+graph"
            and "graphs" in batch
        )
        if use_graph:
            outputs = self._forward_with_graph(noisy_ids, attention_mask, batch, input_ids)
        else:
            outputs = self.llada.model(input_ids=noisy_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, L, V]

        # 3. Compute loss (+ prediction metrics)
        loss_dict = self.loss_fn(
            logits=logits,
            input_ids=input_ids,
            labels=labels,
            mask_indices=mask_indices,
            p_mask=p_mask,
            tasks=batch.get("tasks"),
            global_step=batch.get("global_step", 0),
            log_train_detail=batch.get("_log_train_detail", False),
        )

        # Attach attention_mask for text decoding in train prediction logger
        if "_train_sample_detail" in loss_dict and attention_mask is not None:
            loss_dict["_train_sample_detail"]["attention_mask"] = attention_mask[0].cpu()

        return loss_dict

    def _forward_with_graph(self, noisy_ids, attention_mask, batch, input_ids):
        """Stage 2 forward: GNN(frozen) → Q-Former → inject at <mol> positions → LLM via inputs_embeds."""
        # Get noisy text embeddings via the live (PEFT-wrapped) embedding lookup
        text_embeds = self.llada.model.get_input_embeddings()(noisy_ids)

        graphs = batch["graphs"]
        # tune_gnn=False (Stage 2): GNN forward in no_grad, then detach
        tune_gnn = self.cfg.model.get("tune_gnn", False)
        if not tune_gnn:
            with torch.no_grad():
                graph_emb, graph_mask = self.gnn(
                    graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch,
                )
            graph_emb = graph_emb.detach()
        else:
            graph_emb, graph_mask = self.gnn(
                graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch,
            )

        # graph_emb dtype을 text_embeds와 맞춰서 mixed precision에서도 안전하게
        graph_emb = graph_emb.to(text_embeds.dtype)
        mol_tokens = self.qformer(graph_emb, graph_mask)  # [B, num_query_token, D]

        # Inject Q-Former tokens at <mol> positions
        is_mol = (input_ids == self.mol_token_id)  # [B, L]
        if is_mol.any():
            query_idx = is_mol.cumsum(dim=1) - 1
            bi, ti = is_mol.nonzero(as_tuple=True)
            text_embeds = text_embeds.clone()  # avoid in-place on inference graph
            text_embeds[bi, ti] = mol_tokens[bi, query_idx[bi, ti]].to(text_embeds.dtype)

        return self.llada.model(inputs_embeds=text_embeds, attention_mask=attention_mask)

    def _molpo_forward(self, batch: dict) -> dict:
        """V-MolPO forward: compute n_t-MC ELBO for both πθ and πref (antithetic),
        derive r_θ = β·(B̂_θ − B̂_ref), assemble L_pref + L_anchor (+ optional L_SFT).

        Expects batch from MolPOTrainCollator:
            input_ids[2B or 3B, L], labels, attention_mask, tasks, molpo_batch_size=B,
            molpo_batch_division=2 or 3, optionally graphs (replicated per slot).
        """
        from src.training.vrpo_elbo import compute_elbo
        from src.training.v_molpo_loss import compute_v_molpo_loss, combine_total_loss

        molpo_cfg = self.cfg.molpo
        B = int(batch["molpo_batch_size"])
        div = int(batch["molpo_batch_division"])
        n_t = int(molpo_cfg.get("n_t", 2))
        antithetic = bool(molpo_cfg.get("antithetic", True))
        beta = float(molpo_cfg.get("beta", 0.1))
        global_step = int(batch.get("global_step", 0))

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        all_tasks = batch.get("tasks", [])

        # Slice [chosen | rejected] (or [sft | chosen | rejected])
        if div == 2:
            chosen_slice = slice(0, B)
            rejected_slice = slice(B, 2 * B)
            sft_slice = None
        elif div == 3:
            sft_slice = slice(0, B)
            chosen_slice = slice(B, 2 * B)
            rejected_slice = slice(2 * B, 3 * B)
        else:
            raise ValueError(f"Invalid molpo_batch_division={div}")

        tasks_chosen = list(all_tasks[chosen_slice]) if all_tasks else [""] * B

        # Forward callable factories — capture batch for graph injection
        def _theta_fwd_for(slc):
            sub_batch = self._slice_batch(batch, slc)
            @torch.compile(disable=True) if False else (lambda f: f)
            def fwd(noisy_ids, attn_mask):
                return self._forward_logits(noisy_ids, attn_mask, sub_batch)
            return fwd

        def _ref_fwd_for(slc):
            sub_batch = self._slice_batch(batch, slc)
            return self.ref_model.make_forward_fn(sub_batch)

        # Antithetic: same seed for θ and ref on each y. Different seed across w/l.
        seed_w = global_step * 1000 + 7
        seed_l = global_step * 1000 + 13
        seed_w_ref = seed_w if antithetic else seed_w + 11
        seed_l_ref = seed_l if antithetic else seed_l + 11

        def _elbo(model_fwd, slc, seed):
            ids = input_ids[slc]
            lab = labels[slc]
            am = attention_mask[slc] if attention_mask is not None else None
            return compute_elbo(
                model_fwd, ids, lab, n_t=n_t, seed=seed,
                mask_token_id=MASK_TOKEN_ID, attention_mask=am,
            )

        # ELBOs (chosen / rejected, θ / ref)
        elbo_theta_w = _elbo(_theta_fwd_for(chosen_slice), chosen_slice, seed_w)
        elbo_theta_l = _elbo(_theta_fwd_for(rejected_slice), rejected_slice, seed_l)
        with torch.no_grad():
            elbo_ref_w = _elbo(_ref_fwd_for(chosen_slice), chosen_slice, seed_w_ref)
            elbo_ref_l = _elbo(_ref_fwd_for(rejected_slice), rejected_slice, seed_l_ref)

        # V-MolPO loss
        clip_burn_in = int(molpo_cfg.get("margin_clip_burn_in", 1000))
        v_out = compute_v_molpo_loss(
            elbo_theta_w=elbo_theta_w, elbo_ref_w=elbo_ref_w,
            elbo_theta_l=elbo_theta_l, elbo_ref_l=elbo_ref_l,
            tasks_chosen=tasks_chosen,
            task_anchor_ema=self.task_anchor_ema,
            beta=beta,
            molpo_lambda=float(molpo_cfg.get("lambda", 0.5)),
            margin_clip_scale=float(molpo_cfg.get("margin_clip_scale", 1.0)),
            margin_clip_active=(global_step < clip_burn_in),
            anc_rejected_weight=float(molpo_cfg.get("anc_rejected_weight", 0.0)),
            rejected_lambda=float(molpo_cfg.get("rejected_lambda", 1.5)),
            loss_type=str(molpo_cfg.get("loss_type", "sigmoid")),
        )

        # Optional SFT branch (mol_div=3)
        loss_sft = None
        per_sample_loss_sft = None
        per_sample_loss_no_eos_sft = None
        answer_length_mean = 0.0
        if div == 3 and float(molpo_cfg.get("sft_weight", 1.0)) > 0:
            sft_batch = self._slice_batch(batch, sft_slice)
            sft_out = self._sft_forward_internal(sft_batch)
            loss_sft = sft_out["loss"]
            per_sample_loss_sft = sft_out.get("per_sample_loss")
            per_sample_loss_no_eos_sft = sft_out.get("per_sample_loss_no_eos")
            answer_length_mean = sft_out.get("answer_length_mean", 0.0)

        # Total loss
        total = combine_total_loss(
            loss_sft=loss_sft, v_molpo_out=v_out,
            sft_weight=float(molpo_cfg.get("sft_weight", 1.0)),
            molpo_weight=float(molpo_cfg.get("molpo_weight", 0.25)),
            anc_rejected_weight=float(molpo_cfg.get("anc_rejected_weight", 0.0)),
        )

        # Build output dict (loss + per-sample for trainer/log)
        out = {
            "loss": total,
            "answer_length_mean": answer_length_mean,
            # V-MolPO sub-metrics (per-sample for logging)
            "v_molpo/loss_pref": v_out["loss_pref"],
            "v_molpo/loss_anchor": v_out["loss_anchor"],
            "v_molpo/margin": v_out["margin"],
            "v_molpo/margin_unclipped": v_out["margin_unclipped"],
            "v_molpo/rewards_chosen": v_out["rewards_chosen"],
            "v_molpo/rewards_rejected": v_out["rewards_rejected"],
            "v_molpo/gamma": v_out["gamma"],
            "v_molpo/avg_chosen_reward": v_out["avg_chosen_reward"],
            "v_molpo/margin_clipped_frac": v_out["margin_clipped_frac"],
            "v_molpo/elbo_theta_w_mean": elbo_theta_w.mean(),
            "v_molpo/elbo_ref_w_mean": elbo_ref_w.mean(),
            "v_molpo/elbo_theta_l_mean": elbo_theta_l.mean(),
            "v_molpo/elbo_ref_l_mean": elbo_ref_l.mean(),
        }
        # Stub per-sample fields for trainer compatibility
        device = total.device
        if per_sample_loss_sft is None:
            out["per_sample_loss"] = torch.zeros(B, device=device)
            out["per_sample_loss_no_eos"] = torch.zeros(B, device=device)
        else:
            out["per_sample_loss"] = per_sample_loss_sft
            out["per_sample_loss_no_eos"] = per_sample_loss_no_eos_sft
        return out

    def _forward_logits(self, noisy_ids, attention_mask, sub_batch):
        """Return logits only (no loss), reusing string+graph injection if available.

        Used inside _molpo_forward → vrpo_elbo.compute_elbo.
        """
        use_graph = (
            self.stage >= 2
            and self.mol_representation == "string+graph"
            and "graphs" in sub_batch
        )
        if use_graph:
            outputs = self._forward_with_graph(
                noisy_ids, attention_mask, sub_batch, sub_batch["input_ids"]
            )
        else:
            outputs = self.llada.model(
                input_ids=noisy_ids, attention_mask=attention_mask
            )
        return outputs.logits

    def _slice_batch(self, batch: dict, slc: slice) -> dict:
        """Slice all tensor fields of batch by `slc`. graphs sliced per-graph."""
        sub = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                sub[k] = v[slc]
            elif k == "tasks" and isinstance(v, list):
                sub[k] = list(v[slc])
            elif k == "graphs" and v is not None and hasattr(v, "to_data_list"):
                # PyG Batch: split → take subset → re-batch
                from torch_geometric.data import Batch
                data_list = v.to_data_list()
                sub_list = data_list[slc]
                sub[k] = Batch.from_data_list(sub_list)
            else:
                sub[k] = v
        return sub

    def _sft_forward_internal(self, sub_batch: dict) -> dict:
        """Run standard SFT forward on a batch slice. Used by mol_div=3 SFT branch."""
        input_ids = sub_batch["input_ids"]
        labels = sub_batch["labels"]
        noisy_ids, mask_indices, p_mask = self.loss_fn.make_noisy(input_ids, labels)
        attention_mask = sub_batch.get("attention_mask")
        logits = self._forward_logits(noisy_ids, attention_mask, sub_batch)
        return self.loss_fn(
            logits=logits, input_ids=input_ids, labels=labels,
            mask_indices=mask_indices, p_mask=p_mask,
            tasks=sub_batch.get("tasks"),
            global_step=sub_batch.get("global_step", 0),
            log_train_detail=False,
        )

    def _apply_stage_freeze_policy(self):
        """Stage 2: only Q-Former bridge (qformer/query_tokens/opt_proj/ln_graph) trainable.

        Stage 3 V-MolPO: LoRA + embed + lm_head trainable. tune_gnn=true 면 GNN/Q-Former 도.
        """
        if self.stage < 2:
            return

        n_trainable = n_frozen = 0
        if self.stage == 2:
            trainable_keys = STAGE2_TRAINABLE_KEYS
        else:
            # Stage 3: LoRA + embed always trainable; QFormer/GNN per cfg.model.tune_gnn
            tune_gnn = bool(self.cfg.model.get("tune_gnn", False))
            trainable_keys = STAGE3_LORA_TRAINABLE_KEYS
            if tune_gnn:
                trainable_keys = trainable_keys + STAGE3_QFORMER_TRAINABLE_KEYS

        for name, p in self.named_parameters():
            # ref_model parameters: always frozen (RefMolDA already sets requires_grad=False
            # but iteration order may overwrite — guard explicitly)
            if name.startswith("ref_model."):
                p.requires_grad = False
                n_frozen += p.numel()
                continue

            lower = name.lower()
            if any(k in lower for k in trainable_keys):
                p.requires_grad = True
                n_trainable += p.numel()
            else:
                p.requires_grad = False
                n_frozen += p.numel()
        logger.info(
            f"[Stage {self.stage}] freeze policy: trainable={n_trainable:,} "
            f"frozen={n_frozen:,}"
        )

    @torch.no_grad()
    def compute_binary_prob_likelihood(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Classification eval: likelihood scoring for True vs False.

        Masks the ENTIRE response and runs ONE forward pass per candidate.
        Reference: LLaDA paper Appendix B.5 (MMLU likelihood evaluation)

        Args:
            prompt_input_ids: [B, P] prompt token ids (left-padded)
            prompt_attention_mask: [B, P] attention mask

        Returns:
            probs: [B, 2] with [P(False), P(True)]
        """
        tokenizer = self.llada.tokenizer

        # Encode True / False responses
        true_response = "<BOOLEAN> True </BOOLEAN>"
        false_response = "<BOOLEAN> False </BOOLEAN>"
        true_ids = tokenizer.encode(true_response, add_special_tokens=False, return_tensors="pt")
        false_ids = tokenizer.encode(false_response, add_special_tokens=False, return_tensors="pt")

        true_ids = true_ids.to(prompt_input_ids.device)  # [1, T_true]
        false_ids = false_ids.to(prompt_input_ids.device)  # [1, T_false]

        true_ll = self._compute_candidate_likelihood(
            prompt_input_ids, prompt_attention_mask, true_ids
        )
        false_ll = self._compute_candidate_likelihood(
            prompt_input_ids, prompt_attention_mask, false_ids
        )

        # [B, 2]: [P(False), P(True)]
        probs = F.softmax(torch.stack([false_ll, true_ll], dim=1), dim=1)
        return probs

    def _compute_candidate_likelihood(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-likelihood of a candidate response given prompt.

        Args:
            prompt_ids: [B, P]
            prompt_mask: [B, P]
            response_ids: [1, R] candidate response tokens

        Returns:
            log_likelihood: [B] normalized log-likelihood per sample
        """
        B, P = prompt_ids.shape
        R = response_ids.shape[1]

        # Expand response for batch: [B, R]
        response_expanded = response_ids.expand(B, R)

        # Create fully masked response
        masked_response = torch.full_like(response_expanded, MASK_TOKEN_ID)

        # Concatenate: prompt + masked_response → [B, P+R]
        full_ids = torch.cat([prompt_ids, masked_response], dim=1)
        full_mask = torch.cat([
            prompt_mask,
            torch.ones(B, R, device=prompt_ids.device, dtype=prompt_mask.dtype)
        ], dim=1)

        # Forward pass
        outputs = self.llada.model(input_ids=full_ids, attention_mask=full_mask)
        logits = outputs.logits  # [B, P+R, V]

        # Extract logits at response positions
        response_logits = logits[:, P:P+R, :]  # [B, R, V]
        log_probs = F.log_softmax(response_logits, dim=-1)  # [B, R, V]

        # Gather log-prob at target token positions
        target_log_probs = log_probs.gather(
            2, response_expanded.unsqueeze(-1)
        ).squeeze(-1)  # [B, R]

        # Average over response length
        log_likelihood = target_log_probs.sum(dim=1) / R  # [B]

        return log_likelihood
