"""
MolDA: unified model combining LLaDA + (optional) GNN + Q-Former.

Stage 1: LLaDA-only (string_only), LoRA + embed + head trainable
Stage 2+: + GNN + Q-Former (to be implemented)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.llada_wrapper import LLaDAWrapper
from src.training.loss import MaskedDiffusionLoss, MASK_TOKEN_ID

logger = logging.getLogger(__name__)


class MolDA(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stage = cfg.stage

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

        # Stage 2+: GNN + Q-Former (stubs for now)
        if self.stage >= 2:
            from src.model.gnn import GINETokenGT
            from src.model.qformer import QFormer
            self.gnn = GINETokenGT(cfg)
            self.qformer = QFormer(cfg)

    @property
    def tokenizer(self):
        return self.llada.tokenizer

    def forward(self, batch: dict) -> dict:
        """Training forward pass with masked diffusion loss.

        Args:
            batch: dict with input_ids [B,L], labels [B,L], prompt_lengths [B], tasks [B]

        Returns:
            dict with "loss", "answer_length_mean"
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # 1. Forward process: mask answer tokens
        noisy_ids, mask_indices, p_mask = self.loss_fn.make_noisy(input_ids, labels)

        # 2. Model forward
        attention_mask = batch.get("attention_mask", None)
        outputs = self.llada.model(
            input_ids=noisy_ids,
            attention_mask=attention_mask,
        )
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
