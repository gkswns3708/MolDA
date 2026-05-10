"""
LLaDA Masked Diffusion SFT Loss.

Reference: official_LLaDA/GUIDELINES.md (SFT section)
- t ~ Uniform(0,1) per sample
- p_mask = (1-eps)*t + eps
- answer 위치에만 masking (prompt 원본 유지)
- loss = CE / p_mask / answer_length, averaged over batch
"""

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 126336  # LLaDA <|mdm_mask|>
EPS = 1e-3


class MaskedDiffusionLoss(nn.Module):

    def __init__(self, mask_token_id: int = MASK_TOKEN_ID, eps: float = EPS,
                 log_nan: bool = True, nan_log_dir: str = "./nan_logs",
                 eos_token_id: int = None, normalization: str = "global"):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id
        self.eps = eps
        self.normalization = normalization  # "global" | "per_sample"
        self.log_nan = log_nan
        self.nan_log_dir = nan_log_dir

    def make_noisy(self, input_ids: torch.Tensor, labels: torch.Tensor,
                   t_override: torch.Tensor | None = None,
                   mask_indices_override: torch.Tensor | None = None):
        """Forward process: mask answer tokens with probability p_mask.

        Args:
            input_ids: [B, L] original token ids
            labels: [B, L] with -100 for prompt positions
            t_override: [B] optional pre-sampled timesteps (for V-MolPO antithetic).
                When provided, skip internal `torch.rand(B)` call. Mutually exclusive
                with mask_indices_override only by purpose — both can be supplied.
            mask_indices_override: [B, L] optional pre-computed boolean mask
                (already AND-ed with answer_mask, ≥1 mask guarantee applied externally).
                When provided, skip internal mask sampling.

        Returns:
            noisy_ids: [B, L] with MASK tokens in answer positions
            mask_indices: [B, L] boolean mask of actually masked positions
            p_mask: [B, 1] masking probability per sample
        """
        b, l = input_ids.shape
        device = input_ids.device

        # t ~ Uniform(0,1), p_mask = (1-eps)*t + eps
        if t_override is not None:
            assert t_override.shape == (b,), (
                f"t_override must be [B={b}], got {tuple(t_override.shape)}"
            )
            t = t_override.to(device)
        else:
            t = torch.rand(b, device=device)
        p_mask = (1 - self.eps) * t + self.eps  # [B]

        # Only mask answer positions (labels != -100)
        answer_mask = (labels != -100)  # [B, L]

        if mask_indices_override is not None:
            assert mask_indices_override.shape == (b, l), (
                f"mask_indices_override must be [B,L]=[{b},{l}], "
                f"got {tuple(mask_indices_override.shape)}"
            )
            mask_indices = mask_indices_override.to(device).bool()
        else:
            p_mask_expanded = p_mask[:, None].expand(b, l)  # [B, L]
            rand_mask = torch.rand((b, l), device=device) < p_mask_expanded
            mask_indices = rand_mask & answer_mask

            # Guarantee >= 1 masked token per sample
            for i in range(b):
                if mask_indices[i].sum() == 0 and answer_mask[i].sum() > 0:
                    answer_positions = answer_mask[i].nonzero(as_tuple=False).squeeze(-1)
                    rand_idx = answer_positions[torch.randint(len(answer_positions), (1,))]
                    mask_indices[i, rand_idx] = True

        noisy_ids = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_ids, mask_indices, p_mask[:, None]  # p_mask: [B, 1]

    def forward(self, logits: torch.Tensor, input_ids: torch.Tensor,
                labels: torch.Tensor, mask_indices: torch.Tensor,
                p_mask: torch.Tensor, tasks=None, global_step: int = 0,
                log_train_detail: bool = False) -> dict:
        """Compute masked diffusion SFT loss.

        Args:
            logits: [B, L, V] model output logits
            input_ids: [B, L] original (clean) token ids
            labels: [B, L] with -100 for prompt positions
            mask_indices: [B, L] boolean, which positions were masked
            p_mask: [B, 1] masking probability per sample
            tasks: optional list of task names for NaN logging
            global_step: current training step for NaN logging

        Returns:
            dict with "loss", "answer_length_mean",
            "per_sample_loss" [B], "per_sample_loss_no_eos" [B]
        """
        b, l = input_ids.shape
        device = input_ids.device

        answer_mask = (labels != -100)  # [B, L]
        answer_lengths = answer_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
        answer_lengths_expanded = answer_lengths.expand(b, l)  # [B, L]

        # Token-level CE loss only at masked positions
        # logits[mask_indices]: [N_masked, V], input_ids[mask_indices]: [N_masked]
        token_loss = F.cross_entropy(
            logits[mask_indices], input_ids[mask_indices], reduction='none'
        )  # [N_masked]

        # Importance weighting: / p_mask per token
        p_mask_expanded = p_mask.expand(b, l)  # [B, 1] → [B, L]
        token_loss = token_loss / p_mask_expanded[mask_indices]

        # Loss normalization
        if self.normalization == "global":
            # Old_MolDA/SMDM 방식: 배치 전체 answer 토큰 수로 나눔
            total_answer_length = answer_lengths.sum()
            ce_loss = token_loss.sum() / (total_answer_length + 1e-8)
        else:
            # Per-sample: 각 샘플 answer_length로 나눈 뒤 배치 평균
            ce_loss = torch.sum(
                token_loss / answer_lengths_expanded[mask_indices]
            ) / b

        # NaN guard
        if ce_loss.isnan() and self.log_nan:
            self._log_nan(input_ids, labels, mask_indices, tasks, global_step)

        # --- Per-sample & no-EOS metrics (detached, for logging only) ---
        with torch.no_grad():
            # Scatter importance-weighted token losses to [B, L]
            weighted = torch.zeros(b, l, device=device)
            weighted[mask_indices] = token_loss.detach()

            per_sample_loss = weighted.sum(dim=1) / answer_lengths.squeeze(1)  # [B]

            # EOS mask: answer region 내 모든 EOS 토큰 위치 (eos_token_id=None이면 EOS 없음)
            if self.eos_token_id is not None:
                eos_mask = (input_ids == self.eos_token_id) & answer_mask  # [B, L]
            else:
                eos_mask = torch.zeros_like(answer_mask, dtype=torch.bool)
            eos_counts = eos_mask.sum(dim=1)  # [B]

            # Loss excluding all EOS tokens, re-normalized by content length
            no_eos_weighted = weighted.clone()
            no_eos_weighted[eos_mask] = 0.0
            content_lengths = (answer_lengths.squeeze(1) - eos_counts).clamp(min=1)
            per_sample_loss_no_eos = no_eos_weighted.sum(dim=1) / content_lengths  # [B]

            # --- Prediction quality metrics at masked positions ---
            masked_logits = logits[mask_indices]          # [N_masked, V]
            masked_targets = input_ids[mask_indices]      # [N_masked]
            pred_tokens = masked_logits.argmax(dim=-1)    # [N_masked]

            mask_accuracy = (pred_tokens == masked_targets).float().mean().item()

            # EOS 제외 accuracy (eos_token_id=None이면 모든 위치를 non-EOS로 간주)
            if self.eos_token_id is not None:
                eos_at_masked = (masked_targets == self.eos_token_id)
            else:
                eos_at_masked = torch.zeros_like(masked_targets, dtype=torch.bool)
            non_eos_mask = ~eos_at_masked
            if non_eos_mask.any():
                mask_accuracy_no_eos = (pred_tokens[non_eos_mask] == masked_targets[non_eos_mask]).float().mean().item()
            else:
                mask_accuracy_no_eos = 0.0

            # Per-sample accuracy: scatter correctness back to [B, L] then aggregate per row
            per_token_correct = (pred_tokens == masked_targets).float()  # [N_masked]
            correct_bl = torch.zeros(b, l, device=device)
            correct_bl[mask_indices] = per_token_correct
            n_masked_per_sample = mask_indices.sum(dim=1).clamp(min=1)   # [B]
            per_sample_mask_accuracy = correct_bl.sum(dim=1) / n_masked_per_sample

            # Per-sample accuracy excluding EOS targets
            non_eos_bl = torch.zeros(b, l, dtype=torch.bool, device=device)
            non_eos_bl[mask_indices] = non_eos_mask
            correct_no_eos_bl = correct_bl * non_eos_bl.float()
            n_non_eos_per_sample = non_eos_bl.sum(dim=1).clamp(min=1)    # [B]
            per_sample_mask_accuracy_no_eos = correct_no_eos_bl.sum(dim=1) / n_non_eos_per_sample

            log_probs = F.log_softmax(masked_logits, dim=-1)
            target_log_probs = log_probs.gather(
                1, masked_targets.unsqueeze(1)
            ).squeeze(1)
            target_prob_mean = target_log_probs.exp().mean().item()

        result = {
            "loss": ce_loss,
            "answer_length_mean": answer_lengths.mean().item(),
            "per_sample_loss": per_sample_loss,
            "per_sample_loss_no_eos": per_sample_loss_no_eos,
            "per_sample_mask_accuracy": per_sample_mask_accuracy,
            "per_sample_mask_accuracy_no_eos": per_sample_mask_accuracy_no_eos,
            "p_mask_per_sample": p_mask.squeeze(1),
            "mask_accuracy": mask_accuracy,
            "mask_accuracy_no_eos": mask_accuracy_no_eos,
            "target_prob_mean": target_prob_mean,
        }

        # --- Detailed sample data (conditional, first sample only) ---
        if log_train_detail:
            with torch.no_grad():
                s_mask = mask_indices[0]                       # [L] bool
                s_logits = logits[0, s_mask]                   # [N_i, V]
                s_targets = input_ids[0, s_mask]               # [N_i]
                s_preds = s_logits.argmax(dim=-1)              # [N_i]
                s_probs = F.softmax(s_logits, dim=-1)          # [N_i, V]

                result["_train_sample_detail"] = {
                    "mask_positions": s_mask.nonzero().squeeze(-1).cpu(),
                    "target_tokens": s_targets.cpu(),
                    "pred_tokens": s_preds.cpu(),
                    "target_probs": s_probs.gather(
                        1, s_targets.unsqueeze(1)
                    ).squeeze(1).cpu(),
                    "pred_probs": s_probs.gather(
                        1, s_preds.unsqueeze(1)
                    ).squeeze(1).cpu(),
                    "p_mask": p_mask[0].item(),
                }

                # Full-sequence data for human-readable text logging
                s_answer_mask = answer_mask[0]                          # [L] bool
                s_all_pred_ids = logits[0].argmax(dim=-1)               # [L]
                result["_train_sample_detail"]["input_ids"] = input_ids[0].cpu()
                result["_train_sample_detail"]["labels"] = labels[0].cpu()
                result["_train_sample_detail"]["all_answer_pred_ids"] = s_all_pred_ids[s_answer_mask].cpu()
                result["_train_sample_detail"]["all_answer_gt_ids"] = input_ids[0, s_answer_mask].cpu()

        return result

    def _log_nan(self, input_ids, labels, mask_indices, tasks, global_step):
        logger.warning(f"NaN loss at step {global_step}")
        if self.nan_log_dir:
            Path(self.nan_log_dir).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(self.nan_log_dir, f"nan_step{global_step}.pt")
            torch.save({
                "input_ids": input_ids.cpu(),
                "labels": labels.cpu(),
                "mask_indices": mask_indices.cpu(),
                "tasks": tasks,
                "global_step": global_step,
            }, save_path)
            logger.warning(f"NaN details saved to {save_path}")
