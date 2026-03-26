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
                 log_nan: bool = True, nan_log_dir: str = "./nan_logs"):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.eps = eps
        self.log_nan = log_nan
        self.nan_log_dir = nan_log_dir

    def make_noisy(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """Forward process: mask answer tokens with probability p_mask.

        Args:
            input_ids: [B, L] original token ids
            labels: [B, L] with -100 for prompt positions

        Returns:
            noisy_ids: [B, L] with MASK tokens in answer positions
            mask_indices: [B, L] boolean mask of actually masked positions
            p_mask: [B, 1] masking probability per sample
        """
        b, l = input_ids.shape
        device = input_ids.device

        # t ~ Uniform(0,1), p_mask = (1-eps)*t + eps
        t = torch.rand(b, device=device)
        p_mask = (1 - self.eps) * t + self.eps  # [B]
        p_mask_expanded = p_mask[:, None].expand(b, l)  # [B, L]

        # Only mask answer positions (labels != -100)
        answer_mask = (labels != -100)  # [B, L]
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
                p_mask: torch.Tensor, tasks=None, global_step: int = 0) -> dict:
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
            dict with "loss" and "answer_length_mean"
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

        # Per-sample normalization: / answer_length
        ce_loss = torch.sum(
            token_loss / answer_lengths_expanded[mask_indices]
        ) / b

        # NaN guard
        if ce_loss.isnan() and self.log_nan:
            self._log_nan(input_ids, labels, mask_indices, tasks, global_step)

        return {
            "loss": ce_loss,
            "answer_length_mean": answer_lengths.mean().item(),
        }

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
