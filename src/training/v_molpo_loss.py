"""V-MolPO loss: DPO-E preference + task-adaptive anchor + (optional) anchor_rejected.

Combines:
  - VRPO (LLaDA 1.5, arXiv:2505.19223): reference-relative reward via ELBO,
    antithetic noise (handled in vrpo_elbo.py).
  - MolPO (MICCAI 26): per-task EMA anchor for task-adaptive margin γ_i.

Loss formulas (see plan/stage3.md §3 for full derivation):

  r_θ(y)    = β · (B̂_θ(y) − B̂_ref(y))           ← ELBO from vrpo_elbo
  γ_i       = molpo_lambda · |EMA_per_task[r_θ(y_w)]|
  margin    = r_θ(y_w) − r_θ(y_l)
  margin*   = clip(margin, ±margin_clip_scale·|r_θ(y_w)|)   if burn-in active
  L_pref    = −logσ(margin* − γ_i).mean()
  L_anchor  = −logσ(−(r_θ(y_l) − rejected_lambda · EMA)).mean()   ← optional

The trainer combines:
  L_total   = sft_weight · L_SFT + molpo_weight · L_pref + anc_rejected_weight · L_anchor

ELBO inputs are computed by src/training/vrpo_elbo.py compute_elbo (n_t-MC + antithetic).
Caller is responsible for routing chosen/rejected slices through θ and ref policies.
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Per-task EMA anchor
# ─────────────────────────────────────────────────────────────────

class TaskAnchorEMA:
    """Per-task exponential moving average of chosen rewards.

    Used to compute task-adaptive margin γ_i = molpo_lambda · |E[r_w, task]|.
    EMA is maintained separately per task (e.g., chebi-20-mol2text vs property-pred)
    so that reward magnitude differences between tasks don't bias the margin.

    State_dict round-trip supported via state_dict() / load_state_dict() so the
    anchor survives checkpoint reloads (CheckpointMixin patches this in).
    """

    def __init__(self, alpha: float = 0.99):
        assert 0 < alpha < 1, f"EMA alpha must be in (0,1), got {alpha}"
        self.alpha = alpha
        self._values: dict[str, float] = {}

    def update(self, rewards: torch.Tensor, tasks: list[str]) -> None:
        """Update EMA with new rewards. rewards shape [B], tasks list len B."""
        if rewards.numel() == 0:
            return
        flat = rewards.detach().float().cpu().tolist()
        for r, t in zip(flat, tasks):
            if t in self._values:
                self._values[t] = self.alpha * self._values[t] + (1 - self.alpha) * r
            else:
                # First observation: bootstrap with the value itself
                self._values[t] = float(r)

    def get(self, tasks: list[str], default: float = 0.0,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Return tensor [len(tasks)] of EMA values (using `default` for unseen)."""
        vals = [self._values.get(t, default) for t in tasks]
        return torch.tensor(vals, device=device, dtype=dtype or torch.float32)

    def state_dict(self) -> dict:
        return {"alpha": self.alpha, "values": dict(self._values)}

    def load_state_dict(self, state: dict) -> None:
        self.alpha = float(state.get("alpha", self.alpha))
        self._values = dict(state.get("values", {}))

    def reset(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)


# ─────────────────────────────────────────────────────────────────
# V-MolPO loss
# ─────────────────────────────────────────────────────────────────

def compute_v_molpo_loss(
    elbo_theta_w: torch.Tensor,    # [B]
    elbo_ref_w: torch.Tensor,      # [B] (no_grad)
    elbo_theta_l: torch.Tensor,    # [B]
    elbo_ref_l: torch.Tensor,      # [B] (no_grad)
    tasks_chosen: list[str],       # len B
    task_anchor_ema: TaskAnchorEMA,
    *,
    beta: float = 0.1,
    molpo_lambda: float = 0.5,
    margin_clip_scale: float = 1.0,
    margin_clip_active: bool = False,
    anc_rejected_weight: float = 0.0,
    rejected_lambda: float = 1.5,
    loss_type: str = "sigmoid",
) -> dict:
    """Compute V-MolPO preference + anchor losses for one batch.

    Args:
        elbo_*       : per-sample log-likelihood approximations from vrpo_elbo.compute_elbo
        tasks_chosen : per-sample task names (used as EMA bucket key)
        task_anchor_ema : EMA tracker (mutated in place — caller owns)
        beta            : reward temperature
        molpo_lambda    : task-adaptive margin coefficient
        margin_clip_*   : clip during burn-in for stability
        anc_rejected_weight : if >0, include L_anchor on rejected
        rejected_lambda : anchor scaling for rejected
        loss_type    : "sigmoid" (default DPO) — extension point

    Returns:
        dict with:
            loss_pref          [scalar]  -logσ(margin* − γ_i).mean()
            loss_anchor        [scalar]  anchor_rejected loss (0 if anc_w=0)
            margin             [B]       r_w − r_l (post-clip)
            margin_unclipped   [B]       r_w − r_l (raw)
            rewards_chosen     [B]       r_θ(y_w)
            rewards_rejected   [B]       r_θ(y_l)
            gamma              [B]       γ_i per sample (task-adaptive)
            avg_chosen_reward  [B]       EMA value per task (post-update)
            margin_clipped_frac [scalar] fraction of samples whose margin was clipped
    """
    assert elbo_theta_w.shape == elbo_ref_w.shape == elbo_theta_l.shape == elbo_ref_l.shape
    assert elbo_theta_w.dim() == 1
    assert len(tasks_chosen) == elbo_theta_w.shape[0]

    # Reference-relative reward
    r_w = beta * (elbo_theta_w - elbo_ref_w)
    r_l = beta * (elbo_theta_l - elbo_ref_l)

    # Update EMA on this batch's chosen rewards (detached)
    task_anchor_ema.update(r_w, tasks_chosen)

    # Task-adaptive γ_i
    avg_r_w = task_anchor_ema.get(
        tasks_chosen, default=0.0, device=r_w.device, dtype=r_w.dtype
    )
    gamma_i = molpo_lambda * avg_r_w.abs()

    # Margin (raw)
    margin_raw = r_w - r_l

    # Optional burn-in clipping
    if margin_clip_active and margin_clip_scale > 0:
        clip_thresh = margin_clip_scale * r_w.detach().abs()
        margin = torch.clamp(margin_raw, min=-clip_thresh, max=clip_thresh)
        clipped = (margin != margin_raw).float().mean().item()
    else:
        margin = margin_raw
        clipped = 0.0

    # Preference loss — keep [B] per-sample form before mean so the trainer
    # can slice it by task. Scalar L_pref is still used for backprop / global
    # logging; per-sample form goes to per-task aggregation.
    if loss_type == "sigmoid":
        loss_pref_per_sample = -F.logsigmoid(margin - gamma_i)  # [B]
    elif loss_type == "hinge":
        loss_pref_per_sample = F.relu(1.0 - (margin - gamma_i))  # [B]
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    L_pref = loss_pref_per_sample.mean()

    # Anchor on rejected (optional)
    if anc_rejected_weight > 0.0:
        loss_anchor_per_sample = -F.logsigmoid(-(r_l - rejected_lambda * avg_r_w))  # [B]
        L_anchor = loss_anchor_per_sample.mean()
    else:
        loss_anchor_per_sample = torch.zeros_like(r_w)
        L_anchor = torch.zeros((), device=r_w.device, dtype=r_w.dtype)

    # GDR (Generation Direction Ratio / rewards_accuracies):
    # per-sample binary indicator that chosen reward exceeds rejected reward.
    # Old_MolDA / mol-llm_official compute this as `metrics[rewards/accuracies]`.
    # Trainer averages it over batch & per-task to log fraction of correctly
    # ordered preference pairs — the most direct DPO health metric.
    rewards_accuracies = (margin_raw > 0).float()  # [B]

    return {
        "loss_pref": L_pref,
        "loss_anchor": L_anchor,
        "loss_pref_per_sample": loss_pref_per_sample,        # [B]  for per-task aggregation
        "loss_anchor_per_sample": loss_anchor_per_sample,    # [B]  for per-task aggregation
        "margin": margin,
        "margin_unclipped": margin_raw,
        "rewards_chosen": r_w,
        "rewards_rejected": r_l,
        "rewards_accuracies": rewards_accuracies,            # [B]  GDR — chosen > rejected indicator
        "gamma": gamma_i,
        "avg_chosen_reward": avg_r_w,
        "margin_clipped_frac": clipped,
    }


# ─────────────────────────────────────────────────────────────────
# Convenience: combine SFT + V-MolPO into total loss
# ─────────────────────────────────────────────────────────────────

def combine_total_loss(
    loss_sft: torch.Tensor | None,
    v_molpo_out: dict,
    *,
    sft_weight: float = 1.0,
    molpo_weight: float = 0.25,
    anc_rejected_weight: float = 0.0,
) -> torch.Tensor:
    """L_total = sft_w·L_SFT + molpo_w·L_pref + anc_w·L_anchor.

    Args:
        loss_sft : MaskedDiffusionLoss output [scalar] or None (no SFT branch)
        v_molpo_out : dict from compute_v_molpo_loss
    """
    L_pref = v_molpo_out["loss_pref"]
    L_anchor = v_molpo_out["loss_anchor"]

    total = molpo_weight * L_pref + anc_rejected_weight * L_anchor
    if loss_sft is not None and sft_weight > 0.0:
        total = total + sft_weight * loss_sft
    return total
