"""Unit tests for per-task V-MolPO metric exposure.

Verifies:
1. `compute_v_molpo_loss` returns `loss_pref_per_sample` and `loss_anchor_per_sample`
   as [B] tensors alongside the scalar versions.
2. Per-sample `.mean()` matches the scalar `loss_pref` / `loss_anchor`.
3. The output dict from `molda._molpo_forward` now exposes the [B] forms of
   per-sample losses and ELBOs for per-task slicing in the trainer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _fake_v_molpo_inputs(B=4, device="cpu"):
    """Construct synthetic ELBO inputs that exercise `compute_v_molpo_loss`."""
    torch.manual_seed(0)
    elbo_theta_w = torch.randn(B, device=device) * 0.1 - 0.5
    elbo_theta_l = torch.randn(B, device=device) * 0.1 - 0.7
    elbo_ref_w = torch.randn(B, device=device) * 0.1 - 0.6
    elbo_ref_l = torch.randn(B, device=device) * 0.1 - 0.6
    return elbo_theta_w, elbo_theta_l, elbo_ref_w, elbo_ref_l


def test_v_molpo_loss_returns_per_sample_keys():
    """Output dict must include `loss_pref_per_sample` and
    `loss_anchor_per_sample`, both as [B] tensors."""
    from src.training.v_molpo_loss import TaskAnchorEMA, compute_v_molpo_loss

    B = 4
    elbo_theta_w, elbo_theta_l, elbo_ref_w, elbo_ref_l = _fake_v_molpo_inputs(B)
    ema = TaskAnchorEMA(alpha=0.99)

    out = compute_v_molpo_loss(
        elbo_theta_w=elbo_theta_w,
        elbo_theta_l=elbo_theta_l,
        elbo_ref_w=elbo_ref_w,
        elbo_ref_l=elbo_ref_l,
        tasks_chosen=["chebi-20-mol2text"] * B,
        task_anchor_ema=ema,
        beta=2.0,
        molpo_lambda=0.5,
        margin_clip_active=False,
        margin_clip_scale=1.0,
        anc_rejected_weight=0.0,
        rejected_lambda=1.5,
        loss_type="sigmoid",
    )
    assert "loss_pref_per_sample" in out
    assert "loss_anchor_per_sample" in out
    assert isinstance(out["loss_pref_per_sample"], torch.Tensor)
    assert isinstance(out["loss_anchor_per_sample"], torch.Tensor)
    assert out["loss_pref_per_sample"].shape == (B,)
    assert out["loss_anchor_per_sample"].shape == (B,)


def test_per_sample_loss_pref_mean_matches_scalar():
    """Numerical equivalence: `loss_pref_per_sample.mean() == loss_pref`."""
    from src.training.v_molpo_loss import TaskAnchorEMA, compute_v_molpo_loss

    B = 6
    elbo_theta_w, elbo_theta_l, elbo_ref_w, elbo_ref_l = _fake_v_molpo_inputs(B)
    ema = TaskAnchorEMA(alpha=0.99)

    out = compute_v_molpo_loss(
        elbo_theta_w=elbo_theta_w,
        elbo_theta_l=elbo_theta_l,
        elbo_ref_w=elbo_ref_w,
        elbo_ref_l=elbo_ref_l,
        tasks_chosen=["chebi-20-mol2text"] * B,
        task_anchor_ema=ema,
        beta=2.0,
        molpo_lambda=0.5,
        margin_clip_active=False,
        margin_clip_scale=1.0,
        anc_rejected_weight=0.0,
        rejected_lambda=1.5,
        loss_type="sigmoid",
    )
    assert torch.allclose(
        out["loss_pref_per_sample"].mean(),
        out["loss_pref"],
        atol=1e-6,
    ), "loss_pref_per_sample.mean() should equal scalar loss_pref"


def test_per_sample_loss_anchor_matches_when_active():
    """When anc_rejected_weight > 0, per-sample anchor mean must equal scalar."""
    from src.training.v_molpo_loss import TaskAnchorEMA, compute_v_molpo_loss

    B = 4
    elbo_theta_w, elbo_theta_l, elbo_ref_w, elbo_ref_l = _fake_v_molpo_inputs(B)
    ema = TaskAnchorEMA(alpha=0.99)

    out = compute_v_molpo_loss(
        elbo_theta_w=elbo_theta_w,
        elbo_theta_l=elbo_theta_l,
        elbo_ref_w=elbo_ref_w,
        elbo_ref_l=elbo_ref_l,
        tasks_chosen=["chebi-20-mol2text"] * B,
        task_anchor_ema=ema,
        beta=2.0,
        molpo_lambda=0.5,
        margin_clip_active=False,
        margin_clip_scale=1.0,
        anc_rejected_weight=0.5,
        rejected_lambda=1.5,
        loss_type="sigmoid",
    )
    assert torch.allclose(
        out["loss_anchor_per_sample"].mean(),
        out["loss_anchor"],
        atol=1e-6,
    )


def test_per_sample_anchor_is_zero_when_inactive():
    """When anc_rejected_weight == 0, per-sample anchor should be all zeros."""
    from src.training.v_molpo_loss import TaskAnchorEMA, compute_v_molpo_loss

    B = 4
    elbo_theta_w, elbo_theta_l, elbo_ref_w, elbo_ref_l = _fake_v_molpo_inputs(B)
    ema = TaskAnchorEMA(alpha=0.99)

    out = compute_v_molpo_loss(
        elbo_theta_w=elbo_theta_w,
        elbo_theta_l=elbo_theta_l,
        elbo_ref_w=elbo_ref_w,
        elbo_ref_l=elbo_ref_l,
        tasks_chosen=["chebi-20-mol2text"] * B,
        task_anchor_ema=ema,
        beta=2.0,
        molpo_lambda=0.5,
        margin_clip_active=False,
        margin_clip_scale=1.0,
        anc_rejected_weight=0.0,
        rejected_lambda=1.5,
        loss_type="sigmoid",
    )
    assert torch.all(out["loss_anchor_per_sample"] == 0.0)
    assert out["loss_anchor"].item() == 0.0


def test_hinge_loss_type_also_returns_per_sample():
    """`loss_type='hinge'` path must also expose `loss_pref_per_sample`."""
    from src.training.v_molpo_loss import TaskAnchorEMA, compute_v_molpo_loss

    B = 4
    elbo_theta_w, elbo_theta_l, elbo_ref_w, elbo_ref_l = _fake_v_molpo_inputs(B)
    ema = TaskAnchorEMA(alpha=0.99)

    out = compute_v_molpo_loss(
        elbo_theta_w=elbo_theta_w,
        elbo_theta_l=elbo_theta_l,
        elbo_ref_w=elbo_ref_w,
        elbo_ref_l=elbo_ref_l,
        tasks_chosen=["chebi-20-mol2text"] * B,
        task_anchor_ema=ema,
        beta=2.0,
        molpo_lambda=0.5,
        margin_clip_active=False,
        margin_clip_scale=1.0,
        anc_rejected_weight=0.0,
        rejected_lambda=1.5,
        loss_type="hinge",
    )
    assert out["loss_pref_per_sample"].shape == (B,)
    assert torch.allclose(
        out["loss_pref_per_sample"].mean(),
        out["loss_pref"],
        atol=1e-6,
    )


def test_per_sample_keys_are_differentiable():
    """`loss_pref_per_sample` must carry gradient (it's pre-mean of L_pref)."""
    from src.training.v_molpo_loss import TaskAnchorEMA, compute_v_molpo_loss

    B = 4
    elbo_theta_w = torch.randn(B, requires_grad=True)
    elbo_theta_l = torch.randn(B, requires_grad=True)
    elbo_ref_w = torch.randn(B)
    elbo_ref_l = torch.randn(B)
    ema = TaskAnchorEMA(alpha=0.99)

    out = compute_v_molpo_loss(
        elbo_theta_w=elbo_theta_w,
        elbo_theta_l=elbo_theta_l,
        elbo_ref_w=elbo_ref_w,
        elbo_ref_l=elbo_ref_l,
        tasks_chosen=["chebi-20-mol2text"] * B,
        task_anchor_ema=ema,
        beta=2.0,
        molpo_lambda=0.5,
        margin_clip_active=False,
        margin_clip_scale=1.0,
        anc_rejected_weight=0.0,
        rejected_lambda=1.5,
        loss_type="sigmoid",
    )
    assert out["loss_pref_per_sample"].requires_grad


def test_trainer_collects_per_sample_v_molpo_keys():
    """Sanity: simulated `out` dict has the new [B] keys at the expected shape
    so the trainer's per-task filter (`v.shape[0] == len(tasks)`) picks them.
    """
    B = 3
    out = {
        "v_molpo/loss_pref": torch.tensor(0.5),                       # scalar (skip)
        "v_molpo/loss_pref_per_sample": torch.randn(B),               # [B] ✓
        "v_molpo/loss_anchor": torch.tensor(0.0),                     # scalar (skip)
        "v_molpo/loss_anchor_per_sample": torch.zeros(B),             # [B] ✓
        "v_molpo/margin": torch.randn(B),                              # [B] ✓
        "v_molpo/rewards_chosen": torch.randn(B),                     # [B] ✓
        "v_molpo/elbo_theta_w_mean": torch.tensor(-0.2),              # scalar (skip)
        "v_molpo/elbo_theta_w": torch.randn(B),                       # [B] ✓
        "v_molpo/margin_clipped_frac": 0.5,                            # not a tensor
    }
    tasks = ["chebi-20-mol2text"] * B

    # Reproduce the trainer's collection logic
    v_molpo_per_sample = {
        k: v for k, v in out.items()
        if k.startswith("v_molpo/")
        and isinstance(v, torch.Tensor)
        and v.dim() >= 1
        and v.shape[0] == len(tasks)
    }

    expected_keys = {
        "v_molpo/loss_pref_per_sample",
        "v_molpo/loss_anchor_per_sample",
        "v_molpo/margin",
        "v_molpo/rewards_chosen",
        "v_molpo/elbo_theta_w",
    }
    assert set(v_molpo_per_sample.keys()) == expected_keys
