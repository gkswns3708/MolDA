"""Tests for src/training/v_molpo_loss.py.

Validates:
  - πθ = πref → margin = 0 (regardless of which y is "chosen")
  - Swap chosen ↔ rejected → margin sign flip
  - TaskAnchorEMA: per-task tracking, state_dict round-trip
  - margin_clip burn-in: clipped vs unclipped
  - anc_rejected_weight=0 → L_anchor = 0 (no contribution)
  - combine_total_loss arithmetic
"""
import torch
import pytest

from src.training.v_molpo_loss import (
    TaskAnchorEMA,
    compute_v_molpo_loss,
    combine_total_loss,
)


# ─────────────────────────────────────────────────────────────────
# TaskAnchorEMA
# ─────────────────────────────────────────────────────────────────

class TestTaskAnchorEMA:

    def test_first_observation_bootstraps(self):
        ema = TaskAnchorEMA(alpha=0.99)
        ema.update(torch.tensor([2.0, 3.0]), ["a", "b"])
        # First observation = value itself (no smoothing)
        vals = ema.get(["a", "b"])
        assert torch.allclose(vals, torch.tensor([2.0, 3.0]))

    def test_subsequent_smoothing(self):
        ema = TaskAnchorEMA(alpha=0.9)
        ema.update(torch.tensor([1.0]), ["a"])  # bootstrap to 1.0
        ema.update(torch.tensor([3.0]), ["a"])  # 0.9*1 + 0.1*3 = 1.2
        vals = ema.get(["a"])
        assert torch.allclose(vals, torch.tensor([1.2]), atol=1e-6)

    def test_per_task_independence(self):
        ema = TaskAnchorEMA(alpha=0.99)
        ema.update(torch.tensor([10.0, 100.0]), ["task_X", "task_Y"])
        # Tasks tracked independently
        v = ema.get(["task_X", "task_Y", "task_Z"])
        assert v[0].item() == 10.0
        assert v[1].item() == 100.0
        assert v[2].item() == 0.0  # unseen → default

    def test_state_dict_roundtrip(self):
        ema = TaskAnchorEMA(alpha=0.95)
        ema.update(torch.tensor([1.5, 2.5]), ["a", "b"])
        sd = ema.state_dict()

        ema2 = TaskAnchorEMA(alpha=0.99)  # different default alpha
        ema2.load_state_dict(sd)
        assert ema2.alpha == 0.95
        v = ema2.get(["a", "b"])
        assert torch.allclose(v, torch.tensor([1.5, 2.5]))

    def test_reset(self):
        ema = TaskAnchorEMA(alpha=0.99)
        ema.update(torch.tensor([1.0]), ["a"])
        assert len(ema) == 1
        ema.reset()
        assert len(ema) == 0


# ─────────────────────────────────────────────────────────────────
# compute_v_molpo_loss
# ─────────────────────────────────────────────────────────────────

class TestVMolPOLoss:

    def test_theta_equals_ref_margin_zero(self):
        """If πθ outputs match πref outputs, r_w=r_l=0 → margin=0."""
        B = 4
        elbo = torch.tensor([-1.5, -2.0, -1.8, -1.2])
        ema = TaskAnchorEMA()
        out = compute_v_molpo_loss(
            elbo_theta_w=elbo, elbo_ref_w=elbo.clone(),
            elbo_theta_l=elbo, elbo_ref_l=elbo.clone(),
            tasks_chosen=["a"] * B,
            task_anchor_ema=ema, beta=0.1,
        )
        assert torch.allclose(out["margin"], torch.zeros(B))
        assert torch.allclose(out["rewards_chosen"], torch.zeros(B))
        assert torch.allclose(out["rewards_rejected"], torch.zeros(B))

    def test_chosen_better_than_rejected_positive_margin(self):
        """If πθ assigns higher likelihood to y_w than y_l (relative to ref) → margin > 0."""
        # π_θ assigns -1.0 to y_w (better), -3.0 to y_l (worse)
        # π_ref same for both
        ema = TaskAnchorEMA()
        out = compute_v_molpo_loss(
            elbo_theta_w=torch.tensor([-1.0]),
            elbo_ref_w=torch.tensor([-2.0]),
            elbo_theta_l=torch.tensor([-3.0]),
            elbo_ref_l=torch.tensor([-2.0]),
            tasks_chosen=["a"], task_anchor_ema=ema, beta=1.0,
        )
        # r_w = 1·(-1 - (-2)) = 1.0; r_l = 1·(-3 - (-2)) = -1.0
        # margin = r_w - r_l = 2.0 (positive)
        assert out["margin"].item() == pytest.approx(2.0)

    def test_swap_inverts_margin_sign(self):
        ema = TaskAnchorEMA()
        out_orig = compute_v_molpo_loss(
            elbo_theta_w=torch.tensor([-1.0]),
            elbo_ref_w=torch.tensor([-2.0]),
            elbo_theta_l=torch.tensor([-3.0]),
            elbo_ref_l=torch.tensor([-2.0]),
            tasks_chosen=["a"], task_anchor_ema=ema, beta=1.0,
        )
        ema2 = TaskAnchorEMA()
        out_swap = compute_v_molpo_loss(
            elbo_theta_w=torch.tensor([-3.0]),
            elbo_ref_w=torch.tensor([-2.0]),
            elbo_theta_l=torch.tensor([-1.0]),
            elbo_ref_l=torch.tensor([-2.0]),
            tasks_chosen=["a"], task_anchor_ema=ema2, beta=1.0,
        )
        assert torch.allclose(out_orig["margin"], -out_swap["margin"])

    def test_anc_rejected_zero_when_weight_zero(self):
        ema = TaskAnchorEMA()
        out = compute_v_molpo_loss(
            elbo_theta_w=torch.tensor([-1.0]),
            elbo_ref_w=torch.tensor([-2.0]),
            elbo_theta_l=torch.tensor([-3.0]),
            elbo_ref_l=torch.tensor([-2.0]),
            tasks_chosen=["a"], task_anchor_ema=ema, beta=1.0,
            anc_rejected_weight=0.0,
        )
        assert out["loss_anchor"].item() == 0.0

    def test_anc_rejected_nonzero_when_active(self):
        ema = TaskAnchorEMA()
        # Run once so EMA has a value
        ema.update(torch.tensor([0.5]), ["a"])
        out = compute_v_molpo_loss(
            elbo_theta_w=torch.tensor([-1.0]),
            elbo_ref_w=torch.tensor([-2.0]),
            elbo_theta_l=torch.tensor([-3.0]),
            elbo_ref_l=torch.tensor([-2.0]),
            tasks_chosen=["a"], task_anchor_ema=ema, beta=1.0,
            anc_rejected_weight=0.1,
        )
        # L_anchor = -logσ(-(r_l - 1.5*ema)) = -logσ(-(-1.0 - 1.5*r_w_post_update))
        # The exact value depends on the EMA dynamics; just check it's positive
        assert out["loss_anchor"].item() > 0.0

    def test_margin_clip_active_truncates(self):
        ema = TaskAnchorEMA()
        # |r_w| = 0.1, margin_clip_scale=1.0 → clip to ±0.1 (very tight)
        out = compute_v_molpo_loss(
            elbo_theta_w=torch.tensor([-1.0]),  # r_w = 0.1*(- (-2)) wait
            elbo_ref_w=torch.tensor([-1.0]),    # r_w = 0.1*0 = 0
            elbo_theta_l=torch.tensor([-5.0]),  # r_l = 0.1*(-5 - (-1)) = -0.4
            elbo_ref_l=torch.tensor([-1.0]),
            tasks_chosen=["a"], task_anchor_ema=ema, beta=0.1,
            margin_clip_scale=1.0, margin_clip_active=True,
        )
        # raw margin = r_w - r_l = 0 - (-0.4) = 0.4
        # clip threshold = 1.0 * |0| = 0 → margin clipped to 0
        assert out["margin_unclipped"].item() == pytest.approx(0.4)
        assert out["margin"].item() == pytest.approx(0.0, abs=1e-6)
        assert out["margin_clipped_frac"] > 0.0

    def test_margin_clip_inactive_passthrough(self):
        ema = TaskAnchorEMA()
        out = compute_v_molpo_loss(
            elbo_theta_w=torch.tensor([-1.0]),
            elbo_ref_w=torch.tensor([-2.0]),
            elbo_theta_l=torch.tensor([-3.0]),
            elbo_ref_l=torch.tensor([-2.0]),
            tasks_chosen=["a"], task_anchor_ema=ema, beta=1.0,
            margin_clip_scale=1.0, margin_clip_active=False,
        )
        assert torch.allclose(out["margin"], out["margin_unclipped"])
        assert out["margin_clipped_frac"] == 0.0

    def test_per_task_gamma(self):
        """Different tasks → different EMA → different γ_i per sample."""
        ema = TaskAnchorEMA(alpha=0.99)
        # Pre-populate EMA with task-specific values
        ema.update(torch.tensor([10.0, 1.0]), ["task_high", "task_low"])

        out = compute_v_molpo_loss(
            elbo_theta_w=torch.zeros(2),
            elbo_ref_w=torch.zeros(2),
            elbo_theta_l=torch.zeros(2),
            elbo_ref_l=torch.zeros(2),
            tasks_chosen=["task_high", "task_low"],
            task_anchor_ema=ema, beta=0.1, molpo_lambda=0.5,
        )
        # γ_i = 0.5 * |EMA_i|
        # Note: EMA gets updated during compute_v_molpo_loss with current rewards (=0)
        # → for task_high: 0.99*10 + 0.01*0 = 9.9, γ = 0.5*9.9 = 4.95
        # → for task_low:  0.99*1 + 0.01*0 = 0.99, γ = 0.5*0.99 = 0.495
        # Just check ordering: task_high γ > task_low γ
        assert out["gamma"][0].item() > out["gamma"][1].item()


# ─────────────────────────────────────────────────────────────────
# combine_total_loss
# ─────────────────────────────────────────────────────────────────

class TestCombineTotalLoss:

    def test_no_sft(self):
        v_out = {
            "loss_pref": torch.tensor(2.0),
            "loss_anchor": torch.tensor(0.5),
        }
        total = combine_total_loss(loss_sft=None, v_molpo_out=v_out,
                                   sft_weight=1.0, molpo_weight=0.25,
                                   anc_rejected_weight=0.1)
        # 0.25*2.0 + 0.1*0.5 = 0.5 + 0.05 = 0.55
        assert total.item() == pytest.approx(0.55)

    def test_with_sft(self):
        v_out = {
            "loss_pref": torch.tensor(2.0),
            "loss_anchor": torch.tensor(0.0),
        }
        loss_sft = torch.tensor(1.0)
        total = combine_total_loss(loss_sft=loss_sft, v_molpo_out=v_out,
                                   sft_weight=1.0, molpo_weight=0.25,
                                   anc_rejected_weight=0.0)
        # 1.0*1.0 + 0.25*2.0 = 1.5
        assert total.item() == pytest.approx(1.5)

    def test_zero_sft_weight_skips_sft(self):
        v_out = {
            "loss_pref": torch.tensor(2.0),
            "loss_anchor": torch.tensor(0.0),
        }
        loss_sft = torch.tensor(1.0)
        total = combine_total_loss(loss_sft=loss_sft, v_molpo_out=v_out,
                                   sft_weight=0.0, molpo_weight=0.25,
                                   anc_rejected_weight=0.0)
        # SFT skipped: 0.25*2.0 = 0.5
        assert total.item() == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────
# Loss is differentiable through θ ELBOs
# ─────────────────────────────────────────────────────────────────

def test_loss_differentiable():
    """L_pref grad flows through elbo_theta but not through elbo_ref (no_grad expected)."""
    elbo_theta_w = torch.tensor([-1.0], requires_grad=True)
    elbo_theta_l = torch.tensor([-3.0], requires_grad=True)
    elbo_ref_w = torch.tensor([-2.0])  # no_grad
    elbo_ref_l = torch.tensor([-2.0])

    ema = TaskAnchorEMA()
    out = compute_v_molpo_loss(
        elbo_theta_w=elbo_theta_w, elbo_ref_w=elbo_ref_w,
        elbo_theta_l=elbo_theta_l, elbo_ref_l=elbo_ref_l,
        tasks_chosen=["a"], task_anchor_ema=ema, beta=1.0,
    )
    out["loss_pref"].backward()
    assert elbo_theta_w.grad is not None
    assert elbo_theta_l.grad is not None
    # Sign: increasing theta_w should DECREASE loss → grad < 0
    assert elbo_theta_w.grad.item() < 0
    # Increasing theta_l should INCREASE loss → grad > 0
    assert elbo_theta_l.grad.item() > 0
