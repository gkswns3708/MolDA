"""Tests for WarmupStableDecayLRScheduler in src/training/scheduler.py."""

import pytest
import torch

from src.training.scheduler import WarmupStableDecayLRScheduler


def _make_optimizer(lrs=(1e-3,)):
    """Create a simple optimizer with given per-group LRs."""
    param_groups = []
    for lr in lrs:
        p = torch.nn.Parameter(torch.randn(2))
        param_groups.append({"params": [p], "lr": lr})
    return torch.optim.SGD(param_groups, lr=0.0)


class TestWarmupPhase:

    def test_warmup_starts_at_zero(self):
        opt = _make_optimizer([1e-3])
        sched = WarmupStableDecayLRScheduler(opt, max_step=1000, warmup_steps=50)
        sched.step(0)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-10)

    def test_warmup_linear_increase(self):
        opt = _make_optimizer([1e-3])
        sched = WarmupStableDecayLRScheduler(opt, max_step=1000, warmup_steps=100)
        sched.step(50)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5e-3, rel=1e-5)

    def test_warmup_end_reaches_initial_lr(self):
        opt = _make_optimizer([2.5e-3])
        sched = WarmupStableDecayLRScheduler(opt, max_step=1000, warmup_steps=50)
        sched.step(50)
        assert opt.param_groups[0]["lr"] == pytest.approx(2.5e-3, rel=1e-5)


class TestStablePhase:

    def test_stable_phase_constant_lr(self):
        opt = _make_optimizer([1e-3])
        sched = WarmupStableDecayLRScheduler(
            opt, max_step=1000, warmup_steps=50, decay_ratio=0.1
        )
        # Stable phase: step 50 to 900
        for step in [100, 300, 500, 800]:
            sched.step(step)
            assert opt.param_groups[0]["lr"] == pytest.approx(1e-3, rel=1e-5), \
                f"LR not stable at step {step}"


class TestDecayPhase:

    def test_decay_decreases_lr(self):
        opt = _make_optimizer([1e-3])
        sched = WarmupStableDecayLRScheduler(
            opt, max_step=1000, warmup_steps=50, decay_ratio=0.1, min_lr_ratio=0.1
        )
        sched.step(900)
        lr_start = opt.param_groups[0]["lr"]
        sched.step(950)
        lr_mid = opt.param_groups[0]["lr"]
        sched.step(1000)
        lr_end = opt.param_groups[0]["lr"]
        assert lr_start > lr_mid > lr_end

    def test_decay_end_reaches_min_lr(self):
        opt = _make_optimizer([1e-3])
        sched = WarmupStableDecayLRScheduler(
            opt, max_step=1000, warmup_steps=50, decay_ratio=0.1, min_lr_ratio=0.1
        )
        sched.step(1000)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-4, rel=1e-5)

    def test_decay_start_step_calculation(self):
        opt = _make_optimizer([1e-3])
        sched = WarmupStableDecayLRScheduler(
            opt, max_step=1000, warmup_steps=50, decay_ratio=0.1
        )
        assert sched.decay_start_step == 900


class TestMultiParamGroup:

    def test_ratio_preserved(self):
        opt = _make_optimizer([2.5e-3, 2.5e-5])
        sched = WarmupStableDecayLRScheduler(
            opt, max_step=1000, warmup_steps=50, decay_ratio=0.1, min_lr_ratio=0.1
        )
        # After warmup
        sched.step(50)
        assert opt.param_groups[0]["lr"] == pytest.approx(2.5e-3, rel=1e-5)
        assert opt.param_groups[1]["lr"] == pytest.approx(2.5e-5, rel=1e-5)

        # Mid decay
        sched.step(950)
        ratio = opt.param_groups[0]["lr"] / opt.param_groups[1]["lr"]
        assert ratio == pytest.approx(100.0, rel=1e-3)

        # End
        sched.step(1000)
        assert opt.param_groups[0]["lr"] == pytest.approx(2.5e-4, rel=1e-5)
        assert opt.param_groups[1]["lr"] == pytest.approx(2.5e-6, rel=1e-5)
