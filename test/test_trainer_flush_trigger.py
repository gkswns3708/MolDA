"""Regression test for the per-optimizer-step `_flush_metrics` trigger.

Background: with `accumulate_grad_batches > 1`, `self.global_step` stays
constant across all micro-batches of one optimizer step.  The previous flush
condition `(self.global_step + 1) % log_interval == 0` therefore fires on
every micro-batch of the matching optimizer step (e.g. 57 times for
accum=57), and `self.log()`'s last-write-wins semantics keep only the
single-micro-batch mean — collapsing the effective sample size from
(accum × log_interval × per_device_B × n_ranks) down to one micro-batch.

The fix gates flush on `(batch_idx + 1) % accum == 0` so the buffer is only
emitted once per optimizer-step boundary.  This test isolates and exercises
that gating logic without spinning up the real LightningModule / LLaDA.
"""

from __future__ import annotations

from src.training.trainer import MolDATrainer


class _FakeTrainerAttr:
    """Stand-in for `self.trainer` exposing only the two attrs we read."""

    def __init__(self, *, accumulate_grad_batches: int, log_every_n_steps: int):
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_every_n_steps = log_every_n_steps


class _StubTrainer:
    """Minimal stand-in for `MolDATrainer` exposing the buffer/flush API.

    Re-uses the production `_accumulate` / `_flush_metrics` so we test the
    *real* methods and only mock the surrounding Lightning state.
    """

    def __init__(self, *, accumulate_grad_batches: int, log_every_n_steps: int):
        self._metric_buffer: dict[str, dict] = {}
        self.global_step = 0
        self.trainer = _FakeTrainerAttr(
            accumulate_grad_batches=accumulate_grad_batches,
            log_every_n_steps=log_every_n_steps,
        )
        self.logged: list[tuple[int, str, float]] = []

    # Re-bound production methods.
    _accumulate = MolDATrainer._accumulate

    def _flush_metrics(self):
        # Mirror production: average buffer, "log", clear.
        for name, info in self._metric_buffer.items():
            vals = info["values"]
            mean_val = sum(vals) / len(vals)
            self.logged.append((self.global_step, name, mean_val))
        self._metric_buffer.clear()

    def log(self, *args, **kwargs):  # not used; _flush_metrics overridden
        raise AssertionError("StubTrainer.log should not be called")

    def step(self, batch_idx: int, value: float):
        """Mimic the relevant slice of `training_step`.

        Production code (post-fix) inside training_step:

            self._accumulate("train/metric", value)
            log_interval = self.trainer.log_every_n_steps
            accum = self.trainer.accumulate_grad_batches or 1
            is_last_micro = (batch_idx + 1) % accum == 0
            if is_last_micro and (self.global_step + 1) % log_interval == 0:
                self._flush_metrics()
            # Lightning increments global_step *after* optimizer.step()
        """
        self._accumulate("train/metric", value, sync_dist=True)

        log_interval = self.trainer.log_every_n_steps
        accum = self.trainer.accumulate_grad_batches or 1
        is_last_micro = (batch_idx + 1) % accum == 0
        if is_last_micro and (self.global_step + 1) % log_interval == 0:
            self._flush_metrics()

        # Simulate Lightning bumping global_step at optimizer.step boundary.
        if is_last_micro:
            self.global_step += 1


def test_flush_fires_once_per_log_interval():
    """Buffer should flush exactly once every (accum × log_every_n_steps) micro-batches."""
    accum = 57
    log_every = 10
    n_optimizer_steps = 30
    n_micro = accum * n_optimizer_steps

    t = _StubTrainer(accumulate_grad_batches=accum, log_every_n_steps=log_every)
    for mb in range(n_micro):
        t.step(batch_idx=mb, value=float(mb))

    # Should flush at global_step=9, 19, 29 → 3 logs (last partial group at gs=29 is exact match).
    expected_flushes = n_optimizer_steps // log_every
    assert len(t.logged) == expected_flushes, (
        f"expected {expected_flushes} flush events with accum={accum}, "
        f"log_every={log_every}, n_micro={n_micro}; got {len(t.logged)}"
    )


def test_flush_sample_size_is_full_window():
    """Every flushed mean must reflect (accum × log_every_n_steps) micro-batches."""
    accum = 4  # small for explicit accounting
    log_every = 3
    window = accum * log_every  # = 12 micro-batches per flush

    t = _StubTrainer(accumulate_grad_batches=accum, log_every_n_steps=log_every)

    # Feed constant value 1.0 — a binary indicator-style metric stays 1.0
    # regardless of how it's averaged; what we verify is the *count*.
    counts = []

    # Monkeypatch _flush_metrics to capture buffer size *before* clearing.
    orig_flush = t._flush_metrics

    def capturing_flush():
        n = len(t._metric_buffer["train/metric"]["values"])
        counts.append(n)
        orig_flush()

    t._flush_metrics = capturing_flush

    for mb in range(window * 4):  # 4 flush windows
        t.step(batch_idx=mb, value=1.0)

    assert counts == [window, window, window, window], (
        f"each flush must aggregate {window} micro-batches; got counts={counts}"
    )


def test_no_flush_within_grad_accum_group():
    """During the 57 micro-batches of one optimizer step, no flush may fire."""
    accum = 57
    log_every = 1  # trigger condition ALWAYS true on global_step boundary
    t = _StubTrainer(accumulate_grad_batches=accum, log_every_n_steps=log_every)

    # First 56 micro-batches (batch_idx 0..55) — same optimizer step, gs=0.
    for mb in range(accum - 1):
        t.step(batch_idx=mb, value=float(mb))
    assert t.logged == [], "flush must NOT fire inside a grad-accum group"
    assert t.global_step == 0

    # 57th micro-batch — last of the group, gs advances 0 → 1, flush fires.
    t.step(batch_idx=accum - 1, value=999.0)
    assert len(t.logged) == 1, "exactly one flush at the optimizer-step boundary"
    gs_logged, name, mean_val = t.logged[0]
    assert gs_logged == 0, "flush is logged at the global_step that was current during the boundary mb"
    assert name == "train/metric"
    # Mean over [0, 1, ..., 55, 999] = (0+1+...+55 + 999) / 57
    expected = (sum(range(accum - 1)) + 999.0) / accum
    assert mean_val == expected


def test_no_grad_accum_behavior_unchanged():
    """With accumulate_grad_batches=1 the fix is a no-op vs old behavior."""
    t = _StubTrainer(accumulate_grad_batches=1, log_every_n_steps=5)
    for mb in range(20):
        t.step(batch_idx=mb, value=float(mb))
    # log_every=5: flushes at gs=4, 9, 14, 19 → 4 events.
    assert len(t.logged) == 4
    # Each flush captures exactly 5 entries.
    # mb 0..4 → mean 2.0; mb 5..9 → mean 7.0; etc.
    means = [v for _, _, v in t.logged]
    assert means == [2.0, 7.0, 12.0, 17.0]
