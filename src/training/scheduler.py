r"""
Warmup-Stable-Decay LR Scheduler.

Origin: Old_MolDA/model/scheduler.py (WarmupStableDecayLRScheduler only)
Removed lavis registry dependency and unused scheduler classes.

동작:
  LR
  │       ┌─────────────────────┐
  │      /   warmup    stable    \  decay(linear)
  │_____/                         ───────────
  0   warmup_steps       (1-decay_ratio)×total   total

각 param group의 초기 LR 비율이 유지됨:
  예: LoRA 2.5e-3, Embed 2.5e-5 → decay 후 2.5e-4, 2.5e-6
"""


class WarmupStableDecayLRScheduler:
    """
    각 param group의 초기 LR 비율을 유지하면서 스케줄링:
    - Warmup: 0 -> group_lr (비율 유지)
    - Stable: group_lr 유지
    - Decay: group_lr -> group_lr × min_lr_ratio (비율 유지, Linear Decay)
    """

    def __init__(self, optimizer, max_step, warmup_steps=50,
                 decay_ratio=0.1, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.max_step = max_step
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio

        # Decay starts at (1 - decay_ratio) * max_step
        self.decay_start_step = int(max_step * (1 - decay_ratio))

        # Store initial LRs per param group (set during optimizer creation)
        self.initial_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def step(self, cur_step):
        if cur_step < self.warmup_steps:
            # Warmup: 0 → 1.0
            ratio = cur_step / max(1, self.warmup_steps)
        elif cur_step < self.decay_start_step:
            # Stable: 1.0
            ratio = 1.0
        else:
            # Decay: 1.0 → min_lr_ratio (linear)
            decay_steps = self.max_step - self.decay_start_step
            progress = min(1.0, (cur_step - self.decay_start_step) / max(1, decay_steps))
            ratio = 1.0 - (1.0 - self.min_lr_ratio) * progress

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.initial_lrs[i] * ratio
