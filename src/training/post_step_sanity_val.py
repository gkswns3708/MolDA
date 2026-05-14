"""Post-step sanity validation callback.

Lightning's built-in `num_sanity_val_steps` runs validation BEFORE training
starts. The optimizer's m/v moment tensors (AdamW state) and training-time
activations haven't been allocated yet, so the GPU memory profile of that
sanity val is significantly LIGHTER than real training. A sanity that
passes this way can still OOM on the very first real backward.

This callback fires a short validation AFTER the first optimizer step
completes, when:
  - optimizer.step() has allocated m/v moment tensors (≈ trainable_params × 8B)
  - the first backward graph has been built (activations memory peaked)
  - expandable_segments has already hit its real peak

→ The val that follows runs under a memory profile that closely matches
later training. If it OOMs, we know the config can't survive the run.

DDP-safe: all ranks fire the callback at the same global_step. Lightning's
val_loop coordinates the val pass across ranks just like a normal val.

Usage (in scripts/train.py):
    from src.training.post_step_sanity_val import PostStepSanityValCallback
    callbacks.append(PostStepSanityValCallback(
        fire_at_step=1,
        max_batches=20,
    ))
"""
from __future__ import annotations

import pytorch_lightning as pl


class PostStepSanityValCallback(pl.Callback):
    """Run a short validation after the first optimizer step has completed.

    Args:
        fire_at_step: trigger once when `trainer.global_step >= fire_at_step`.
            Default 1 = after the first opt.step() (optimizer state allocated).
        max_batches: temporary limit_val_batches override for this single
            sanity run. Default 20.
        verbose: if True, print log lines on enter/exit.
    """

    def __init__(
        self,
        fire_at_step: int = 1,
        max_batches: int = 20,
        verbose: bool = True,
    ):
        super().__init__()
        self.fire_at_step = max(1, int(fire_at_step))
        self.max_batches = max(1, int(max_batches))
        self.verbose = verbose
        self._done = False

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        if self._done:
            return
        # Wait until optimizer.step() has run at least `fire_at_step` times.
        # `global_step` increments per optimizer step (post-accumulation).
        if trainer.global_step < self.fire_at_step:
            return

        self._done = True
        rank = trainer.global_rank

        if self.verbose:
            print(
                f"[PostStepSanityVal] Rank {rank}: triggering "
                f"{self.max_batches}-batch sanity val at global_step="
                f"{trainer.global_step}",
                flush=True,
            )

        # Stash original config so we restore it after the sanity val.
        orig_limit = trainer.limit_val_batches
        trainer.limit_val_batches = self.max_batches

        # Lightning's training_epoch_loop normally toggles `trainer.training`
        # around val_loop.run(). We mirror that here for a clean state.
        was_training = trainer.training
        trainer.training = False
        try:
            # Use the same internal entrypoint that Lightning's regular
            # val_check_interval-triggered validation uses.
            trainer.fit_loop.epoch_loop.val_loop.run()
        except Exception as e:
            print(
                f"[PostStepSanityVal] Rank {rank}: val_loop.run() failed "
                f"with {type(e).__name__}: {e}. Continuing training.",
                flush=True,
            )
        finally:
            trainer.training = was_training
            trainer.limit_val_batches = orig_limit
            if self.verbose:
                print(
                    f"[PostStepSanityVal] Rank {rank}: sanity val done. "
                    f"Restored limit_val_batches={orig_limit}.",
                    flush=True,
                )
