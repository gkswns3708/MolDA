"""
MolDATrainer: PyTorch Lightning module for MolDA training.

Handles:
- configure_optimizers: 5 param groups with WSD scheduler
- training_step: masked diffusion SFT loss
- validation_step: classification (likelihood) + generation (diffusion)
- on_validation_epoch_end: per-task metric aggregation + sample logging
- on_save_checkpoint: trainable params only
"""

import logging
from collections import defaultdict

import torch
import pytorch_lightning as pl
from torch.optim import AdamW

from src.model.molda import MolDA
from src.training.scheduler import WarmupStableDecayLRScheduler
from src.training.metrics import (
    CLASSIFICATION_TASKS, NAME_CONVERSION_TASKS,
    get_task_type, classification_evaluate,
    regression_evaluate, molecule_evaluate, caption_evaluate,
)
from src.logging.sample_logger import ValidationSampleLogger
from src.logging.stepwise_logger import StepwiseLogger

logger = logging.getLogger(__name__)


def _is_output_head_param(name: str) -> bool:
    """Output head 파라미터인지 판별. Block-level ff_out은 제외.

    매칭 대상: transformer.ff_out (최종 output head)
    제외 대상: transformer.blocks.*.ff_out (block FFN down projection)
    """
    lower = name.lower()
    if "lm_head" in lower:
        return True
    if "ff_out" in lower:
        # block-level ff_out 제외: "blocks." 뒤에 ff_out이 오는 패턴
        return "blocks." not in lower
    return False


class MolDATrainer(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.model = MolDA(cfg)

        # Normalize remasking_strategy to list
        rs = cfg.generation.remasking_strategy
        self._remasking_strategies = [rs] if isinstance(rs, str) else list(rs)

        # Validation output buffers (reset each epoch)
        self._val_cls_outputs = []   # (probs, label_texts, tasks)
        self._val_gen_outputs = []   # (pred_texts, target_texts, tasks, strategy)

        # Sample / stepwise loggers (log_dir 확보 후 setup()에서 초기화)
        self._sample_logger = None
        self._stepwise_logger = None

    @property
    def tokenizer(self):
        return self.model.tokenizer

    # ─────────────────────────────────────────
    # Optimizer + Scheduler
    # ─────────────────────────────────────────

    def configure_optimizers(self):
        """5 param groups: LoRA / embed_orig / embed_new / head / other."""
        cfg = self.cfg
        lr = cfg.lr
        orig_vocab_size = cfg.model.original_vocab_size

        # Collect params by group
        lora_params = []
        embed_orig_params = []
        embed_new_params = []
        head_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "lora_" in name:
                lora_params.append(param)
            elif "embed" in name.lower() or "wte" in name.lower():
                # Split embedding into orig vs new rows
                # Full embedding is one parameter; we handle it as one group
                # (per-row LR split is complex; use embed_orig LR for now)
                embed_orig_params.append(param)
            elif _is_output_head_param(name):
                head_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": lora_params, "lr": lr.lora, "name": "lora"},
            {"params": embed_orig_params, "lr": lr.embed_orig, "name": "embed"},
            {"params": head_params, "lr": lr.head_orig, "name": "head"},
        ]

        # Only add other group if there are params (Stage 2+)
        if other_params and lr.other > 0:
            param_groups.append(
                {"params": other_params, "lr": lr.other, "name": "other"}
            )

        optimizer = AdamW(
            param_groups,
            betas=(0.9, 0.95),
            weight_decay=cfg.training.weight_decay,
        )

        # Estimate total steps
        total_steps = self._estimate_total_steps()

        scheduler = WarmupStableDecayLRScheduler(
            optimizer=optimizer,
            max_step=total_steps,
            warmup_steps=cfg.scheduler.warmup_steps,
            decay_ratio=cfg.scheduler.decay_ratio,
            min_lr_ratio=cfg.scheduler.min_lr_ratio,
        )

        # Store scheduler for manual step in training_step
        # (PL expects torch _LRScheduler; ours is custom)
        self._scheduler = scheduler

        return optimizer

    def _estimate_total_steps(self) -> int:
        """Estimate total training steps for scheduler."""
        cfg = self.cfg
        if cfg.training.max_steps > 0:
            return cfg.training.max_steps

        # Estimate from epochs (rough; trainer may override)
        try:
            n_samples = len(self.trainer.datamodule.train_dataset)
        except Exception:
            n_samples = 2100  # fallback for toy dataset

        steps_per_epoch = n_samples // (
            cfg.training.batch_size * max(1, len(str(cfg.hardware.devices).split(",")))
        )
        steps_per_epoch = max(1, steps_per_epoch // cfg.training.accumulate_grad_batches)
        return steps_per_epoch * cfg.training.max_epochs

    # ─────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        batch["global_step"] = self.global_step
        out = self.model(batch)
        loss = out["loss"]

        # NaN guard
        if loss.isnan():
            logger.warning(f"NaN loss at step {self.global_step}, zeroing")
            loss = loss * 0.0

        # Manual LR scheduler step
        self._scheduler.step(self.global_step)

        # Logging
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/answer_length_mean", out["answer_length_mean"], sync_dist=True)

        # LR logging
        opt = self.optimizers()
        if hasattr(opt, "param_groups"):
            for group in opt.param_groups:
                if "name" in group:
                    self.log(f"lr/{group['name']}", group["lr"], sync_dist=False)

        return loss

    def on_before_optimizer_step(self, optimizer):
        """Manual scheduler step (not using PL's automatic scheduler)."""
        _, scheduler = self.optimizers(), None
        # Access our scheduler
        if hasattr(self, "_scheduler"):
            self._scheduler.step(self.global_step)

    def on_fit_start(self):
        """Store scheduler reference for manual stepping."""
        opt_sched = self.configure_optimizers()
        if isinstance(opt_sched, tuple) and len(opt_sched) == 2:
            self._scheduler = opt_sched[1]

    # ─────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────

    def setup(self, stage=None):
        """Trainer 연결 후 log_dir 기반으로 sample/stepwise logger 초기화."""
        if self._sample_logger is not None:
            return  # 이미 초기화됨

        # lightning_logs/version_N/ 경로 확보
        log_dir = self.trainer.log_dir or "."

        self._sample_logger = ValidationSampleLogger(
            log_dir=log_dir,
            samples_per_gpu=self.cfg.logging.get("val_log_samples_per_gpu", 1),
        )
        self._stepwise_logger = StepwiseLogger(
            log_dir=log_dir,
            max_samples=self.cfg.logging.get("stepwise_max_samples", 8),
            enabled=self.cfg.logging.get("log_stepwise_denoising", False),
        )
        logger.info(f"Loggers initialized: log_dir={log_dir}")

    def on_validation_epoch_start(self):
        self._val_cls_outputs = []
        self._val_gen_outputs = []  # (pred_texts, label_texts, tasks, strategy)
        if self._sample_logger:
            self._sample_logger.reset()
        if self._stepwise_logger:
            self._stepwise_logger.reset()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        tasks = batch["tasks"]
        prompt_ids = batch["prompt_input_ids"]
        prompt_mask = batch["prompt_attention_mask"]
        target_texts = batch["target_texts"]

        # Split by task type
        cls_idx = [i for i, t in enumerate(tasks) if t in CLASSIFICATION_TASKS]
        gen_idx = [i for i, t in enumerate(tasks)
                   if t not in CLASSIFICATION_TASKS and t not in NAME_CONVERSION_TASKS]

        # --- Classification: likelihood scoring ---
        if cls_idx:
            cls_prompt_ids = prompt_ids[cls_idx]
            cls_prompt_mask = prompt_mask[cls_idx]
            probs = self.model.compute_binary_prob_likelihood(
                cls_prompt_ids, cls_prompt_mask
            )
            cls_labels = [target_texts[i] for i in cls_idx]
            cls_tasks = [tasks[i] for i in cls_idx]
            self._val_cls_outputs.append((probs.cpu(), cls_labels, cls_tasks))

            # Sample 수집 (GPU당 N개 제한)
            if self._sample_logger:
                for i, ci in enumerate(cls_idx):
                    self._sample_logger.collect_classification(
                        tasks[ci], probs[i].cpu(), target_texts[ci],
                    )

        # --- Generation: diffusion sampling (각 remasking strategy별 반복) ---
        if gen_idx:
            gen_prompt_ids = prompt_ids[gen_idx]
            gen_prompt_mask = prompt_mask[gen_idx]
            gen_cfg = self.cfg.generation
            gen_labels = [target_texts[i] for i in gen_idx]
            gen_tasks = [tasks[i] for i in gen_idx]

            for strategy in self._remasking_strategies:
                # Stepwise logging 조건: enabled + max_samples 이내
                if self._stepwise_logger and self._stepwise_logger.should_log():
                    from src.generation.generate import generate_with_logging
                    pred_ids, snapshots = generate_with_logging(
                        self.model.llada.model,
                        gen_prompt_ids,
                        attention_mask=gen_prompt_mask,
                        gen_length=self.cfg.data.gen_max_len,
                        steps=gen_cfg.sampling_steps,
                        remasking=strategy,
                    )
                    # Deferred write: generation 완료 후 일괄 decode + file write
                    self._stepwise_logger.write_stepwise_log(
                        task=tasks[gen_idx[0]],
                        epoch=self.current_epoch,
                        global_step=self.global_step,
                        target_text=gen_labels[0],
                        step_snapshots=[s[0] for s in snapshots],  # 첫 번째 sample만
                        tokenizer=self.tokenizer,
                        config={
                            "steps": gen_cfg.sampling_steps,
                            "remasking": strategy,
                            "semi_ar": gen_cfg.semi_ar.get("enabled", False),
                        },
                    )
                else:
                    from src.generation.generate import generate
                    pred_ids = generate(
                        self.model.llada.model,
                        gen_prompt_ids,
                        attention_mask=gen_prompt_mask,
                        gen_length=self.cfg.data.gen_max_len,
                        steps=gen_cfg.sampling_steps,
                        remasking=strategy,
                    )

                # Decode predictions (only generated part, after prompt)
                prompt_len = gen_prompt_ids.shape[1]
                gen_part = pred_ids[:, prompt_len:]
                pred_texts = self.tokenizer.batch_decode(gen_part, skip_special_tokens=False)

                self._val_gen_outputs.append((pred_texts, gen_labels, gen_tasks, strategy))

                # Sample 수집 (GPU당 N개 제한, strategy 포함)
                if self._sample_logger:
                    for i, gi in enumerate(gen_idx):
                        self._sample_logger.collect_generation(
                            tasks[gi], pred_texts[i], target_texts[gi],
                            strategy=strategy,
                        )

    def on_validation_epoch_end(self):
        # --- Classification metrics ---
        cls_by_task = defaultdict(lambda: {"probs": [], "labels": []})
        for probs, labels, tasks in self._val_cls_outputs:
            for i, task in enumerate(tasks):
                cls_by_task[task]["probs"].append(probs[i:i+1])
                cls_by_task[task]["labels"].append(labels[i])

        for task, data in cls_by_task.items():
            all_probs = torch.cat(data["probs"], dim=0)
            metrics = classification_evaluate(all_probs, data["labels"], task)
            for k, v in metrics.items():
                self.log(f"val/{task}/{k}", v, sync_dist=True)

        # --- Generation metrics (strategy별 분리) ---
        # key: (task, strategy)
        gen_by_key = defaultdict(lambda: {"preds": [], "labels": []})
        for pred_texts, label_texts, tasks, strategy in self._val_gen_outputs:
            for i, task in enumerate(tasks):
                gen_by_key[(task, strategy)]["preds"].append(pred_texts[i])
                gen_by_key[(task, strategy)]["labels"].append(label_texts[i])

        for (task, strategy), data in gen_by_key.items():
            task_type = get_task_type(task)
            if task_type == "regression":
                metrics = regression_evaluate(data["preds"], data["labels"], task)
            elif task_type == "molecule":
                metrics = molecule_evaluate(data["preds"], data["labels"], task)
            elif task_type == "caption":
                metrics = caption_evaluate(data["preds"], data["labels"], task)
            else:
                continue

            for k, v in metrics.items():
                self.log(f"val/{task}/{strategy}/{k}", v, sync_dist=True)

        # Sample logging: 수집된 sample 일괄 write
        if self._sample_logger:
            self._sample_logger.flush(
                self.current_epoch, self.global_step,
                rank=self.global_rank,
            )

        # Clear buffers
        self._val_cls_outputs = []
        self._val_gen_outputs = []

    # ─────────────────────────────────────────
    # Checkpoint: save trainable params only
    # ─────────────────────────────────────────

    def on_save_checkpoint(self, checkpoint):
        to_remove = []
        for key in checkpoint["state_dict"]:
            # Keep: lora params, embedding, output head, PEFT wrappers, GNN/QFormer
            keep = (
                any(k in key for k in [
                    "lora_", "embed", "wte", "lm_head", "output_embeddings",
                    "modules_to_save", "original_module",
                    "qformer", "gnn", "query_tokens", "opt_proj", "ln_graph",
                ])
                or _is_output_head_param(key)
            )
            if not keep:
                to_remove.append(key)

        for key in to_remove:
            del checkpoint["state_dict"][key]

        logger.info(f"Checkpoint: kept {len(checkpoint['state_dict'])} params, "
                    f"removed {len(to_remove)} frozen params")
