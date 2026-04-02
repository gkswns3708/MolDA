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
from itertools import product

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
from src.loggers.sample_logger import ValidationSampleLogger
from src.loggers.stepwise_logger import StepwiseLogger
from src.loggers.train_prediction_logger import TrainPredictionLogger
from src.loggers.grad_logger import compute_grad_norms

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

        # Normalize val_strategies (sampling strategies) to list
        vs = cfg.generation.get("val_strategies", ["random"])
        self._val_strategies = [vs] if isinstance(vs, str) else list(vs)

        # Validation output buffers (reset each epoch)
        self._val_cls_outputs = []   # (probs, label_texts, tasks)
        self._val_gen_outputs = []   # (pred_texts, target_texts, tasks, strategy)

        # Sample / stepwise / train prediction loggers (log_dir 확보 후 setup()에서 초기화)
        self._sample_logger = None
        self._stepwise_logger = None
        self._train_pred_logger = None

        # Metric 구간 평균 버퍼 (CPU-only: Python float list)
        # flush 시점에 구간 평균을 self.log()로 기록
        self._metric_buffer: dict[str, dict] = {}

    @property
    def tokenizer(self):
        return self.model.tokenizer

    # ─────────────────────────────────────────
    # Optimizer + Scheduler
    # ─────────────────────────────────────────

    def configure_optimizers(self):
        """Param groups: LoRA / embed / head(optional) / other.

        embed_new/head_new LR은 gradient scaling으로 구현:
        - embed/head param group은 orig LR로 등록
        - on_before_optimizer_step에서 new vocab rows의 gradient에
          (lr_new / lr_orig) 비율을 곱해 effective LR을 높임
        - add_mol_dict=false(no_dict)이면 scaling 안 함
        """
        cfg = self.cfg
        lr = cfg.lr
        orig_vocab_size = cfg.model.original_vocab_size

        # Collect params by group
        lora_params = []
        embed_params = []
        head_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "lora_" in name:
                lora_params.append(param)
            elif "embed" in name.lower() or "wte" in name.lower():
                embed_params.append(param)
            elif _is_output_head_param(name):
                head_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": lora_params, "lr": lr.lora, "name": "lora"},
            {"params": embed_params, "lr": lr.embed_orig, "name": "embed"},
        ]

        # head group: weight_tying=True면 별도 ff_out이 없어 비어있을 수 있음
        if head_params:
            param_groups.append(
                {"params": head_params, "lr": lr.head_orig, "name": "head"}
            )

        # Only add other group if there are params (Stage 2+)
        if other_params and lr.other > 0:
            param_groups.append(
                {"params": other_params, "lr": lr.other, "name": "other"}
            )

        # --- Gradient scaling info for new vocab rows ---
        # add_mol_dict=false (no_dict)이면 new vocab이 없으므로 scaling 불필요
        use_new_scaling = cfg.tokenizer.get("add_mol_dict", False)

        if use_new_scaling and lr.embed_orig > 0:
            lr_ratio_embed = lr.embed_new / lr.embed_orig
        else:
            lr_ratio_embed = 1.0

        if use_new_scaling and lr.head_orig > 0 and head_params:
            lr_ratio_head = lr.head_new / lr.head_orig
        else:
            lr_ratio_head = 1.0

        self._embed_head_split_info = {
            "original_vocab_size": orig_vocab_size,
            "lr_ratio_embed": lr_ratio_embed,
            "lr_ratio_head": lr_ratio_head,
            "embed_params": embed_params,
            "head_params": head_params,
        }

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

        num_devices = max(1, len(str(cfg.hardware.devices).split(",")))
        per_device_steps = n_samples // (cfg.training.batch_size * num_devices)
        accumulate = cfg.training.global_batch_size // (cfg.training.batch_size * num_devices)
        steps_per_epoch = max(1, per_device_steps // max(1, accumulate))
        return steps_per_epoch * cfg.training.max_epochs

    # ─────────────────────────────────────────
    # Training — metric 구간 평균
    # ─────────────────────────────────────────

    def _accumulate(self, name: str, value, *, sync_dist: bool = True, prog_bar: bool = False):
        """Metric 값을 버퍼에 누적 (GPU→CPU float 즉시 변환)."""
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        if name not in self._metric_buffer:
            self._metric_buffer[name] = {"values": [], "sync_dist": sync_dist, "prog_bar": prog_bar}
        self._metric_buffer[name]["values"].append(value)

    def _flush_metrics(self):
        """버퍼의 구간 평균을 self.log()로 기록 후 리셋."""
        for name, info in self._metric_buffer.items():
            vals = info["values"]
            mean_val = sum(vals) / len(vals)
            self.log(name, mean_val, prog_bar=info["prog_bar"], sync_dist=info["sync_dist"])
        self._metric_buffer.clear()

    # ─────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        batch["global_step"] = self.global_step

        # Set detailed prediction logging flag
        if self._train_pred_logger and self._train_pred_logger.should_log(self.global_step):
            batch["_log_train_detail"] = True

        out = self.model(batch)
        loss = out["loss"]

        # NaN guard
        if loss.isnan():
            logger.warning(f"NaN loss at step {self.global_step}, zeroing")
            loss = loss * 0.0

        # Manual LR scheduler step
        self._scheduler.step(self.global_step)

        # Metric 누적 (구간 평균용, CPU-only)
        self._accumulate("train/loss", loss, sync_dist=True, prog_bar=True)
        self._accumulate("train/loss_no_eos", out["per_sample_loss_no_eos"].mean(), sync_dist=True)
        self._accumulate("train/answer_length_mean", out["answer_length_mean"], sync_dist=True)

        # Per-task loss 누적 (sync_dist=False: tasks may differ across ranks)
        tasks = batch.get("tasks", [])
        if tasks:
            per_sample_loss = out["per_sample_loss"]
            per_sample_loss_no_eos = out["per_sample_loss_no_eos"]
            seen = set()
            for task in tasks:
                if task in seen:
                    continue
                seen.add(task)
                mask = torch.tensor(
                    [t == task for t in tasks], device=loss.device, dtype=torch.bool
                )
                self._accumulate(f"train/{task}/loss", per_sample_loss[mask].mean(), sync_dist=False)
                self._accumulate(f"train/{task}/loss_no_eos", per_sample_loss_no_eos[mask].mean(), sync_dist=False)

        # Prediction quality metrics 누적
        if "mask_accuracy" in out:
            self._accumulate("train/mask_accuracy", out["mask_accuracy"], sync_dist=True)
            self._accumulate("train/mask_accuracy_no_eos", out["mask_accuracy_no_eos"], sync_dist=True)
            self._accumulate("train/target_prob_mean", out["target_prob_mean"], sync_dist=True)

        # Detailed prediction sample log (periodic)
        if "_train_sample_detail" in out and self._train_pred_logger:
            detail = out["_train_sample_detail"]
            task = batch.get("tasks", ["unknown"])[0]
            self._train_pred_logger.write_sample_log(
                global_step=self.global_step,
                epoch=self.current_epoch,
                task=task,
                p_mask=detail["p_mask"],
                mask_positions=detail["mask_positions"],
                target_tokens=detail["target_tokens"],
                pred_tokens=detail["pred_tokens"],
                target_probs=detail["target_probs"],
                pred_probs=detail["pred_probs"],
                tokenizer=self.tokenizer,
                # Full-sequence data for text logging
                input_ids=detail.get("input_ids"),
                labels=detail.get("labels"),
                all_answer_pred_ids=detail.get("all_answer_pred_ids"),
                all_answer_gt_ids=detail.get("all_answer_gt_ids"),
                attention_mask=detail.get("attention_mask"),
            )

        # LR logging
        opt = self.optimizers()
        if hasattr(opt, "param_groups"):
            for group in opt.param_groups:
                if "name" in group:
                    self.log(f"lr/{group['name']}", group["lr"], sync_dist=False)

            # Log effective LR for new vocab rows (embed_new, head_new)
            info = getattr(self, "_embed_head_split_info", None)
            if info is not None:
                for group in opt.param_groups:
                    name = group.get("name", "")
                    if name == "embed":
                        self.log(
                            "lr/embed_new",
                            group["lr"] * info["lr_ratio_embed"],
                            sync_dist=False,
                        )
                    elif name == "head" and info["lr_ratio_head"] != 1.0:
                        self.log(
                            "lr/head_new",
                            group["lr"] * info["lr_ratio_head"],
                            sync_dist=False,
                        )

        # 구간 평균 flush (log_every_n_steps 간격)
        log_interval = getattr(self.trainer, "log_every_n_steps", 1)
        if (self.global_step + 1) % log_interval == 0:
            self._flush_metrics()

        return loss

    def on_before_optimizer_step(self, optimizer):
        """Apply gradient scaling for new vocab rows, then log grad norms."""
        # --- Gradient scaling: new vocab rows에 lr_ratio 곱하기 ---
        info = getattr(self, "_embed_head_split_info", None)
        if info is not None:
            orig_size = info["original_vocab_size"]
            ratio_embed = info["lr_ratio_embed"]
            ratio_head = info["lr_ratio_head"]

            if ratio_embed != 1.0:
                for param in info["embed_params"]:
                    if param.grad is not None and param.shape[0] > orig_size:
                        param.grad[orig_size:] *= ratio_embed

            if ratio_head != 1.0:
                for param in info["head_params"]:
                    if param.grad is not None and param.shape[0] > orig_size:
                        param.grad[orig_size:] *= ratio_head

        # --- Gradient norm 누적 (구간 평균) ---
        interval = self.cfg.logging.get("weight_norm_interval", 10)
        if self.global_step % interval != 0:
            return
        for name, norm in compute_grad_norms(optimizer).items():
            self._accumulate(f"train/grad_norm/{name}", norm, sync_dist=False)

    def on_train_end(self):
        """학습 종료 시 버퍼에 남은 metric flush."""
        if self._metric_buffer:
            self._flush_metrics()

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
        self._train_pred_logger = TrainPredictionLogger(
            log_dir=log_dir,
            log_interval=self.cfg.logging.get("train_prediction_log_interval", 100),
            max_positions=self.cfg.logging.get("train_prediction_max_positions", 50),
            enabled=self.cfg.logging.get("log_train_predictions", True),
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

        # --- Generation: diffusion sampling (remasking × sampling 전략 조합) ---
        if gen_idx:
            gen_prompt_ids = prompt_ids[gen_idx]
            gen_prompt_mask = prompt_mask[gen_idx]
            gen_cfg = self.cfg.generation
            gen_labels = [target_texts[i] for i in gen_idx]
            gen_tasks = [tasks[i] for i in gen_idx]

            for remasking, sampling in product(
                self._remasking_strategies, self._val_strategies
            ):
                is_semi_ar = (sampling == "semi_ar")
                block_length = gen_cfg.semi_ar.block_size if is_semi_ar else None
                strategy_key = f"{remasking}_{sampling}"

                # Stepwise logging 조건: enabled + max_samples 이내
                if self._stepwise_logger and self._stepwise_logger.should_log():
                    from src.generation.generate import generate_with_logging
                    pred_ids, snapshots, _ = generate_with_logging(
                        self.model.llada.model,
                        gen_prompt_ids,
                        attention_mask=gen_prompt_mask,
                        gen_length=self.cfg.data.gen_max_len,
                        steps=gen_cfg.sampling_steps,
                        remasking=remasking,
                        semi_ar=is_semi_ar,
                        block_length=block_length or gen_cfg.sampling_steps,
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
                            "remasking": remasking,
                            "semi_ar": is_semi_ar,
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
                        remasking=remasking,
                        semi_ar=is_semi_ar,
                        block_length=block_length or gen_cfg.sampling_steps,
                    )

                # Decode predictions (only generated part, after prompt)
                prompt_len = gen_prompt_ids.shape[1]
                gen_part = pred_ids[:, prompt_len:]
                pred_texts = self.tokenizer.batch_decode(gen_part, skip_special_tokens=False)

                self._val_gen_outputs.append((pred_texts, gen_labels, gen_tasks, strategy_key))

                # Sample 수집 (GPU당 N개 제한, strategy 포함)
                if self._sample_logger:
                    for i, gi in enumerate(gen_idx):
                        self._sample_logger.collect_generation(
                            tasks[gi], pred_texts[i], target_texts[gi],
                            strategy=strategy_key,
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

        # Sample logging: WandB Table (rank 0만, flush 전에 호출) + TXT 파일
        if self._sample_logger:
            if self.global_rank == 0:
                wandb_lg = self._get_wandb_logger()
                if wandb_lg is not None:
                    self._sample_logger.flush_to_wandb(
                        wandb_lg.experiment,
                        self.current_epoch, self.global_step,
                    )
            self._sample_logger.flush(
                self.current_epoch, self.global_step,
                rank=self.global_rank,
            )

        # Clear buffers
        self._val_cls_outputs = []
        self._val_gen_outputs = []

    # ─────────────────────────────────────────
    # WandB helpers
    # ─────────────────────────────────────────

    def _get_wandb_logger(self):
        """WandbLogger가 있으면 반환, 없으면 None."""
        for lg in self.loggers:
            if type(lg).__name__ == "WandbLogger":
                return lg
        return None

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
