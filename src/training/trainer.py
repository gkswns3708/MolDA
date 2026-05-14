"""
MolDATrainer: PyTorch Lightning module for MolDA training.

Core 로직(__init__, training_step, metric buffer)만 유지.
나머지는 Mixin으로 분리:
- OptimizerMixin (optimizer.py): configure_optimizers, gradient scaling
- ValidationMixin (validation.py): setup, validation_step, epoch hooks
- CheckpointMixin (checkpoint.py): on_save_checkpoint
"""

import logging

import torch
import pytorch_lightning as pl

from src.model.molda import MolDA
from src.training.optimizer import OptimizerMixin
from src.training.validation import ValidationMixin
from src.training.checkpoint import CheckpointMixin

logger = logging.getLogger(__name__)


class MolDATrainer(OptimizerMixin, ValidationMixin, CheckpointMixin, pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.model = MolDA(cfg)

        # Stage N→N+1 이전: pretrained_ckpt_path로 weight만 로드 (없으면 skip)
        pretrained_ckpt = cfg.get("pretrained_ckpt_path")
        if pretrained_ckpt:
            assert not cfg.get("ckpt_path"), (
                "ckpt_path와 pretrained_ckpt_path는 동시에 사용 불가"
            )
            self.load_pretrained_state_dict(pretrained_ckpt)

        # Stage-specific freeze policy (Stage 2+ 에서만 적용)
        self.model._apply_stage_freeze_policy()

        # Normalize remasking_strategy to list
        rs = cfg.generation.remasking_strategy
        self._remasking_strategies = [rs] if isinstance(rs, str) else list(rs)

        # Normalize val_strategies (sampling strategies) to list
        vs = cfg.generation.get("val_strategies", ["random"])
        self._val_strategies = [vs] if isinstance(vs, str) else list(vs)

        # Validation: JSONL file handles (initialized in on_validation_epoch_start)
        self._val_cls_fh = None
        self._val_gen_fh = None

        # Stepwise / train prediction loggers (log_dir 확보 후 setup()에서 초기화)
        self._stepwise_logger = None
        self._train_pred_logger = None

        # Metric 구간 평균 버퍼 (CPU-only: Python float list)
        # flush 시점에 구간 평균을 self.log()로 기록
        self._metric_buffer: dict[str, dict] = {}

        # Masking-ratio bucket edges for per-bucket accuracy logging
        # bucket i covers p_mask in [edges[i], edges[i+1])
        self._mask_ratio_bucket_edges = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self._bucket_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

        # Validation strategy-wise custom chart history (populated by ValidationMixin)
        self._val_custom_chart_history: dict = {}

        # Eval phase: "val" (default) or "test" — toggled by on_test_epoch_start
        # so async aggregation prefixes metrics with the right namespace.
        self._eval_phase: str = "val"

    @property
    def tokenizer(self):
        return self.model.tokenizer

    # test_step reuses validation_step: Test loop produces the same
    # JSONL + failure/prediction artifacts as validation.
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def on_test_epoch_start(self):
        self._eval_phase = "test"
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

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
        if batch_idx == 0 and self.global_rank == 0:
            is_molpo_batch = "molpo_batch_size" in batch
            molpo_enabled = getattr(self.model, "molpo_enabled", False)
            mode = "V-MolPO" if (is_molpo_batch and molpo_enabled) else "SFT"
            molpo_size = batch.get("molpo_batch_size", "n/a")
            molpo_div = batch.get("molpo_batch_division", "n/a")
            print(
                f"\n{'=' * 60}\n"
                f"[Sanity check, Rank 0, batch 0] Active mode = {mode}\n"
                f"  molpo_enabled (model)        = {molpo_enabled}\n"
                f"  molpo_batch_size in batch    = {is_molpo_batch}\n"
                f"  molpo_batch_size value       = {molpo_size}\n"
                f"  molpo_batch_division value   = {molpo_div}\n"
                f"  input_ids shape              = {tuple(batch['input_ids'].shape)}\n"
                f"  has graphs?                  = {'graphs' in batch}\n"
                f"  ref_model present?           = {self.model.ref_model is not None}\n"
                f"{'=' * 60}",
                flush=True,
            )
        if batch_idx == 0:
            print(f"[Rank {self.global_rank}] training_step: batch_idx=0 ENTERED", flush=True)
        batch["global_step"] = self.global_step

        # Set detailed prediction logging flag (global_step 기준, step당 1번만)
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

        # V-MolPO sub-metrics (key prefix "v_molpo/" — produced by MolDA._molpo_forward).
        # Per-task slicing happens below using `*_per_sample` and the [B]-form
        # elbo tensors; skip those in the global iter to avoid duplicate scalar
        # logs of the same value already covered by `loss_pref` / `elbo_*_mean`.
        _v_molpo_global_skip = {
            "v_molpo/elbo_theta_w", "v_molpo/elbo_ref_w",
            "v_molpo/elbo_theta_l", "v_molpo/elbo_ref_l",
        }
        for k, v in out.items():
            if not k.startswith("v_molpo/"):
                continue
            if k.endswith("_per_sample") or k in _v_molpo_global_skip:
                continue
            if isinstance(v, torch.Tensor):
                v = v.mean() if v.numel() > 1 else v
            self._accumulate(f"train/{k}", v, sync_dist=True)

        # Per-task loss / mask_accuracy 누적 (sync_dist=False: tasks may differ across ranks)
        # V-MolPO 모드에서는 out["tasks"] (length B, chosen-only) 가 batch["tasks"] 보다 우선.
        tasks = out.get("tasks") or batch.get("tasks", [])
        if tasks:
            per_sample_loss = out["per_sample_loss"]
            per_sample_loss_no_eos = out["per_sample_loss_no_eos"]
            per_sample_acc = out.get("per_sample_mask_accuracy")
            per_sample_acc_no_eos = out.get("per_sample_mask_accuracy_no_eos")

            # Collect V-MolPO per-sample tensors of shape [B] for task slicing.
            # Includes `*_per_sample`, the [B]-form elbos, and the originally-[B]
            # metrics (margin / rewards_* / gamma / avg_chosen_reward) — all match
            # len(tasks)=B because tasks_chosen is the chosen-side slice.
            B_tasks = len(tasks)
            v_molpo_per_sample = {
                k: v for k, v in out.items()
                if k.startswith("v_molpo/")
                and isinstance(v, torch.Tensor)
                and v.dim() >= 1
                and v.shape[0] == B_tasks
            }

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
                if per_sample_acc is not None:
                    self._accumulate(
                        f"train/{task}/mask_accuracy",
                        per_sample_acc[mask].mean(),
                        sync_dist=False,
                    )
                    self._accumulate(
                        f"train/{task}/mask_accuracy_no_eos",
                        per_sample_acc_no_eos[mask].mean(),
                        sync_dist=False,
                    )
                # Per-task V-MolPO: slice every [B]-shape v_molpo tensor by task.
                # Keys like `v_molpo/rewards_chosen`, `v_molpo/margin`,
                # `v_molpo/loss_pref_per_sample`, `v_molpo/elbo_theta_w` etc.
                for k, v in v_molpo_per_sample.items():
                    self._accumulate(
                        f"train/{task}/{k}",
                        v[mask].mean(),
                        sync_dist=False,
                    )

        # Prediction quality metrics 누적
        if "mask_accuracy" in out:
            self._accumulate("train/mask_accuracy", out["mask_accuracy"], sync_dist=True)
            self._accumulate("train/mask_accuracy_no_eos", out["mask_accuracy_no_eos"], sync_dist=True)
            self._accumulate("train/target_prob_mean", out["target_prob_mean"], sync_dist=True)

        # Masking-ratio bucket별 mask accuracy 누적 (전역 — task 무관)
        if "p_mask_per_sample" in out and "per_sample_mask_accuracy" in out:
            p_mask_vec = out["p_mask_per_sample"]                        # [B]
            acc_vec = out["per_sample_mask_accuracy"]                    # [B]
            inner_edges = self._mask_ratio_bucket_edges[1:-1].to(p_mask_vec.device)
            bucket_ids = torch.bucketize(p_mask_vec, inner_edges)        # [B] in [0..4]
            for b_idx in range(len(self._bucket_labels)):
                sel = (bucket_ids == b_idx)
                if sel.any():
                    self._accumulate(
                        f"train/mask_acc/bucket_{self._bucket_labels[b_idx]}",
                        acc_vec[sel].mean(),
                        sync_dist=False,
                    )

        # Detailed prediction sample log (periodic, step당 1번만 — should_log 내부에서 중복 방지)
        if "_train_sample_detail" in out and self._train_pred_logger:
            detail = out["_train_sample_detail"]
            task = batch.get("tasks", ["unknown"])[0]
            self._train_pred_logger.write_sample_log(
                global_step=self.global_step,
                epoch=self.current_epoch,
                rank=self.global_rank,
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

        # 구간 평균 flush (log_every_n_steps 간격, optimizer-step 경계에서만 1회).
        # `self.global_step` 은 optimizer step 단위라 grad-accum 의 모든 micro-batch
        # 에서 동일 값을 유지한다. 조건만으로 flush 하면 같은 step 동안 57회 호출되어
        # 두 번째 호출부터 buffer 가 1 entry 로 줄고, `self.log()` 의 last-write-wins
        # 으로 직전 누적치가 단일 micro-batch 평균에 덮여 표본 수가 ~18 pair 로 추락.
        # `is_last_micro` 가드로 optimizer.step() 이 실제 일어나는 마지막 micro-batch
        # 에서만 flush 하여 누적된 (accum × log_interval) micro-batch 평균을 보존.
        log_interval = getattr(self.trainer, "log_every_n_steps", 1)
        accum = getattr(self.trainer, "accumulate_grad_batches", 1) or 1
        is_last_micro = (batch_idx + 1) % accum == 0
        if is_last_micro and (self.global_step + 1) % log_interval == 0:
            self._flush_metrics()

        return loss

    def on_train_epoch_start(self):
        """Training epoch 시작 시 GPU 메모리 정리 + collator epoch index 동기화."""
        print(f"[Rank {self.global_rank}] train_epoch_start: START", flush=True)
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Sync graph-rejection rotation index with current epoch so the
        # MolPOTrainCollator picks a different `{i}-th_rejected_*` variant
        # each epoch (Old_MolDA `reject_cardinal = current_epoch` pattern).
        try:
            dm = self.trainer.datamodule
            loaders = getattr(self.trainer, "train_dataloader", None)
            if loaders is not None and hasattr(loaders, "collate_fn"):
                collator = loaders.collate_fn
                if hasattr(collator, "current_epoch"):
                    collator.current_epoch = int(self.current_epoch)
        except Exception as e:
            print(f"[Rank {self.global_rank}] collator epoch sync skipped: {e}",
                  flush=True)
        print(f"[Rank {self.global_rank}] train_epoch_start: DONE", flush=True)

    def on_train_epoch_end(self):
        """Epoch 종료 시 버퍼에 남은 metric flush.

        on_train_end에서는 self.log() 사용 금지(Lightning 제약).
        epoch 끝이 훅 체인 상 on_train_end 직전이라 여기서 마지막 flush 처리.
        """
        if self._metric_buffer:
            self._flush_metrics()

    def on_train_end(self):
        """학습 종료 — self.log() 금지 구역. 잔여 버퍼가 있으면 조용히 버림.

        정상 경로에서는 on_train_epoch_end에서 이미 flush되어 버퍼가 비어있음.
        예외 경로로 잔여가 있으면 logger.experiment로 직접 기록해도 되지만,
        단순화를 위해 버퍼만 clear.
        """
        self._metric_buffer.clear()

