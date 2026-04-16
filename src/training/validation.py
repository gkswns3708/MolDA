"""ValidationMixin: validation step/epoch 로직 및 JSONL DDP-safe helpers.

분리 대상:
- setup: sample/stepwise/train prediction logger 초기화
- on_validation_epoch_start/end: JSONL 파일 관리 + metric 집계
- validation_step: classification (likelihood) + generation (diffusion)
- _get_wandb_logger: WandbLogger 탐색
- JSONL helpers: _val_jsonl_path, _open_val_jsonl, _write_jsonl 등
"""

import json
import logging
import os
from collections import defaultdict
from itertools import product

import torch

from src.training.metrics import (
    CLASSIFICATION_TASKS, NAME_CONVERSION_TASKS,
    get_task_type, classification_evaluate,
    regression_evaluate, molecule_evaluate, caption_evaluate,
)
from src.loggers.stepwise_logger import StepwiseLogger
from src.loggers.train_prediction_logger import TrainPredictionLogger

logger = logging.getLogger(__name__)


class ValidationMixin:
    """Validation step/epoch 및 JSONL DDP-safe helpers를 담당하는 Mixin."""

    # ─────────────────────────────────────────
    # Validation JSONL helpers (DDP-safe)
    # ─────────────────────────────────────────

    def _val_jsonl_path(self, tag: str, rank: int = None,
                        epoch: int = None, step: int = None) -> str:
        """val-epoch{E}-step{S}-rank{R}-{tag}.jsonl 경로 반환."""
        if rank is None:
            rank = self.global_rank
        if epoch is None:
            epoch = self.current_epoch
        if step is None:
            step = self.global_step
        log_dir = self.trainer.log_dir or "."
        return os.path.join(
            log_dir,
            f"val-epoch{epoch}-step{step}-rank{rank}-{tag}.jsonl",
        )

    def _open_val_jsonl(self, tag: str):
        """JSONL 파일을 append 모드로 열어 file handle 반환."""
        path = self._val_jsonl_path(tag)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return open(path, "a", encoding="utf-8")

    def _write_jsonl(self, fh, record: dict):
        """JSONL 파일에 한 줄 기록."""
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_all_val_predictions(self, tag: str,
                                   epoch: int = None, step: int = None) -> list:
        """Rank 0: 모든 rank의 JSONL 파일을 로드하여 병합."""
        records = []
        world_size = self.trainer.world_size
        for rank in range(world_size):
            path = self._val_jsonl_path(tag, rank=rank, epoch=epoch, step=step)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    def _cleanup_val_jsonl(self, tag: str,
                           epoch: int = None, step: int = None):
        """사용 완료된 JSONL 파일 삭제."""
        for rank in range(self.trainer.world_size):
            path = self._val_jsonl_path(tag, rank=rank, epoch=epoch, step=step)
            if os.path.exists(path):
                os.remove(path)

    # ─────────────────────────────────────────
    # Static helpers (스레드에서 self.trainer 접근 없이 사용)
    # ─────────────────────────────────────────

    @staticmethod
    def _jsonl_path_static(log_dir, tag, rank, epoch, step):
        return os.path.join(
            log_dir, f"val-epoch{epoch}-step{step}-rank{rank}-{tag}.jsonl")

    @staticmethod
    def _load_all_val_predictions_static(log_dir, world_size, tag, epoch, step):
        records = []
        for rank in range(world_size):
            path = ValidationMixin._jsonl_path_static(log_dir, tag, rank, epoch, step)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    @staticmethod
    def _cleanup_val_jsonl_static(log_dir, world_size, tag, epoch, step):
        for rank in range(world_size):
            path = ValidationMixin._jsonl_path_static(log_dir, tag, rank, epoch, step)
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _save_final_predictions_static(log_dir, cls_data, gen_data, epoch, step):
        pred_dir = os.path.join(log_dir, "val_predictions")
        os.makedirs(pred_dir, exist_ok=True)
        path = os.path.join(pred_dir, f"predictions_epoch{epoch}_step{step}.json")
        payload = {
            "epoch": epoch, "global_step": step,
            "classification": cls_data, "generation": gen_data,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(cls_data)} cls + {len(gen_data)} gen predictions → {path}")

    def _save_final_predictions(self, cls_data: list, gen_data: list,
                                epoch: int = None, step: int = None):
        """Rank 0: 전체 prediction 결과를 영구 JSON 파일로 저장 (재현용).

        Classification: task, probs [P(False), P(True)], label
        Generation: task, strategy, pred_text, label_text
        """
        if epoch is None:
            epoch = self.current_epoch
        if step is None:
            step = self.global_step
        log_dir = self.trainer.log_dir or "."
        pred_dir = os.path.join(log_dir, "val_predictions")
        os.makedirs(pred_dir, exist_ok=True)

        filename = f"predictions_epoch{epoch}_step{step}.json"
        path = os.path.join(pred_dir, filename)

        payload = {
            "epoch": epoch,
            "global_step": step,
            "classification": cls_data,
            "generation": gen_data,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(cls_data)} cls + {len(gen_data)} gen predictions → {path}")

    # ─────────────────────────────────────────
    # Validation lifecycle
    # ─────────────────────────────────────────

    def setup(self, stage=None):
        """Trainer 연결 후 log_dir 기반으로 sample/stepwise logger 초기화."""
        if self._stepwise_logger is not None:
            return  # 이미 초기화됨

        # lightning_logs/version_N/ 경로 확보
        log_dir = self.trainer.log_dir or "."

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
        # Generation steps 일관성 검증 (첫 validation epoch에서 1회)
        # generate.py 실제 로직: num_blocks = gen_length // block_size
        #                        steps_per_block = steps // num_blocks
        # → steps가 num_blocks로 나누어 떨어져야 함
        gen_cfg = self.cfg.generation
        if "semi_ar" in gen_cfg.val_strategies:
            gen_len = self.cfg.data.gen_max_len
            block_size = gen_cfg.semi_ar.block_size
            num_blocks = gen_len // block_size
            if num_blocks == 0:
                raise ValueError(
                    f"semi_ar block_size({block_size}) > gen_max_len({gen_len})"
                )
            if gen_cfg.sampling_steps % num_blocks != 0:
                raise ValueError(
                    f"Generation steps 불일치: "
                    f"sampling_steps({gen_cfg.sampling_steps})는 "
                    f"num_blocks(gen_max_len={gen_len} // block_size={block_size} = {num_blocks})로 "
                    f"나누어 떨어져야 합니다."
                )

        # Open per-rank JSONL files for this epoch
        self._val_cls_fh = self._open_val_jsonl("cls")
        self._val_gen_fh = self._open_val_jsonl("gen")
        if self._stepwise_logger:
            self._stepwise_logger.reset()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        tasks = batch["tasks"]
        prompt_ids = batch["prompt_input_ids"]
        prompt_mask = batch["prompt_attention_mask"]
        target_texts = batch["target_texts"]
        input_mol_strings = batch.get("input_mol_strings", [""] * len(tasks))
        prompt_texts = batch.get("prompt_texts", [""] * len(tasks))

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
            # Write to JSONL (per-rank, per-sample)
            probs_cpu = probs.cpu()
            for i, ci in enumerate(cls_idx):
                self._write_jsonl(self._val_cls_fh, {
                    "task": tasks[ci],
                    "probs": probs_cpu[i].tolist(),
                    "label": target_texts[ci],
                    "input_mol_string": input_mol_strings[ci],
                    "prompt_text": prompt_texts[ci],
                })

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

                # Write to JSONL (per-rank, per-sample)
                for i in range(len(gen_tasks)):
                    self._write_jsonl(self._val_gen_fh, {
                        "task": gen_tasks[i],
                        "strategy": strategy_key,
                        "pred_text": pred_texts[i],
                        "label_text": gen_labels[i],
                        "input_mol_string": input_mol_strings[gen_idx[i]],
                        "prompt_text": prompt_texts[gen_idx[i]],
                    })

    def on_validation_epoch_end(self):
        print(f"[Rank {self.global_rank}] epoch_end: START", flush=True)

        # ── 1. Close JSONL files (모든 rank) ──
        if self._val_cls_fh:
            self._val_cls_fh.close()
            self._val_cls_fh = None
        if self._val_gen_fh:
            self._val_gen_fh.close()
            self._val_gen_fh = None
        print(f"[Rank {self.global_rank}] epoch_end: files closed", flush=True)

        # ── 2. Rank 0: 비동기 스레드로 metric 계산 시작 ──
        # 주의: self.trainer.log_dir은 내부적으로 broadcast()를 호출하므로
        # if 블록 바깥에서 모든 rank가 함께 호출해야 함
        val_epoch = self.current_epoch
        val_step = self.global_step
        log_dir = self.trainer.log_dir or "."
        world_size = self.trainer.world_size

        if self.global_rank == 0:
            import threading
            loggers = list(self.loggers)
            tokenizer = self.tokenizer
            print(f"[Rank 0] epoch_end: launching async thread", flush=True)
            t = threading.Thread(
                target=self._process_validation_async,
                args=(val_epoch, val_step, log_dir, world_size, loggers, tokenizer),
                daemon=True,
            )
            t.start()
            print(f"[Rank 0] epoch_end: thread launched", flush=True)

        print(f"[Rank {self.global_rank}] epoch_end: RETURN", flush=True)

    def _process_validation_async(self, epoch, step, log_dir, world_size,
                                   loggers, tokenizer):
        """Rank 0 전용 비동기 스레드: JSONL 로드 → metric 계산 → 로깅.

        별도 스레드에서 실행되므로 Lightning hook을 blocking하지 않음.
        self.log() 대신 logger.log_metrics()를 직접 호출 (thread-safe).
        주의: self.trainer 접근 금지 (내부적으로 NCCL broadcast 호출함).
        """
        try:
            print(f"[Async] loading JSONL (epoch={epoch}, step={step})...", flush=True)
            cls_data = self._load_all_val_predictions_static(
                log_dir, world_size, "cls", epoch, step)
            gen_data = self._load_all_val_predictions_static(
                log_dir, world_size, "gen", epoch, step)
            print(f"[Async] loaded {len(cls_data)} cls + {len(gen_data)} gen", flush=True)

            val_metrics = {}

            # --- Classification metrics ---
            cls_by_task = defaultdict(lambda: {"probs": [], "labels": []})
            for item in cls_data:
                cls_by_task[item["task"]]["probs"].append(item["probs"])
                cls_by_task[item["task"]]["labels"].append(item["label"])

            for task, data in cls_by_task.items():
                all_probs = torch.tensor(data["probs"])
                metrics = classification_evaluate(all_probs, data["labels"], task)
                for k, v in metrics.items():
                    val_metrics[f"val/{task}/{k}"] = v

            # --- Generation metrics (strategy별 분리) ---
            gen_by_key = defaultdict(lambda: {"preds": [], "labels": []})
            for item in gen_data:
                gen_by_key[(item["task"], item["strategy"])]["preds"].append(item["pred_text"])
                gen_by_key[(item["task"], item["strategy"])]["labels"].append(item["label_text"])

            for (task, strategy), data in gen_by_key.items():
                task_type = get_task_type(task)
                if task_type == "regression":
                    metrics = regression_evaluate(data["preds"], data["labels"], task)
                elif task_type == "molecule":
                    metrics = molecule_evaluate(data["preds"], data["labels"], task,
                                                tokenizer=tokenizer)
                elif task_type == "caption":
                    metrics = caption_evaluate(data["preds"], data["labels"], task,
                                                tokenizer=tokenizer)
                else:
                    continue

                for k, v in metrics.items():
                    val_metrics[f"val/{task}/{strategy}/{k}"] = v

            # 직접 logger 호출 (self.log() 대신 — thread-safe)
            print(f"[Async] computing metrics done, logging {len(val_metrics)} metrics...", flush=True)
            flat = {}
            for k, v in val_metrics.items():
                flat[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
            for lg in loggers:
                lg.log_metrics(flat, step=step)
            print(f"[Async] logger.log_metrics done", flush=True)

            # 영구 prediction 저장
            print(f"[Async] saving predictions...", flush=True)
            self._save_final_predictions_static(
                log_dir, cls_data, gen_data, epoch, step)

            # Cleanup temp JSONL
            self._cleanup_val_jsonl_static(log_dir, world_size, "cls", epoch, step)
            self._cleanup_val_jsonl_static(log_dir, world_size, "gen", epoch, step)
            print(f"[Async] ALL DONE (predictions saved, JSONL cleaned)", flush=True)
        except Exception as e:
            print(f"[Async] FAILED: {e}", flush=True)
            logger.error(f"[Async] Validation metric processing failed: {e}", exc_info=True)

    # ─────────────────────────────────────────
    # WandB helpers
    # ─────────────────────────────────────────

    def _get_wandb_logger(self):
        """WandbLogger가 있으면 반환, 없으면 None."""
        for lg in self.loggers:
            if type(lg).__name__ == "WandbLogger":
                return lg
        return None
