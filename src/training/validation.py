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
        """Rank 0: 모든 rank의 JSONL 파일을 로드하여 병합 + padding duplicate dedup.

        (val_idx, strategy) key로 unique. DDP DistributedSampler가 dataset을
        world_size 배수로 padding할 때 발생하는 중복 제거.
        """
        records = []
        seen_keys = set()
        dup_count = 0
        world_size = self.trainer.world_size
        for rank in range(world_size):
            path = self._val_jsonl_path(tag, rank=rank, epoch=epoch, step=step)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    vidx = rec.get("val_idx", -1)
                    if vidx >= 0:
                        key = (vidx, rec.get("strategy", ""))
                        if key in seen_keys:
                            dup_count += 1
                            continue
                        seen_keys.add(key)
                    records.append(rec)
        if dup_count:
            logger.info(f"dedup: removed {dup_count} padding-duplicate records for tag={tag}")
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
        """JSONL들을 로드 + (val_idx, strategy) 기준 dedup.

        DDP DistributedSampler가 dataset을 world_size 배수로 padding할 때 앞쪽
        sample 일부가 rank에 중복 등장. 각 sample에 원본 _val_idx가 기록되어
        있으므로 (val_idx, strategy) key로 중복 제거. 같은 sample이 여러 rank에
        기록돼도 첫 번째만 keep.

        - classification: strategy 없음 → val_idx만으로 유일
        - generation: strategy 포함 (strategy별 별도 prediction 보존)
        - val_idx == -1 (sentinel): dedup 불가 → 그대로 keep (레거시 호환)
        """
        records = []
        seen_keys = set()
        dup_count = 0
        for rank in range(world_size):
            path = ValidationMixin._jsonl_path_static(log_dir, tag, rank, epoch, step)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    vidx = rec.get("val_idx", -1)
                    if vidx >= 0:
                        key = (vidx, rec.get("strategy", ""))
                        if key in seen_keys:
                            dup_count += 1
                            continue
                        seen_keys.add(key)
                    records.append(rec)
        if dup_count:
            print(
                f"[Async] dedup: removed {dup_count} padding-duplicate records "
                f"for tag={tag}",
                flush=True,
            )
        return records

    @staticmethod
    def _cleanup_val_jsonl_static(log_dir, world_size, tag, epoch, step):
        for rank in range(world_size):
            path = ValidationMixin._jsonl_path_static(log_dir, tag, rank, epoch, step)
            if os.path.exists(path):
                os.remove(path)

    def _update_val_chart_history(self, task: str, metric_name: str,
                                   strategy: str, epoch: int, value):
        """Strategy 비교 line_series 차트용 누적 history 업데이트.

        History 구조: {(task, metric): {"epochs": [E0, E1, ...],
                                        "strategies": {strategy_name: [v0, v1, ...]}}}
        서로 다른 strategy의 길이를 epoch 배열 기준으로 NaN padding하여 맞춘다.
        """
        key = (task, metric_name)
        hist = self._val_custom_chart_history.setdefault(
            key, {"epochs": [], "strategies": {}}
        )
        if not hist["epochs"] or hist["epochs"][-1] != epoch:
            hist["epochs"].append(epoch)
            for vals in hist["strategies"].values():
                while len(vals) < len(hist["epochs"]) - 1:
                    vals.append(float("nan"))
        if strategy not in hist["strategies"]:
            hist["strategies"][strategy] = [float("nan")] * (len(hist["epochs"]) - 1)
        v = value.item() if hasattr(value, "item") else float(value)
        # 현재 epoch 값을 기록 (이미 있으면 마지막 값을 덮어씀)
        vals = hist["strategies"][strategy]
        if len(vals) == len(hist["epochs"]):
            vals[-1] = v
        else:
            vals.append(v)

    def _log_strategy_comparison_charts(self, loggers, step: int):
        """strategy 2개 이상인 (task, metric)마다 wandb.plot.line_series로 line 비교 차트 로깅.

        Chart 직렬화는 `/tmp/tmp*wandb-media/` 임시 경로에 파일을 쓰는데, 이 디렉터리가
        (a) 외부 /tmp cleanup, (b) async thread에서 stale run 접근, (c) run 재초기화로
        인해 사라져 있으면 FileNotFoundError가 난다. Chart 실패가 뒤의 prediction 저장을
        막지 않도록 각 chart 호출을 격리하고, media tempdir은 호출 직전에 복구한다.
        """
        try:
            import wandb
        except ImportError:
            return
        wandb_logger = None
        for lg in loggers:
            if type(lg).__name__ == "WandbLogger":
                wandb_logger = lg
                break
        if wandb_logger is None:
            return

        # wandb media tempdir이 없으면 복구(없으면 bind_to_run에서 FileNotFoundError)
        try:
            run = wandb_logger.experiment
            media_dir = getattr(run, "_settings", None)
            media_dir = getattr(media_dir, "_tmp_dir", None) if media_dir else None
            if media_dir and not os.path.isdir(str(media_dir)):
                os.makedirs(str(media_dir), exist_ok=True)
        except Exception as e:
            logger.warning(f"[charts] media tempdir check skipped: {e}")

        for (task, metric_name), hist in self._val_custom_chart_history.items():
            if len(hist["strategies"]) < 2:
                continue
            try:
                # strategy별 ys 길이를 epochs 길이에 맞춤
                n_epochs = len(hist["epochs"])
                ys = []
                keys = []
                for s_name, vals in hist["strategies"].items():
                    padded = list(vals) + [float("nan")] * (n_epochs - len(vals))
                    ys.append(padded)
                    keys.append(s_name)
                chart = wandb.plot.line_series(
                    xs=hist["epochs"],
                    ys=ys,
                    keys=keys,
                    title=f"{task} — {metric_name}",
                    xname="epoch",
                )
                wandb_logger.experiment.log(
                    {f"val_chart/{metric_name}/{task}": chart}, step=step
                )
            except Exception as e:
                logger.warning(
                    f"[charts] skip val_chart/{metric_name}/{task}: {e}"
                )

    @staticmethod
    def _save_failed_per_task_static(log_dir, task, strategy, epoch, step, records):
        """실패 샘플을 {log_dir}/val_predictions/failed/{task}/ 아래에 JSON으로 저장."""
        if not records:
            return
        fail_dir = os.path.join(log_dir, "val_predictions", "failed", task)
        os.makedirs(fail_dir, exist_ok=True)
        filename = f"epoch{epoch}_step{step}_{strategy or 'cls'}.json"
        path = os.path.join(fail_dir, filename)
        payload = {
            "task": task,
            "strategy": strategy,
            "epoch": epoch,
            "global_step": step,
            "num_failed": len(records),
            "failed_samples": records,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(records)} failed samples → {path}")

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
        val_indices = batch.get("val_indices", [-1] * len(tasks))

        # Split by task type
        cls_idx = [i for i, t in enumerate(tasks) if t in CLASSIFICATION_TASKS]
        gen_idx = [i for i, t in enumerate(tasks)
                   if t not in CLASSIFICATION_TASKS and t not in NAME_CONVERSION_TASKS]

        # ── Coverage debug: trace which (rank, batch_idx) got processed ──
        _cov_trace = os.environ.get("MOLDA_COV_TRACE") == "1"
        if _cov_trace:
            from collections import Counter as _C
            _task_counts = dict(_C(tasks))
            print(
                f"[COV rank={self.global_rank} batch={batch_idx}] "
                f"size={len(tasks)} cls={len(cls_idx)} gen={len(gen_idx)} "
                f"skipped={len(tasks) - len(cls_idx) - len(gen_idx)} "
                f"tasks={_task_counts}",
                flush=True,
            )

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
            _cls_written = 0
            _cls_failed = []
            for i, ci in enumerate(cls_idx):
                try:
                    self._write_jsonl(self._val_cls_fh, {
                        "val_idx": int(val_indices[ci]),
                        "task": tasks[ci],
                        "probs": probs_cpu[i].tolist(),
                        "label": target_texts[ci],
                        "input_mol_string": input_mol_strings[ci],
                        "prompt_text": prompt_texts[ci],
                    })
                    _cls_written += 1
                except Exception as _e:
                    _cls_failed.append((tasks[ci], type(_e).__name__, str(_e)[:100]))
            if _cov_trace or _cls_failed:
                print(
                    f"[COV rank={self.global_rank} batch={batch_idx}] "
                    f"cls_written={_cls_written}/{len(cls_idx)} failed={_cls_failed}",
                    flush=True,
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

                # Write to JSONL (per-rank, per-sample)
                _written = 0
                _failed = []
                for i in range(len(gen_tasks)):
                    try:
                        self._write_jsonl(self._val_gen_fh, {
                            "val_idx": int(val_indices[gen_idx[i]]),
                            "task": gen_tasks[i],
                            "strategy": strategy_key,
                            "pred_text": pred_texts[i],
                            "label_text": gen_labels[i],
                            "input_mol_string": input_mol_strings[gen_idx[i]],
                            "prompt_text": prompt_texts[gen_idx[i]],
                        })
                        _written += 1
                    except Exception as _e:
                        _failed.append((gen_tasks[i], type(_e).__name__, str(_e)[:100]))
                if _cov_trace or _failed:
                    print(
                        f"[COV rank={self.global_rank} batch={batch_idx} "
                        f"strategy={strategy_key}] gen_written={_written}/"
                        f"{len(gen_tasks)} failed={_failed}",
                        flush=True,
                    )

    def on_validation_epoch_end(self):
        print(f"[Rank {self.global_rank}] epoch_end: START", flush=True)

        # ── 1. Close JSONL files (모든 rank) ──
        import os as _os
        for fh_attr in ("_val_cls_fh", "_val_gen_fh"):
            fh = getattr(self, fh_attr)
            if fh:
                fh.flush()
                try:
                    _os.fsync(fh.fileno())  # OS 버퍼까지 flush (타 rank read 안전)
                except (OSError, ValueError):
                    pass
                fh.close()
                setattr(self, fh_attr, None)
        print(f"[Rank {self.global_rank}] epoch_end: files closed", flush=True)

        # ── 1.5. Barrier: 모든 rank의 close가 완료된 후 async 진행 ──
        # 미동기 시 rank 0 async thread가 다른 rank의 partial JSONL을 읽는 race.
        if self.trainer.world_size > 1:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        print(f"[Rank {self.global_rank}] epoch_end: barrier passed", flush=True)

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
            cls_by_task = defaultdict(lambda: {"probs": [], "labels": [], "records": []})
            for item in cls_data:
                cls_by_task[item["task"]]["probs"].append(item["probs"])
                cls_by_task[item["task"]]["labels"].append(item["label"])
                cls_by_task[item["task"]]["records"].append(item)

            for task, data in cls_by_task.items():
                all_probs = torch.tensor(data["probs"])
                metrics = classification_evaluate(all_probs, data["labels"], task)
                failure_idxs = metrics.pop("_failure_indices", [])
                for k, v in metrics.items():
                    val_metrics[f"val/{k}/{task}"] = v
                if failure_idxs:
                    failed_records = [data["records"][i] for i in failure_idxs]
                    self._save_failed_per_task_static(
                        log_dir, task, None, epoch, step, failed_records)

            # --- Generation metrics (strategy별 분리) ---
            gen_by_key = defaultdict(lambda: {"preds": [], "labels": [], "records": []})
            for item in gen_data:
                key = (item["task"], item["strategy"])
                gen_by_key[key]["preds"].append(item["pred_text"])
                gen_by_key[key]["labels"].append(item["label_text"])
                gen_by_key[key]["records"].append(item)

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

                failure_idxs = metrics.pop("_failure_indices", [])
                for k, v in metrics.items():
                    val_metrics[f"val/{k}/{task}/{strategy}"] = v
                    # strategy 비교 custom chart history 업데이트
                    self._update_val_chart_history(task, k, strategy, epoch, v)
                if failure_idxs:
                    failed_records = [data["records"][i] for i in failure_idxs]
                    self._save_failed_per_task_static(
                        log_dir, task, strategy, epoch, step, failed_records)

            # 직접 logger 호출 (self.log() 대신 — thread-safe)
            print(f"[Async] computing metrics done, logging {len(val_metrics)} metrics...", flush=True)
            flat = {}
            for k, v in val_metrics.items():
                flat[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
            for lg in loggers:
                lg.log_metrics(flat, step=step)
            print(f"[Async] logger.log_metrics done", flush=True)

            # 영구 prediction 저장 (chart 로깅 실패에 영향받지 않도록 먼저 수행)
            print(f"[Async] saving predictions...", flush=True)
            self._save_final_predictions_static(
                log_dir, cls_data, gen_data, epoch, step)

            # Cleanup temp JSONL
            self._cleanup_val_jsonl_static(log_dir, world_size, "cls", epoch, step)
            self._cleanup_val_jsonl_static(log_dir, world_size, "gen", epoch, step)
            print(f"[Async] predictions saved, JSONL cleaned", flush=True)

            # Strategy 비교 custom chart (wandb line_series) 로깅 — 실패 시 경고만
            try:
                self._log_strategy_comparison_charts(loggers, step)
            except Exception as e:
                logger.warning(f"[Async] chart logging failed (ignored): {e}")
            print(f"[Async] ALL DONE", flush=True)
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
