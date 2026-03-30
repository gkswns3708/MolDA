"""Validation sample logger — GPU당 N개 sample을 읽기 쉬운 TXT로 저장.

사용 흐름:
    1. on_validation_epoch_start → reset()
    2. validation_step           → collect_classification() / collect_generation()
    3. on_validation_epoch_end   → flush()
"""

import logging
import os
import re

import torch

logger = logging.getLogger(__name__)

_BOOLEAN_PATTERN = re.compile(r"<BOOLEAN>\s*(True|False)\s*</BOOLEAN>", re.IGNORECASE)


def _parse_boolean_label(label: str) -> bool:
    """Extract True/False from '<BOOLEAN> True </BOOLEAN>' style label text."""
    m = _BOOLEAN_PATTERN.search(label)
    if m:
        return m.group(1).strip().lower() == "true"
    # Fallback: look for bare True/False
    lower = label.lower()
    return "true" in lower and "false" not in lower


class ValidationSampleLogger:
    """Validation 시 GPU당 제한된 수의 sample을 수집하고 epoch 끝에 일괄 저장."""

    def __init__(self, log_dir: str, samples_per_gpu: int = 1):
        self.sample_dir = os.path.join(log_dir, "val_samples")
        self.samples_per_gpu = samples_per_gpu
        self._cls_samples: list = []
        self._gen_samples: list = []

    def reset(self):
        """epoch 시작 시 버퍼 초기화."""
        self._cls_samples = []
        self._gen_samples = []

    def collect_classification(self, task: str, prob: torch.Tensor, label: str):
        """Classification sample 1개를 버퍼에 추가 (제한 이내일 때만).

        Args:
            task: task 이름 (e.g., "BBBP", "HIV")
            prob: [2] tensor — [prob_false, prob_true]
            label: 정답 텍스트 (e.g., "<BOOLEAN> True </BOOLEAN>" or "<BOOLEAN> True </BOOLEAN><|eot_id|>")
        """
        if len(self._cls_samples) >= self.samples_per_gpu:
            return
        pred_true = prob[1].item() > prob[0].item()
        # Parse ground truth from <BOOLEAN> tag
        gt_true = _parse_boolean_label(label)
        correct = pred_true == gt_true
        pred_label = "True" if pred_true else "False"
        self._cls_samples.append({
            "task": task,
            "target": label,
            "prediction": pred_label,
            "prob_true": round(prob[1].item(), 4),
            "correct": correct,
        })

    def collect_generation(self, task: str, pred: str, label: str, strategy: str = ""):
        """Generation sample 1개를 버퍼에 추가 (제한 이내일 때만).

        Args:
            task: task 이름 (e.g., "MoleculeGeneration", "MoleculeCaption")
            pred: 예측 텍스트
            label: 정답 텍스트
            strategy: remasking strategy 이름 (e.g., "low_confidence", "random")
        """
        if len(self._gen_samples) >= self.samples_per_gpu:
            return
        self._gen_samples.append({
            "task": task,
            "strategy": strategy,
            "target": label,
            "prediction": pred,
            "exact_match": pred.strip() == label.strip(),
        })

    def flush(self, epoch: int, global_step: int, rank: int = 0):
        """수집된 모든 sample을 TXT 파일로 일괄 저장 후 버퍼 초기화.

        파일: val_samples/epoch{NN}_step{NNNNN}.txt
        """
        if not self._cls_samples and not self._gen_samples:
            return

        os.makedirs(self.sample_dir, exist_ok=True)
        filename = f"epoch{epoch:02d}_step{global_step}_rank{rank}.txt"
        filepath = os.path.join(self.sample_dir, filename)

        lines = []
        header = f"Validation Samples | Epoch {epoch} | Global Step {global_step} | GPU {rank}"
        sep = "=" * 64
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
        lines.append("")

        # Classification samples
        for s in self._cls_samples:
            mark = "O" if s["correct"] else "X"
            lines.append(f"[Classification] Task: {s['task']}")
            lines.append(f"  Target    : {s['target']}")
            lines.append(f"  Prediction: {s['prediction']} (prob: {s['prob_true']:.4f})")
            lines.append(f"  Correct   : {mark}")
            lines.append("")

        # Generation samples
        for s in self._gen_samples:
            match_str = "EXACT" if s["exact_match"] else "MISMATCH"
            strategy_tag = f" | Strategy: {s['strategy']}" if s.get("strategy") else ""
            lines.append(f"[Generation] Task: {s['task']}{strategy_tag}")
            lines.append(f"  Target    : {s['target']}")
            lines.append(f"  Prediction: {s['prediction']}")
            lines.append(f"  Match     : {match_str}")
            lines.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(
            f"Saved {len(self._cls_samples)} cls + {len(self._gen_samples)} gen "
            f"samples to {filepath}"
        )

        # 버퍼 초기화
        self._cls_samples = []
        self._gen_samples = []

    def flush_to_wandb(self, experiment, epoch: int, global_step: int):
        """수집된 sample을 WandB Table로 로깅. flush() 전에 호출해야 함.

        Args:
            experiment: wandb.Run 객체 (wandb_logger.experiment)
            epoch: 현재 epoch
            global_step: 현재 global step
        """
        import wandb

        if self._cls_samples:
            cls_table = wandb.Table(
                columns=["epoch", "step", "task", "target", "prediction",
                         "prob_true", "correct"]
            )
            for s in self._cls_samples:
                cls_table.add_data(
                    epoch, global_step, s["task"], s["target"],
                    s["prediction"], s["prob_true"], s["correct"],
                )
            experiment.log({"val/classification_samples": cls_table})

        if self._gen_samples:
            gen_table = wandb.Table(
                columns=["epoch", "step", "task", "strategy", "target",
                         "prediction", "exact_match"]
            )
            for s in self._gen_samples:
                gen_table.add_data(
                    epoch, global_step, s["task"], s.get("strategy", ""),
                    s["target"], s["prediction"], s["exact_match"],
                )
            experiment.log({"val/generation_samples": gen_table})
