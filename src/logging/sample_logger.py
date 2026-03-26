"""Validation sample logger — GPU당 N개 sample을 읽기 쉬운 TXT로 저장.

사용 흐름:
    1. on_validation_epoch_start → reset()
    2. validation_step           → collect_classification() / collect_generation()
    3. on_validation_epoch_end   → flush()
"""

import json
import logging
import os
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


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
            label: 정답 텍스트 ("Yes" / "No")
        """
        if len(self._cls_samples) >= self.samples_per_gpu:
            return
        pred_label = "Yes" if prob[1].item() > prob[0].item() else "No"
        self._cls_samples.append({
            "task": task,
            "target": label,
            "prediction": pred_label,
            "prob_true": round(prob[1].item(), 4),
            "correct": pred_label == label,
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
