"""Stepwise denoising logger — diffusion step별 [MASK] → token 변화 과정 기록.

Deferred Write 패턴:
    1. generate_with_logging() 루프 내: tensor.clone().cpu() 만 수집 (수 μs)
    2. 루프 완료 후: write_stepwise_log() 에서 일괄 decode + 1회 file write
"""

import logging
import os
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

MASK_TOKEN = "[MASK]"
TOKENS_PER_LINE = 6
TOKEN_WIDTH = 18


class StepwiseLogger:
    """Diffusion step별 token 변화를 TXT로 기록. max_samples 이내만 로깅."""

    def __init__(self, log_dir: str, max_samples: int = 8, enabled: bool = False):
        self.log_dir = os.path.join(log_dir, "stepwise_logs")
        self.max_samples = max_samples
        self.enabled = enabled
        self._sample_count = 0

    def should_log(self) -> bool:
        """현재 샘플을 로깅할지 결정."""
        return self.enabled and self._sample_count < self.max_samples

    def reset(self):
        """epoch 시작 시 카운터 리셋."""
        self._sample_count = 0

    def write_stepwise_log(
        self,
        task: str,
        epoch: int,
        global_step: int,
        target_text: str,
        step_snapshots: List[torch.Tensor],
        tokenizer,
        config: dict,
        mask_id: int = 126336,
    ):
        """Generation 완료 후 호출 — snapshot 리스트를 일괄 decode + 1회 file write.

        Args:
            task: task 이름 (e.g., "MoleculeGeneration")
            epoch: 현재 epoch
            global_step: 현재 global step
            target_text: 정답 텍스트
            step_snapshots: list[Tensor] — 각 step의 gen_tokens [gen_length] (CPU)
            tokenizer: HuggingFace tokenizer
            config: {"steps": int, "remasking": str} 등 generation 설정
            mask_id: MASK token ID (default: 126336)
        """
        if not step_snapshots:
            return

        os.makedirs(self.log_dir, exist_ok=True)

        counter = self._sample_count
        self._sample_count += 1

        remasking_tag = config.get("remasking", "unk")
        sampling_tag = "semi_ar" if config.get("semi_ar", False) else "standard"
        filename = (
            f"epoch{epoch:02d}_step{global_step}"
            f"_{task}_{sampling_tag}_{remasking_tag}_{counter:04d}.txt"
        )
        filepath = os.path.join(self.log_dir, filename)

        total_steps = len(step_snapshots)
        gen_length = step_snapshots[0].shape[-1]

        # 모든 step의 token을 한 번에 decode하기 위해 준비
        # step_snapshots: list of [gen_length] tensors
        all_tokens = torch.stack(step_snapshots, dim=0)  # [total_steps, gen_length]

        lines = []

        # Header
        sep = "=" * 64
        lines.append(sep)
        lines.append(f"[Sample Info] task={task} | epoch={epoch} | global_step={global_step}")
        sampling_tag = "semi_ar" if config.get("semi_ar", False) else "standard"
        lines.append(
            f"[Config] steps={config.get('steps', '?')} | "
            f"remasking={config.get('remasking', '?')} | "
            f"sampling={sampling_tag}"
        )
        lines.append(f"[Target] {target_text}")
        lines.append(sep)
        lines.append("")

        # 각 step 기록
        for step_idx in range(total_steps):
            tokens = all_tokens[step_idx]  # [gen_length]
            mask_count = (tokens == mask_id).sum().item()
            unmasked = gen_length - mask_count
            pct = (unmasked / gen_length) * 100 if gen_length > 0 else 0

            lines.append(
                f"[Step {step_idx:3d}/{total_steps}] "
                f"Unmasked: {unmasked:3d}/{gen_length} ({pct:5.1f}%)"
            )

            # Token을 문자열로 변환 (MASK는 [MASK] 표시)
            token_strs = []
            for t in tokens.tolist():
                if t == mask_id:
                    token_strs.append(MASK_TOKEN)
                else:
                    token_strs.append(tokenizer.decode([t]))

            # 고정폭 포맷으로 TOKENS_PER_LINE개씩 출력
            for row_start in range(0, len(token_strs), TOKENS_PER_LINE):
                row_end = min(row_start + TOKENS_PER_LINE, len(token_strs))
                row_tokens = token_strs[row_start:row_end]
                formatted = "".join(t.ljust(TOKEN_WIDTH) for t in row_tokens)
                lines.append(f"  [{row_start:3d}-{row_end-1:3d}] {formatted}")

            lines.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Stepwise log ({total_steps} steps) saved to {filepath}")
