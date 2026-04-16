"""Training prediction logger — train step에서 masked position별 예측 상세 기록.

사용 흐름:
    1. setup()에서 TrainPredictionLogger 초기화
    2. training_step()에서 should_log() 확인 → batch에 _log_train_detail 플래그 설정
    3. model forward()가 prediction detail 반환
    4. write_sample_log()로 TXT 저장

파일 구조:
    {log_dir}/train_predictions/{task}/epoch{XX}.txt
    모든 rank가 동일 파일에 append하며, fcntl.flock으로 동시 쓰기를 직렬화한다.
"""

import fcntl
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

TOKEN_WIDTH = 20


class TrainPredictionLogger:
    """Train step의 masked position별 prediction 상세를 주기적으로 TXT로 기록.

    global_step 기준으로 로깅 여부를 결정하되, 동일 step 내
    accumulation micro-batch 중복을 _last_logged_step으로 방지한다.
    모든 GPU(rank)에서 독립적으로 기록하여 multi-task sampling 커버리지를 확보한다.
    """

    def __init__(
        self,
        log_dir: str,
        log_interval: int = 100,
        max_positions: int = 50,
        enabled: bool = True,
    ):
        """
        Args:
            log_dir: 로그 파일 저장 디렉토리 (하위에 train_predictions/{task}/ 생성)
            log_interval: 상세 로그 기록 간격 (global steps)
            max_positions: 출력할 최대 masked position 수
            enabled: False면 모든 로깅 비활성화
        """
        self.log_dir = os.path.join(log_dir, "train_predictions")
        self.log_interval = max(1, log_interval)
        self.max_positions = max_positions
        self.enabled = enabled
        self._last_logged_step: int = -1

    def should_log(self, global_step: int) -> bool:
        """현재 step에서 상세 로그를 기록할지 결정.

        global_step 기준 interval 체크 + 동일 step 중복 방지.
        """
        if not self.enabled:
            return False
        if global_step % self.log_interval != 0:
            return False
        if global_step == self._last_logged_step:
            return False
        return True

    def write_sample_log(
        self,
        global_step: int,
        epoch: int,
        rank: int,
        task: str,
        p_mask: float,
        mask_positions: torch.Tensor,
        target_tokens: torch.Tensor,
        pred_tokens: torch.Tensor,
        target_probs: torch.Tensor,
        pred_probs: torch.Tensor,
        tokenizer,
        # Full-sequence data for human-readable text sections
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        all_answer_pred_ids: Optional[torch.Tensor] = None,
        all_answer_gt_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """1개 sample의 masked position별 prediction 상세를 TXT로 저장.

        파일 경로: {log_dir}/{task}/epoch{XX}.txt
        여러 rank가 동시에 쓸 수 있으므로 fcntl.flock으로 직렬화한다.

        Args:
            global_step: 현재 global step
            epoch: 현재 epoch
            rank: GPU rank (헤더에 표시용)
            task: task 이름 (폴더 분리 기준)
            p_mask: masking probability (0~1)
            mask_positions: [N] masked position indices (CPU)
            target_tokens: [N] ground truth token ids (CPU)
            pred_tokens: [N] predicted token ids (CPU)
            target_probs: [N] ground truth token softmax probs (CPU)
            pred_probs: [N] predicted token softmax probs (CPU)
            tokenizer: HuggingFace tokenizer for decoding
        """
        if not self.enabled:
            return

        # 동일 step 중복 기록 방지
        self._last_logged_step = global_step

        # {log_dir}/{task}/
        task_dir = os.path.join(self.log_dir, task)
        os.makedirs(task_dir, exist_ok=True)

        n_positions = len(mask_positions)
        n_correct = (pred_tokens == target_tokens).sum().item()
        avg_target_prob = target_probs.mean().item() if n_positions > 0 else 0.0

        # EOS 제외 accuracy
        eos_id = tokenizer.eos_token_id
        non_eos = target_tokens != eos_id
        n_positions_no_eos = int(non_eos.sum().item())
        n_correct_no_eos = int((pred_tokens[non_eos] == target_tokens[non_eos]).sum().item()) if n_positions_no_eos > 0 else 0

        filename = f"epoch{epoch:02d}.txt"
        filepath = os.path.join(task_dir, filename)

        # 문자열 전체를 미리 조립 (lock 구간 최소화)
        lines = []
        sep = "=" * 80

        # Header
        lines.append(sep)
        lines.append(
            f"[Train Sample] step={global_step} | epoch={epoch} | "
            f"rank={rank} | task={task} | mask_ratio={p_mask:.3f}"
        )
        lines.append(
            f"[Summary] accuracy={n_correct}/{n_positions} "
            f"({n_correct / max(1, n_positions) * 100:.1f}%) | "
            f"accuracy_no_eos={n_correct_no_eos}/{n_positions_no_eos} "
            f"({n_correct_no_eos / max(1, n_positions_no_eos) * 100:.1f}%) | "
            f"avg_target_prob={avg_target_prob:.4f}"
        )
        lines.append(sep)
        lines.append("")

        # --- Human-readable Input / Output / Prediction text ---
        if input_ids is not None and labels is not None:
            # Determine real content length (exclude padding)
            if attention_mask is not None:
                content_len = int(attention_mask.sum().item())
            else:
                content_len = len(input_ids)

            # Find prompt/answer boundary (first position where labels != -100)
            answer_positions = (labels != -100).nonzero(as_tuple=False)
            if len(answer_positions) > 0:
                answer_start = answer_positions[0].item()
            else:
                answer_start = content_len

            # Decode prompt text
            prompt_token_ids = input_ids[:answer_start].tolist()
            prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)

            # Decode ground truth answer (trim padding)
            real_answer_len = max(0, content_len - answer_start)
            if all_answer_gt_ids is not None:
                gt_ids = all_answer_gt_ids[:real_answer_len].tolist()
            else:
                gt_ids = input_ids[answer_start:content_len].tolist()
            gt_text = tokenizer.decode(gt_ids, skip_special_tokens=False)

            # Decode model prediction for answer region
            if all_answer_pred_ids is not None:
                pred_ids = all_answer_pred_ids[:real_answer_len].tolist()
                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=False)
            else:
                pred_text = "(not available)"

            lines.append("[Input (Prompt)]")
            lines.append(prompt_text)
            lines.append("")
            lines.append("[Output (Ground Truth)]")
            lines.append(gt_text)
            lines.append("")
            lines.append("[Prediction (Model Output)]")
            lines.append(pred_text)
            lines.append("")
            lines.append("-" * 80)
            lines.append("")

        # Column header
        lines.append(
            f"{'Pos':>5}   {'Ground Truth':{TOKEN_WIDTH}}   "
            f"{'Predicted':{TOKEN_WIDTH}}   {'GT_Prob':>7}  {'Pred_Prob':>9}  Match"
        )
        lines.append(
            f"{'---':>5}   {'-' * TOKEN_WIDTH}   "
            f"{'-' * TOKEN_WIDTH}   {'-------':>7}  {'---------':>9}  -----"
        )

        # Token rows (최대 max_positions개)
        display_count = min(n_positions, self.max_positions)
        for i in range(display_count):
            pos = mask_positions[i].item()
            gt_tok = tokenizer.decode([target_tokens[i].item()])
            pred_tok = tokenizer.decode([pred_tokens[i].item()])
            gt_prob = target_probs[i].item()
            pred_prob = pred_probs[i].item()
            match = "\u2713" if target_tokens[i] == pred_tokens[i] else "\u2717"

            # Truncate long tokens for display
            gt_tok = gt_tok[:TOKEN_WIDTH]
            pred_tok = pred_tok[:TOKEN_WIDTH]

            lines.append(
                f"{pos:5d}   {gt_tok:{TOKEN_WIDTH}}   "
                f"{pred_tok:{TOKEN_WIDTH}}   {gt_prob:7.4f}  {pred_prob:9.4f}  {match}"
            )

        if n_positions > self.max_positions:
            lines.append(f"  ... ({n_positions - self.max_positions} more positions omitted)")

        lines.append("")

        content = "\n".join(lines)

        # 파일 잠금으로 multi-rank 동시 쓰기 직렬화
        with open(filepath, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(content)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        logger.info(
            f"Train prediction log (rank={rank}, task={task}, "
            f"acc={n_correct}/{n_positions}, "
            f"acc_no_eos={n_correct_no_eos}/{n_positions_no_eos}, "
            f"target_prob={avg_target_prob:.4f}) saved to {filepath}"
        )
