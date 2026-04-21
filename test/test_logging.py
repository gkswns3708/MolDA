"""Tests for StepwiseLogger and TrainPredictionLogger."""

import os

import pytest
import torch

from src.loggers.stepwise_logger import StepwiseLogger
from src.loggers.train_prediction_logger import TrainPredictionLogger


# ─────────────────────────────────────────────
# StepwiseLogger
# ─────────────────────────────────────────────

class TestStepwiseLogger:

    def test_should_log_enabled(self, tmp_path):
        sl = StepwiseLogger(log_dir=str(tmp_path), max_samples=5, enabled=True)
        assert sl.should_log() is True

    def test_should_log_disabled(self, tmp_path):
        sl = StepwiseLogger(log_dir=str(tmp_path), max_samples=5, enabled=False)
        assert sl.should_log() is False

    def test_should_log_exceeds_max(self, tmp_path):
        sl = StepwiseLogger(log_dir=str(tmp_path), max_samples=2, enabled=True)
        sl._sample_count = 2
        assert sl.should_log() is False

    def test_reset_counter(self, tmp_path):
        sl = StepwiseLogger(log_dir=str(tmp_path), max_samples=5, enabled=True)
        sl._sample_count = 3
        sl.reset()
        assert sl._sample_count == 0

    def test_write_creates_file(self, tmp_path):
        sl = StepwiseLogger(log_dir=str(tmp_path), max_samples=5, enabled=True)

        # Create mock snapshots (3 steps, gen_length=8)
        snapshots = [torch.randint(0, 100, (8,)) for _ in range(3)]
        # Set some positions as mask
        snapshots[0][:] = 126336  # All masked
        snapshots[1][:4] = 126336  # Half masked

        # Mock tokenizer
        class MockTokenizer:
            def decode(self, ids):
                return f"tok_{ids[0]}"

        sl.write_stepwise_log(
            task="reaction",
            epoch=1,
            global_step=500,
            target_text="<SELFIES>[C][O]</SELFIES>",
            step_snapshots=snapshots,
            tokenizer=MockTokenizer(),
            config={"steps": 3, "remasking": "low_confidence"},
        )

        log_dir = tmp_path / "stepwise_logs"
        assert log_dir.exists()
        files = list(log_dir.iterdir())
        assert len(files) == 1
        content = files[0].read_text()
        assert "reaction" in content
        assert "[MASK]" in content
        assert "[Step" in content

    def test_write_increments_counter(self, tmp_path):
        sl = StepwiseLogger(log_dir=str(tmp_path), max_samples=5, enabled=True)
        snapshots = [torch.randint(0, 100, (4,))]

        class MockTokenizer:
            def decode(self, ids):
                return "t"

        assert sl._sample_count == 0
        sl.write_stepwise_log("t", 0, 0, "tgt", snapshots, MockTokenizer(), {"steps": 1})
        assert sl._sample_count == 1


# ─────────────────────────────────────────────
# TrainPredictionLogger
# ─────────────────────────────────────────────

class _MockTokenizer:
    """Simple mock tokenizer: decode maps token_id → 'T<id>'."""
    # 0 represents padding/EOS in these tests — matches fixture data below.
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"T{i}" for i in ids)


class TestTrainPredictionLogger:

    @pytest.fixture
    def tpl(self, tmp_path):
        return TrainPredictionLogger(
            log_dir=str(tmp_path), log_interval=1, max_positions=50, enabled=True,
        )

    def test_should_log_at_interval(self, tpl):
        assert tpl.should_log(0) is True
        assert tpl.should_log(1) is True  # log_interval=1

    def test_should_log_disabled(self, tmp_path):
        lg = TrainPredictionLogger(log_dir=str(tmp_path), enabled=False)
        assert lg.should_log(0) is False

    def test_write_creates_file(self, tpl, tmp_path):
        tpl.write_sample_log(
            global_step=0, epoch=0, rank=0, task="test_task", p_mask=0.5,
            mask_positions=torch.tensor([3]),
            target_tokens=torch.tensor([10]),
            pred_tokens=torch.tensor([10]),
            target_probs=torch.tensor([0.9]),
            pred_probs=torch.tensor([0.9]),
            tokenizer=_MockTokenizer(),
        )
        filepath = tmp_path / "train_predictions" / "test_task" / "epoch00.txt"
        assert filepath.exists()

    def test_write_backward_compatible_without_new_params(self, tpl, tmp_path):
        """write_sample_log works without the new optional parameters."""
        tpl.write_sample_log(
            global_step=5, epoch=1, rank=0, task="compat", p_mask=0.3,
            mask_positions=torch.tensor([2, 4]),
            target_tokens=torch.tensor([10, 20]),
            pred_tokens=torch.tensor([10, 99]),
            target_probs=torch.tensor([0.8, 0.5]),
            pred_probs=torch.tensor([0.8, 0.6]),
            tokenizer=_MockTokenizer(),
        )
        filepath = tmp_path / "train_predictions" / "compat" / "epoch01.txt"
        content = filepath.read_text()
        # Per-token table should still exist
        assert "Ground Truth" in content
        assert "Predicted" in content
        # Text sections should NOT appear (no input_ids provided)
        assert "[Input (Prompt)]" not in content

    def test_write_includes_text_sections(self, tpl, tmp_path):
        """Verify Input/Output/Prediction text sections appear when full data is provided."""
        # Simulate: prompt=[1,2,3], answer=[10,11,12], pad=[0,0]
        input_ids = torch.tensor([1, 2, 3, 10, 11, 12, 0, 0])
        labels = torch.tensor([-100, -100, -100, 10, 11, 12, 0, 0])
        attention_mask = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0])
        all_answer_gt_ids = torch.tensor([10, 11, 12, 0, 0])
        all_answer_pred_ids = torch.tensor([10, 99, 12, 0, 0])

        tpl.write_sample_log(
            global_step=10, epoch=0, rank=0, task="retro", p_mask=0.4,
            mask_positions=torch.tensor([4]),
            target_tokens=torch.tensor([11]),
            pred_tokens=torch.tensor([99]),
            target_probs=torch.tensor([0.3]),
            pred_probs=torch.tensor([0.5]),
            tokenizer=_MockTokenizer(),
            input_ids=input_ids,
            labels=labels,
            all_answer_pred_ids=all_answer_pred_ids,
            all_answer_gt_ids=all_answer_gt_ids,
            attention_mask=attention_mask,
        )

        filepath = tmp_path / "train_predictions" / "retro" / "epoch00.txt"
        content = filepath.read_text()

        assert "[Input (Prompt)]" in content
        assert "[Output (Ground Truth)]" in content
        assert "[Prediction (Model Output)]" in content
        # Prompt should decode to "T1 T2 T3"
        assert "T1 T2 T3" in content
        # GT answer (trimmed to real_answer_len=3) should decode to "T10 T11 T12"
        assert "T10 T11 T12" in content
        # Pred answer should contain T99 (the wrong prediction)
        assert "T99" in content

    def test_padding_trimmed_from_text(self, tpl, tmp_path):
        """Padding EOS tokens should be excluded from Output and Prediction text."""
        # prompt=[1,2], answer=[10,11], padding=[0,0,0]
        input_ids = torch.tensor([1, 2, 10, 11, 0, 0, 0])
        labels = torch.tensor([-100, -100, 10, 11, 0, 0, 0])
        attention_mask = torch.tensor([1, 1, 1, 1, 0, 0, 0])
        # answer_mask includes both real answer + padding-labeled-as-answer
        all_answer_gt_ids = torch.tensor([10, 11, 0, 0, 0])
        all_answer_pred_ids = torch.tensor([10, 11, 0, 0, 0])

        tpl.write_sample_log(
            global_step=0, epoch=0, rank=0, task="pad_test", p_mask=0.5,
            mask_positions=torch.tensor([3]),
            target_tokens=torch.tensor([11]),
            pred_tokens=torch.tensor([11]),
            target_probs=torch.tensor([0.9]),
            pred_probs=torch.tensor([0.9]),
            tokenizer=_MockTokenizer(),
            input_ids=input_ids,
            labels=labels,
            all_answer_pred_ids=all_answer_pred_ids,
            all_answer_gt_ids=all_answer_gt_ids,
            attention_mask=attention_mask,
        )

        filepath = tmp_path / "train_predictions" / "pad_test" / "epoch00.txt"
        content = filepath.read_text()

        # Extract the GT line — should be "T10 T11" (2 real answer tokens only)
        for line in content.split("\n"):
            if line.startswith("[Output"):
                idx = content.split("\n").index(line)
                gt_line = content.split("\n")[idx + 1]
                assert gt_line == "T10 T11"
                break
