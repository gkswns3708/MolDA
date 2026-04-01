"""Tests for ValidationSampleLogger, StepwiseLogger, and TrainPredictionLogger."""

import os

import pytest
import torch

from src.logging.sample_logger import ValidationSampleLogger
from src.logging.stepwise_logger import StepwiseLogger
from src.logging.train_prediction_logger import TrainPredictionLogger


# ─────────────────────────────────────────────
# ValidationSampleLogger
# ─────────────────────────────────────────────

class TestValidationSampleLogger:

    @pytest.fixture
    def logger(self, tmp_path):
        return ValidationSampleLogger(log_dir=str(tmp_path), samples_per_gpu=2)

    def test_reset_clears_buffers(self, logger):
        logger.collect_classification("task", torch.tensor([0.3, 0.7]), "<BOOLEAN> True </BOOLEAN>")
        logger.reset()
        assert len(logger._cls_samples) == 0
        assert len(logger._gen_samples) == 0

    def test_collect_cls_within_limit(self, logger):
        logger.collect_classification("bbbp", torch.tensor([0.3, 0.7]), "<BOOLEAN> True </BOOLEAN>")
        assert len(logger._cls_samples) == 1

    def test_collect_cls_exceeds_limit(self, logger):
        for i in range(5):
            logger.collect_classification(f"task_{i}", torch.tensor([0.3, 0.7]), "<BOOLEAN> True </BOOLEAN>")
        assert len(logger._cls_samples) == 2  # samples_per_gpu=2

    def test_cls_correct_with_boolean_tag(self, logger):
        """Regression: label이 <BOOLEAN> True </BOOLEAN> 형태일 때 correct 판정."""
        logger.collect_classification("bbbp", torch.tensor([0.3, 0.7]), "<BOOLEAN> True </BOOLEAN>")
        assert logger._cls_samples[0]["correct"] is True

    def test_cls_correct_false_prediction(self, logger):
        """prob_false > prob_true 이고 label이 False → correct=True."""
        logger.reset()
        logger.collect_classification("bbbp", torch.tensor([0.8, 0.2]), "<BOOLEAN> False </BOOLEAN>")
        assert logger._cls_samples[0]["correct"] is True

    def test_cls_wrong_prediction(self, logger):
        """prob_true > prob_false 이고 label이 False → correct=False."""
        logger.reset()
        logger.collect_classification("bbbp", torch.tensor([0.3, 0.7]), "<BOOLEAN> False </BOOLEAN>")
        assert logger._cls_samples[0]["correct"] is False

    def test_cls_correct_with_eot_suffix(self, logger):
        """label에 <|eot_id|> suffix가 있어도 올바르게 파싱."""
        logger.reset()
        logger.collect_classification("bbbp", torch.tensor([0.3, 0.7]), "<BOOLEAN> True </BOOLEAN><|eot_id|>")
        assert logger._cls_samples[0]["correct"] is True

    def test_collect_gen_within_limit(self, logger):
        logger.collect_generation("reaction", "pred_text", "gt_text")
        assert len(logger._gen_samples) == 1

    def test_collect_gen_exceeds_limit(self, logger):
        for i in range(5):
            logger.collect_generation(f"task_{i}", f"pred_{i}", f"gt_{i}")
        assert len(logger._gen_samples) == 2

    def test_flush_creates_file(self, logger, tmp_path):
        logger.collect_classification("bbbp", torch.tensor([0.3, 0.7]), "<BOOLEAN> True </BOOLEAN>")
        logger.flush(epoch=1, global_step=100, rank=0)
        expected = tmp_path / "val_samples" / "epoch01_step100_rank0.txt"
        assert expected.exists()

    def test_flush_file_format(self, logger, tmp_path):
        logger.collect_classification("bbbp", torch.tensor([0.3, 0.7]), "<BOOLEAN> True </BOOLEAN>")
        logger.collect_generation("reaction", "pred", "gt")
        logger.flush(epoch=0, global_step=50, rank=0)
        filepath = tmp_path / "val_samples" / "epoch00_step50_rank0.txt"
        content = filepath.read_text()
        assert "[Classification]" in content
        assert "[Generation]" in content
        assert "bbbp" in content

    def test_flush_empty_no_file(self, logger, tmp_path):
        logger.flush(epoch=0, global_step=0, rank=0)
        val_dir = tmp_path / "val_samples"
        assert not val_dir.exists() or not list(val_dir.iterdir())

    def test_flush_clears_buffers(self, logger):
        logger.collect_classification("t", torch.tensor([0.5, 0.5]), "<BOOLEAN> True </BOOLEAN>")
        logger.flush(epoch=0, global_step=0)
        assert len(logger._cls_samples) == 0
        assert len(logger._gen_samples) == 0


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
            global_step=0, epoch=0, task="test_task", p_mask=0.5,
            mask_positions=torch.tensor([3]),
            target_tokens=torch.tensor([10]),
            pred_tokens=torch.tensor([10]),
            target_probs=torch.tensor([0.9]),
            pred_probs=torch.tensor([0.9]),
            tokenizer=_MockTokenizer(),
        )
        filepath = tmp_path / "train_predictions" / "step000000_epoch00_test_task.txt"
        assert filepath.exists()

    def test_write_backward_compatible_without_new_params(self, tpl, tmp_path):
        """write_sample_log works without the new optional parameters."""
        tpl.write_sample_log(
            global_step=5, epoch=1, task="compat", p_mask=0.3,
            mask_positions=torch.tensor([2, 4]),
            target_tokens=torch.tensor([10, 20]),
            pred_tokens=torch.tensor([10, 99]),
            target_probs=torch.tensor([0.8, 0.5]),
            pred_probs=torch.tensor([0.8, 0.6]),
            tokenizer=_MockTokenizer(),
        )
        filepath = tmp_path / "train_predictions" / "step000005_epoch01_compat.txt"
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
            global_step=10, epoch=0, task="retro", p_mask=0.4,
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

        filepath = tmp_path / "train_predictions" / "step000010_epoch00_retro.txt"
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
            global_step=0, epoch=0, task="pad_test", p_mask=0.5,
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

        filepath = tmp_path / "train_predictions" / "step000000_epoch00_pad_test.txt"
        content = filepath.read_text()

        # Extract the GT line — should be "T10 T11" (2 real answer tokens only)
        for line in content.split("\n"):
            if line.startswith("[Output"):
                idx = content.split("\n").index(line)
                gt_line = content.split("\n")[idx + 1]
                assert gt_line == "T10 T11"
                break
