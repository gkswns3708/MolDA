"""Tests for ValidationSampleLogger and StepwiseLogger."""

import os

import pytest
import torch

from src.logging.sample_logger import ValidationSampleLogger
from src.logging.stepwise_logger import StepwiseLogger


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
