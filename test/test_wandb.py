"""Tests for WandB integration — flush_to_wandb, _get_wandb_logger."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import wandb

from src.logging.sample_logger import ValidationSampleLogger


# ─────────────────────────────────────────────
# ValidationSampleLogger.flush_to_wandb
# ─────────────────────────────────────────────

class TestFlushToWandB:
    """flush_to_wandb() 테스트 — experiment.log() 호출을 mock으로 검증."""

    @pytest.fixture
    def logger(self, tmp_path):
        return ValidationSampleLogger(log_dir=str(tmp_path), samples_per_gpu=5)

    def test_cls_samples_logged(self, logger):
        """Classification sample이 있으면 experiment.log()가 호출된다."""
        logger.collect_classification("bbbp", torch.tensor([0.3, 0.7]), "<BOOLEAN> True </BOOLEAN>")

        mock_exp = MagicMock()
        logger.flush_to_wandb(mock_exp, epoch=1, global_step=100)

        mock_exp.log.assert_called_once()
        logged = mock_exp.log.call_args[0][0]
        assert "val/classification_samples" in logged
        assert isinstance(logged["val/classification_samples"], wandb.Table)

    def test_gen_samples_logged(self, logger):
        """Generation sample이 있으면 experiment.log()가 호출된다."""
        logger.collect_generation("reaction", "pred", "gt", strategy="random")

        mock_exp = MagicMock()
        logger.flush_to_wandb(mock_exp, epoch=0, global_step=50)

        mock_exp.log.assert_called_once()
        logged = mock_exp.log.call_args[0][0]
        assert "val/generation_samples" in logged
        assert isinstance(logged["val/generation_samples"], wandb.Table)

    def test_empty_no_call(self, logger):
        """Sample이 없으면 experiment.log()가 호출되지 않는다."""
        mock_exp = MagicMock()
        logger.flush_to_wandb(mock_exp, epoch=0, global_step=0)
        mock_exp.log.assert_not_called()

    def test_does_not_clear_buffers(self, logger):
        """flush_to_wandb는 버퍼를 초기화하지 않는다 (flush가 담당)."""
        logger.collect_classification("bbbp", torch.tensor([0.3, 0.7]), "<BOOLEAN> True </BOOLEAN>")

        mock_exp = MagicMock()
        logger.flush_to_wandb(mock_exp, epoch=0, global_step=0)

        assert len(logger._cls_samples) == 1

    def test_both_cls_and_gen(self, logger):
        """Classification + Generation 모두 있으면 2번 호출된다."""
        logger.collect_classification("bbbp", torch.tensor([0.3, 0.7]), "<BOOLEAN> True </BOOLEAN>")
        logger.collect_generation("reaction", "pred", "gt", strategy="low_confidence")

        mock_exp = MagicMock()
        logger.flush_to_wandb(mock_exp, epoch=1, global_step=200)

        assert mock_exp.log.call_count == 2


# ─────────────────────────────────────────────
# MolDATrainer._get_wandb_logger
# ─────────────────────────────────────────────

class TestGetWandbLogger:

    def test_returns_none_when_no_wandb(self):
        """WandbLogger가 없으면 None을 반환한다."""
        from src.training.trainer import MolDATrainer

        module = MolDATrainer.__new__(MolDATrainer)
        mock_csv = MagicMock()
        type(mock_csv).__name__ = "CSVLogger"
        module._loggers = [mock_csv]
        with patch.object(type(module), "loggers", new_callable=lambda: property(lambda self: self._loggers)):
            assert module._get_wandb_logger() is None

    def test_returns_wandb_logger_when_present(self):
        """WandbLogger가 있으면 해당 logger를 반환한다."""
        from src.training.trainer import MolDATrainer

        module = MolDATrainer.__new__(MolDATrainer)
        mock_csv = MagicMock()
        type(mock_csv).__name__ = "CSVLogger"
        mock_wandb = MagicMock()
        type(mock_wandb).__name__ = "WandbLogger"
        module._loggers = [mock_csv, mock_wandb]
        with patch.object(type(module), "loggers", new_callable=lambda: property(lambda self: self._loggers)):
            result = module._get_wandb_logger()
            assert result is mock_wandb
