"""Tests for WandB integration — _get_wandb_logger."""

from unittest.mock import MagicMock, patch

import pytest


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
