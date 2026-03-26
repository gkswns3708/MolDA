"""Tests for MolDADataModule in src/data/datamodule.py."""

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.datamodule import MolDADataModule
from src.data.collator import EOS_TOKEN_ID, PAD_TOKEN_ID


@pytest.mark.integration
class TestDataModuleSetup:

    def test_setup_fit(self, real_tokenizer, cfg):
        dm = MolDADataModule(tokenizer=real_tokenizer, cfg=cfg)
        dm.setup("fit")
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_setup_test(self, real_tokenizer, cfg):
        dm = MolDADataModule(tokenizer=real_tokenizer, cfg=cfg)
        dm.setup("test")
        assert dm.test_dataset is not None


@pytest.mark.integration
class TestDataLoaders:

    @pytest.fixture
    def dm(self, real_tokenizer, cfg):
        dm = MolDADataModule(tokenizer=real_tokenizer, cfg=cfg)
        dm.setup("fit")
        return dm

    def test_train_dataloader_returns_dataloader(self, dm):
        dl = dm.train_dataloader()
        assert isinstance(dl, DataLoader)

    def test_train_dataloader_drop_last(self, dm):
        dl = dm.train_dataloader()
        assert dl.drop_last is True

    def test_val_dataloader_drop_last(self, dm):
        dl = dm.val_dataloader()
        assert dl.drop_last is True

    def test_train_batch_shapes(self, dm, cfg):
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        B = cfg.training.batch_size
        L = cfg.data.max_length
        assert batch["input_ids"].shape == (B, L)
        assert batch["labels"].shape == (B, L)
        assert batch["attention_mask"].shape == (B, L)

    def test_val_batch_shapes(self, dm):
        dl = dm.val_dataloader()
        batch = next(iter(dl))
        assert "prompt_input_ids" in batch
        assert "prompt_attention_mask" in batch
        assert "tasks" in batch
        assert "target_texts" in batch
        assert batch["prompt_input_ids"].ndim == 2

    def test_train_batch_padding_token(self, dm):
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        ids = batch["input_ids"]
        attn = batch["attention_mask"]
        # Padding positions should be EOS
        pad_mask = (attn == 0)
        if pad_mask.any():
            assert (ids[pad_mask] == EOS_TOKEN_ID).all()

    def test_val_batch_padding_token(self, dm):
        dl = dm.val_dataloader()
        batch = next(iter(dl))
        ids = batch["prompt_input_ids"]
        attn = batch["prompt_attention_mask"]
        pad_mask = (attn == 0)
        if pad_mask.any():
            assert (ids[pad_mask] == PAD_TOKEN_ID).all()
