"""
MolDA Lightning DataModule.

Manages train/val/test datasets with appropriate collators.
- Train: TrainCollator (right-pad), shuffle=True, drop_last=True
- Val/Test: EvalCollator (left-pad), shuffle=False, drop_last=True
"""

import logging
import os
from copy import deepcopy

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.collator import TrainCollator, EvalCollator
from src.data.dataset import MoleculeDataset

logger = logging.getLogger(__name__)


class MolDADataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        # Separate tokenizers for train (right-pad) and eval (left-pad)
        self.train_tokenizer = deepcopy(tokenizer)
        self.train_tokenizer.padding_side = "right"

        self.eval_tokenizer = deepcopy(tokenizer)
        self.eval_tokenizer.padding_side = "left"

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _resolve_path(self, split_name: str) -> str:
        """Resolve data path: dataset/Processed/{root}/{split}.

        듀얼 컬럼 데이터셋은 mol_type 없이 단일 경로에 저장.
        하위 호환: 기존 경로(dataset/Processed/{MOL_TYPE}/...)가 존재하면 그쪽 사용.
        """
        data_cfg = self.cfg.data
        # 새 경로: dataset/Processed/{root}/{split}
        new_path = os.path.join(
            "dataset", "Processed",
            data_cfg.root, data_cfg.splits[split_name],
        )
        if os.path.exists(new_path):
            return new_path

        # 하위 호환: 기존 경로 dataset/Processed/{MOL_TYPE}/{root}/{split}
        mol_type = self.cfg.tokenizer.mol_token_type.upper()
        legacy_path = os.path.join(
            "dataset", "Processed", mol_type,
            data_cfg.root, data_cfg.splits[split_name],
        )
        if os.path.exists(legacy_path):
            logger.warning(
                f"Using legacy path: {legacy_path}. "
                f"Consider regenerating dataset to new format: {new_path}"
            )
            return legacy_path

        # 둘 다 없으면 새 경로 반환 (load_from_disk에서 에러 발생)
        return new_path

    def setup(self, stage=None):
        mol_token_type = self.cfg.tokenizer.mol_token_type

        if stage in ("fit", None):
            self.train_dataset = MoleculeDataset(
                self._resolve_path("train"), mol_token_type=mol_token_type,
            )
            self.val_dataset = MoleculeDataset(
                self._resolve_path("val"), mol_token_type=mol_token_type,
            )
            logger.info(f"Train: {len(self.train_dataset)} samples")
            logger.info(f"Val: {len(self.val_dataset)} samples")

        if stage in ("test", None):
            self.test_dataset = MoleculeDataset(
                self._resolve_path("test"), mol_token_type=mol_token_type,
            )
            logger.info(f"Test: {len(self.test_dataset)} samples")

    def train_dataloader(self):
        collator = TrainCollator(
            tokenizer=self.train_tokenizer,
            mol_representation=self.cfg.model.mol_representation,
            max_length=self.cfg.data.max_length,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )

    def val_dataloader(self):
        collator = EvalCollator(
            tokenizer=self.eval_tokenizer,
            mol_representation=self.cfg.model.mol_representation,
            max_length=self.cfg.data.max_length,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.validation.get("inference_batch_size",
                                                self.cfg.training.batch_size),
            shuffle=False,
            drop_last=True,
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )

    def test_dataloader(self):
        collator = EvalCollator(
            tokenizer=self.eval_tokenizer,
            mol_representation=self.cfg.model.mol_representation,
            max_length=self.cfg.data.max_length,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.validation.get("inference_batch_size",
                                                self.cfg.training.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
