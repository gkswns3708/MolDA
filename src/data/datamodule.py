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

    def _build_train_collator(self):
        """Train collator selection: MolPOTrainCollator if molpo enabled, else TrainCollator."""
        molpo_cfg = self.cfg.get("molpo", None)
        molpo_enabled = bool(molpo_cfg and molpo_cfg.get("enabled", False))

        if molpo_enabled:
            # Lazy import: molpo_collator currently has an unresolved symbol
            # (_build_pyg_batch) used only on the molpo path. Importing at module
            # scope breaks Stage 1/2 startup. Import here so the failure surfaces
            # only when molpo is actually enabled (Stage 3 V-MolPO).
            from src.data.molpo_collator import MolPOTrainCollator
            # Sanity: dataset must have chosen/rejected pair columns
            if not getattr(self.train_dataset, "has_molpo_pair", False):
                logger.warning(
                    "molpo.enabled=true but train_dataset has no chosen/rejected columns. "
                    "MolPOTrainCollator will use require_pair=False (skip non-pair samples). "
                    "Make sure your dataset has target_text_chosen/_rejected columns."
                )
            return MolPOTrainCollator(
                tokenizer=self.train_tokenizer,
                mol_representation=self.cfg.model.mol_representation,
                max_length=self.cfg.data.max_length,
                batch_division=int(molpo_cfg.get("batch_division", 2)),
                mol_token_type=self.cfg.tokenizer.mol_token_type,
                require_pair=bool(molpo_cfg.get("require_pair", True)),
                num_rejected_graphs=int(molpo_cfg.get("num_rejected_graphs", 6)),
            )
        return TrainCollator(
            tokenizer=self.train_tokenizer,
            mol_representation=self.cfg.model.mol_representation,
            max_length=self.cfg.data.max_length,
        )

    def train_dataloader(self):
        collator = self._build_train_collator()
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )

    def _build_val_molpo_collator(self):
        """Val/test-side MolPOTrainCollator with rejected variant fixed to 0-th.

        Same class as train but two safety knobs differ:
        - `num_rejected_graphs=1` so the rotation (i = current_epoch %
          num_rejected_graphs) always picks the 0-th variant, giving GDR a
          stable definition across epochs.
        - `require_pair=False` so multi-task datasets where some tasks lack
          rejected variants don't crash val/test — pair-less samples are
          silently skipped and excluded from the GDR aggregate (which is
          task-bucketed downstream, so partial coverage is fine).
        """
        from src.data.molpo_collator import MolPOTrainCollator
        coll = MolPOTrainCollator(
            tokenizer=self.train_tokenizer,
            mol_representation=self.cfg.model.mol_representation,
            max_length=self.cfg.data.max_length,
            batch_division=2,
            mol_token_type=self.cfg.tokenizer.mol_token_type,
            require_pair=False,
            num_rejected_graphs=1,
        )
        coll.current_epoch = 0
        return coll

    def _should_add_molpo_eval_loader(self, dataset) -> bool:
        """Return True iff a MolPO eval loader should be appended for this dataset."""
        molpo_cfg = self.cfg.get("molpo", None)
        if not (molpo_cfg and molpo_cfg.get("enabled", False)):
            return False
        if not molpo_cfg.get("eval_gdr", True):
            return False
        return bool(getattr(dataset, "has_molpo_pair", False))

    def _build_molpo_eval_loader(self, dataset):
        molpo_cfg = self.cfg.molpo
        cfg_bs = molpo_cfg.get("eval_batch_size", None)
        eval_bs = int(cfg_bs) if cfg_bs else int(self.cfg.training.batch_size)
        return DataLoader(
            dataset,
            batch_size=eval_bs,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=True,
            collate_fn=self._build_val_molpo_collator(),
        )

    def val_dataloader(self):
        collator = EvalCollator(
            tokenizer=self.eval_tokenizer,
            mol_representation=self.cfg.model.mol_representation,
            max_length=self.cfg.data.max_length,
        )
        # drop_last=False: DDP에서도 Lightning DistributedSampler가 padding으로
        # rank별 batch 수를 맞춰주므로 hang 없음. partial batch까지 포함해
        # 전체 val sample을 metric 계산에 사용 (중복 padding 2~)는 Async
        # aggregation에서 dedup하지 않지만 영향은 미미).
        gen_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.validation.get("inference_batch_size",
                                                self.cfg.training.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
        # When V-MolPO is enabled and the val set carries pair info, return a
        # second dataloader so `validation_step(..., dataloader_idx=1)` can
        # compute dataset-level GDR. Always return a list for dataloader_idx
        # signature stability.
        if self._should_add_molpo_eval_loader(self.val_dataset):
            return [gen_loader, self._build_molpo_eval_loader(self.val_dataset)]
        return [gen_loader]

    def test_dataloader(self):
        collator = EvalCollator(
            tokenizer=self.eval_tokenizer,
            mol_representation=self.cfg.model.mol_representation,
            max_length=self.cfg.data.max_length,
        )
        gen_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.validation.get("inference_batch_size",
                                                self.cfg.training.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
        if self._should_add_molpo_eval_loader(self.test_dataset):
            return [gen_loader, self._build_molpo_eval_loader(self.test_dataset)]
        return [gen_loader]
