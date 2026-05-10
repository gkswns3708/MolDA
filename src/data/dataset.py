"""
MoleculeDataset: thin wrapper around HuggingFace Arrow dataset.

Usage:
    ds = MoleculeDataset("dataset/Processed/toy100/Train", mol_token_type="selfies")
    sample = ds[0]  # dict with keys: x, edge_index, edge_attr, ..., prompt_text, target_text

듀얼 컬럼 데이터셋(prompt_text_smiles, prompt_text_selfies 등)에서
mol_token_type에 따라 해당 컬럼을 기존 이름(prompt_text, target_text, input_mol_string)으로 리맵.
기존 단일 컬럼 데이터셋(prompt_text만 존재)도 하위 호환으로 동작.

V-MolPO chosen/rejected 컬럼 (`target_text_chosen`, `target_text_rejected`,
또는 dual-column `target_text_chosen_{selfies,smiles}`) 도 자동 감지하여
기존 키와 함께 리턴. MolPOTrainCollator 가 이를 사용.
"""

from datasets import load_from_disk
from torch.utils.data import Dataset


class MoleculeDataset(Dataset):

    def __init__(self, path: str, mol_token_type: str = "selfies"):
        self.dataset = load_from_disk(path)
        self.mol_token_type = mol_token_type.lower()

        cols = self.dataset.column_names
        # 듀얼 컬럼 존재 여부 확인 (최초 1회)
        self._has_dual_columns = f"prompt_text_{self.mol_token_type}" in cols

        # V-MolPO chosen/rejected 컬럼 감지 (단일 또는 dual-column 형태 둘 다 지원)
        self._has_molpo_pair = (
            "target_text_chosen" in cols or
            f"target_text_chosen_{self.mol_token_type}" in cols
        )
        self._has_molpo_pair_dual = (
            f"target_text_chosen_{self.mol_token_type}" in cols and
            f"target_text_rejected_{self.mol_token_type}" in cols
        )

    @property
    def has_molpo_pair(self) -> bool:
        return self._has_molpo_pair

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        if self._has_dual_columns:
            suffix = self.mol_token_type
            # 듀얼 컬럼 → 기존 키 이름으로 리맵
            # dict()로 복사하여 원본 Arrow row 변경 방지
            item = dict(item)
            item["prompt_text"] = item[f"prompt_text_{suffix}"]
            item["target_text"] = item[f"target_text_{suffix}"]
            item["input_mol_string"] = item[f"input_mol_string_{suffix}"]
        else:
            item = dict(item)

        # V-MolPO chosen/rejected: dual-column 우선, 없으면 단일 컬럼 사용
        if self._has_molpo_pair_dual:
            suffix = self.mol_token_type
            item["target_text_chosen"] = item[f"target_text_chosen_{suffix}"]
            item["target_text_rejected"] = item[f"target_text_rejected_{suffix}"]
        # 단일 컬럼 형태 (target_text_chosen / target_text_rejected) 는 이미 dict 에 그대로 있음

        # DDP DistributedSampler padding duplicate 식별용 원본 dataset idx.
        # validation에서 aggregation 시 (val_idx, strategy) key로 dedup.
        item["_val_idx"] = int(idx)
        return item
