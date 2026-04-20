"""MoleculeDataset이 mol_token_type에 따라 올바른 dual-column을 선택하는지 검증.

학습/추론 시 `cfg.tokenizer.mol_token_type`이 "smiles" 또는 "selfies"로 지정되면,
MoleculeDataset.__getitem__이 해당 표현의 컬럼을 기본 키(prompt_text, target_text,
input_mol_string)로 리맵해 collator에 전달해야 한다.
"""

import os

import pytest
from datasets import load_from_disk

from src.data.dataset import MoleculeDataset

pytestmark = pytest.mark.dataset


def _train_path(cfg) -> str:
    return os.path.join(cfg.data.root, cfg.data.splits.train)


@pytest.mark.parametrize("mol_token_type", ["smiles", "selfies"])
def test_mol_token_type_selects_correct_label(cfg, mol_token_type):
    path = _train_path(cfg)
    ds = MoleculeDataset(path, mol_token_type=mol_token_type)
    raw = load_from_disk(path)

    assert ds._has_dual_columns, (
        f"dual columns not detected in {path}; "
        f"columns: {sorted(raw.column_names)}"
    )

    indices = [0, 100, 500, 1500, len(ds) - 1]
    for idx in indices:
        if idx >= len(ds):
            continue
        item = ds[idx]
        expected_prompt = raw[idx][f"prompt_text_{mol_token_type}"]
        expected_target = raw[idx][f"target_text_{mol_token_type}"]
        expected_ims = raw[idx][f"input_mol_string_{mol_token_type}"]
        assert item["prompt_text"] == expected_prompt, f"idx={idx}"
        assert item["target_text"] == expected_target, f"idx={idx}"
        assert item["input_mol_string"] == expected_ims, f"idx={idx}"


@pytest.mark.parametrize("mol_token_type,expected_tag", [
    ("smiles", "<SMILES>"),
    ("selfies", "<SELFIES>"),
])
def test_tag_in_remapped_target(cfg, mol_token_type, expected_tag):
    """Reaction/generation task 샘플의 target_text에 해당 표현 태그가 포함돼야 함."""
    ds = MoleculeDataset(_train_path(cfg), mol_token_type=mol_token_type)
    # reaction-계열 task row를 찾아 검증
    found = False
    for idx in range(min(500, len(ds))):
        item = ds[idx]
        if item["task"] in {
            "forward_reaction_prediction",
            "retrosynthesis",
            "reagent_prediction",
            "smol-forward_synthesis",
            "smol-retrosynthesis",
            "smol-molecule_generation",
        }:
            assert expected_tag in item["target_text"], (
                f"task={item['task']} idx={idx}: expected {expected_tag} in target_text"
            )
            found = True
            break
    assert found, "no reaction/generation sample found in first 500 rows"


def test_mol_token_type_case_insensitive(cfg):
    """대문자 "SMILES" 입력도 정상 작동해야 함 (dataset.py:21 .lower())."""
    ds_upper = MoleculeDataset(_train_path(cfg), mol_token_type="SMILES")
    ds_lower = MoleculeDataset(_train_path(cfg), mol_token_type="smiles")
    assert ds_upper._has_dual_columns
    assert ds_lower._has_dual_columns
    assert ds_upper[0]["target_text"] == ds_lower[0]["target_text"]


def test_default_mol_token_type_is_selfies(cfg):
    """MoleculeDataset 기본값이 selfies임을 확인 (dataset.py:19)."""
    ds_default = MoleculeDataset(_train_path(cfg))
    ds_selfies = MoleculeDataset(_train_path(cfg), mol_token_type="selfies")
    assert ds_default[0]["target_text"] == ds_selfies[0]["target_text"]
