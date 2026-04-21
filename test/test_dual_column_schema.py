"""Dual-column Arrow dataset 계약 검증.

Step 3 산출물(`dataset/Processed/toy100/{Train,Val,Test}`)이 다음을 만족하는지 테스트:
- 6개 dual column 존재 (prompt/target/input_mol_string × smiles/selfies)
- 태그가 표현 타입에 맞게 배치 (`<SMILES>`는 *_smiles에만, `<SELFIES>`는 *_selfies에만)
- SMILES/SELFIES 내용이 각각 유효하고 round-trip 가능
- 21개 task 존재, 행 수 예상 범위

MoleculeDataset 리맵을 거치지 않고 Arrow를 직접 로드하여 raw 컬럼 검증.
"""

import os
import random

import pytest
from datasets import load_from_disk
from rdkit import Chem, RDLogger

from src.dataset_generation.utils import smiles_to_selfies, selfies_to_smiles, get_canonical_smiles
from src.training.metrics import ALL_TASKS

from validate_generated_dataset import (
    SELFIES_EXCLUSIVE_RE,
    extract_mol_string_content,
    is_selfies,
    is_valid_smiles,
)

RDLogger.logger().setLevel(RDLogger.CRITICAL)

pytestmark = pytest.mark.dataset

EXPECTED_DUAL_COLUMNS = {
    "prompt_text_smiles", "prompt_text_selfies",
    "target_text_smiles", "target_text_selfies",
    "input_mol_string_smiles", "input_mol_string_selfies",
}
EXPECTED_GRAPH_COLUMNS = {
    "x", "edge_index", "edge_attr",
    "additional_x", "additional_edge_index", "additional_edge_attr",
}


def _split_path(cfg, split_key: str) -> str:
    return os.path.join(cfg.data.root, cfg.data.splits[split_key])


@pytest.fixture(scope="module")
def raw_train_arrow(cfg):
    return load_from_disk(_split_path(cfg, "train"))


@pytest.fixture(scope="module")
def raw_val_arrow(cfg):
    return load_from_disk(_split_path(cfg, "val"))


@pytest.fixture(scope="module")
def raw_test_arrow(cfg):
    return load_from_disk(_split_path(cfg, "test"))


@pytest.fixture(scope="module", params=["train", "val", "test"])
def raw_split(request, raw_train_arrow, raw_val_arrow, raw_test_arrow):
    return {
        "train": raw_train_arrow,
        "val": raw_val_arrow,
        "test": raw_test_arrow,
    }[request.param], request.param


# ── schema ───────────────────────────────────────────────────────────

def test_dual_columns_present(raw_split):
    ds, name = raw_split
    missing = EXPECTED_DUAL_COLUMNS - set(ds.column_names)
    assert not missing, f"[{name}] missing dual columns: {missing}"
    missing_graph = EXPECTED_GRAPH_COLUMNS - set(ds.column_names)
    assert not missing_graph, f"[{name}] missing graph columns: {missing_graph}"
    assert "task" in ds.column_names


def test_legacy_mol_representation_column_removed(raw_split):
    ds, name = raw_split
    assert "mol_representation" not in ds.column_names, (
        f"[{name}] legacy single-column 'mol_representation' still present; "
        f"dual-column schema should replace it"
    )


def test_every_row_has_dual_content(raw_split):
    ds, name = raw_split
    for col in EXPECTED_DUAL_COLUMNS:
        values = ds[col]
        assert all(v is not None and len(v) > 0 for v in values), (
            f"[{name}] column '{col}' contains None or empty strings"
        )


# ── tag placement ────────────────────────────────────────────────────

def test_smiles_tag_only_in_smiles_columns(raw_split):
    ds, name = raw_split
    n = min(200, len(ds))
    for i in range(n):
        row = ds[i]
        for col in ("prompt_text_smiles", "target_text_smiles", "input_mol_string_smiles"):
            v = row[col]
            assert "<SELFIES>" not in v, f"[{name}/{i}/{col}] <SELFIES> leaked into smiles column"
        for col in ("prompt_text_selfies", "target_text_selfies", "input_mol_string_selfies"):
            v = row[col]
            assert "<SMILES>" not in v, f"[{name}/{i}/{col}] <SMILES> leaked into selfies column"


# ── content validity ────────────────────────────────────────────────

def test_smiles_content_is_valid_smiles(raw_split):
    ds, name = raw_split
    rng = random.Random(0)
    indices = rng.sample(range(len(ds)), min(50, len(ds)))
    failures = []
    for i in indices:
        ims = ds[i]["input_mol_string_smiles"]
        tag_type, content = extract_mol_string_content(ims)
        if tag_type != "SMILES" or content is None or content == "<None>":
            continue
        # reaction: validate each side & each molecule
        ok = True
        for side in content.split(">>"):
            for m in side.split("."):
                m = m.strip()
                if not m:
                    continue
                if not is_valid_smiles(m):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            failures.append((i, content[:60]))
    assert not failures, f"[{name}] invalid SMILES content in {len(failures)} rows: {failures[:3]}"


def test_selfies_content_is_valid_selfies(raw_split):
    ds, name = raw_split
    rng = random.Random(1)
    indices = rng.sample(range(len(ds)), min(50, len(ds)))
    failures = []
    for i in indices:
        ims = ds[i]["input_mol_string_selfies"]
        tag_type, content = extract_mol_string_content(ims)
        if tag_type != "SELFIES" or content is None or content == "<None>":
            continue
        # reaction-aware: each side's each molecule should look selfies-like
        ok = True
        for side in content.split(">>"):
            for m in side.split("."):
                m = m.strip()
                if not m:
                    continue
                if not is_selfies(m):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            failures.append((i, content[:60]))
    assert not failures, f"[{name}] non-SELFIES content in {len(failures)} rows: {failures[:3]}"


# ── round-trip ──────────────────────────────────────────────────────

def _extract_simple_mol(ims: str, expected_tag: str):
    """Return content inside the tag iff it's a single molecule (not reaction, not <None>)."""
    tag_type, content = extract_mol_string_content(ims)
    if tag_type != expected_tag or not content or content == "<None>":
        return None
    if ">>" in content:
        return None
    return content


def test_smiles_selfies_round_trip(raw_split):
    """smiles_to_selfies(smiles_col) should match selfies_col for simple molecules."""
    ds, name = raw_split
    rng = random.Random(2)
    indices = rng.sample(range(len(ds)), min(30, len(ds)))
    mismatches = 0
    compared = 0
    for i in indices:
        row = ds[i]
        smi = _extract_simple_mol(row["input_mol_string_smiles"], "SMILES")
        sfs = _extract_simple_mol(row["input_mol_string_selfies"], "SELFIES")
        if smi is None or sfs is None:
            continue
        compared += 1
        derived_sfs = smiles_to_selfies(smi)
        if derived_sfs is None or derived_sfs.replace(" ", "") != sfs.replace(" ", ""):
            mismatches += 1
    assert compared >= 5, f"[{name}] too few simple-mol samples to compare: {compared}"
    assert mismatches <= 2, (
        f"[{name}] smiles→selfies round-trip mismatch: {mismatches}/{compared}"
    )


def test_selfies_round_trip_decode_smiles(raw_split):
    """selfies_to_smiles(selfies_col) should match canonical(smiles_col)."""
    ds, name = raw_split
    rng = random.Random(3)
    indices = rng.sample(range(len(ds)), min(30, len(ds)))
    mismatches = 0
    compared = 0
    for i in indices:
        row = ds[i]
        smi = _extract_simple_mol(row["input_mol_string_smiles"], "SMILES")
        sfs = _extract_simple_mol(row["input_mol_string_selfies"], "SELFIES")
        if smi is None or sfs is None:
            continue
        compared += 1
        canonical = get_canonical_smiles(smi)
        decoded = selfies_to_smiles(sfs)
        if canonical is None or decoded is None or canonical != decoded:
            mismatches += 1
    assert compared >= 5, f"[{name}] too few simple-mol samples to compare: {compared}"
    assert mismatches <= 2, (
        f"[{name}] selfies→smiles round-trip mismatch: {mismatches}/{compared}"
    )


# ── task coverage & row counts ───────────────────────────────────────

def test_tasks_present(raw_split):
    ds, name = raw_split
    tasks = set(ds["task"])
    undefined = tasks - ALL_TASKS
    assert not undefined, f"[{name}] tasks not in metrics.ALL_TASKS: {undefined}"
    if name == "train":
        assert len(tasks) == 21, f"[train] expected 21 tasks, got {len(tasks)}: {tasks}"
    else:
        # smol-name_conversion-i2s / s2i have no val/test per run.py
        assert len(tasks) >= 19, f"[{name}] expected ≥19 tasks, got {len(tasks)}: {tasks}"


EXPECTED_ROWS = {"train": 2100, "val": 1900, "test": 1900}


def test_row_counts(raw_split):
    ds, name = raw_split
    expected = EXPECTED_ROWS[name]
    min_expected = int(expected * 0.9)
    assert len(ds) >= min_expected, (
        f"[{name}] row count too low: {len(ds)} < {min_expected} (expected ~{expected})"
    )
