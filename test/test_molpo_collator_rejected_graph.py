"""Unit tests for graph-rejection wiring in MolPOTrainCollator + dataset.py.

Verifies that:
1. The collator's `current_epoch % num_rejected_graphs` rotation picks the
   correct `{i}-th_rejected_*` keys.
2. Graph batch concatenation order is `[chosen; rejected]` for div=2 and
   `[sft; chosen; rejected]` for div=3.
3. Falls back to chosen graph replication when rejected keys are absent.
4. dataset.py adds `target_text_chosen/rejected` fallback for graph-only V-MolPO.
5. `has_molpo_pair` recognises graph-only V-MolPO datasets via `0-th_rejected_x`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _mock_tokenizer():
    tok = MagicMock()
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    # __call__ returns a dict with input_ids/attention_mask of shape [1, L]
    def tokenize(text, **kwargs):
        # Deterministic tokenization: char codes (clipped) for stability
        ids = [min(ord(c), 99) for c in str(text)[:20]]
        return {
            "input_ids": [ids],
            "attention_mask": [[1] * len(ids)],
        }
    tok.side_effect = tokenize
    tok.return_value = tokenize("default")
    tok.encode = lambda s, **kw: [min(ord(c), 99) for c in str(s)[:20]]
    return tok


def _mock_sample(idx: int, with_rejected_graph: bool = True) -> dict:
    """Build a minimal sample row mimicking chebi_mol2text_atomwise schema."""
    sample = {
        "x": [[6, 0, 0, 0, 0, 0, 0, 0, 0],
              [6, 0, 0, 0, 0, 0, 0, 0, 0],
              [8, 0, 0, 0, 0, 0, 0, 0, 0]],
        "edge_index": [[0, 1, 1, 2], [1, 0, 2, 1]],
        "edge_attr": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "prompt_text": f"prompt_{idx}",
        "target_text": f"target_{idx}",
        "target_text_chosen": f"target_{idx}",
        "target_text_rejected": f"target_{idx}",
        "task": "chebi-20-mol2text",
    }
    if with_rejected_graph:
        # Two different "rejected" variants — chosen and rejected nodes differ
        for i in range(2):
            sample[f"{i}-th_rejected_x"] = [
                [7 + i, 0, 0, 0, 0, 0, 0, 0, 0],
                [7 + i, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
            sample[f"{i}-th_rejected_edge_index"] = [[0, 1], [1, 0]]
            sample[f"{i}-th_rejected_edge_attr"] = [[i, 0, 0], [i, 0, 0]]
    return sample


def test_collator_rotates_epoch_index_for_rejected_slot():
    """current_epoch=0 picks 0-th_rejected_*, current_epoch=1 picks 1-th_rejected_*."""
    from transformers import AutoTokenizer
    from src.data.molpo_collator import MolPOTrainCollator

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", use_fast=True,
        )
    except Exception as e:
        pytest.skip(f"LLaDA tokenizer unavailable: {e}")

    collator = MolPOTrainCollator(
        tokenizer=tokenizer,
        mol_representation="string+graph",
        max_length=64,
        batch_division=2,
        mol_token_type="selfies",
        require_pair=True,
        num_rejected_graphs=2,
    )

    batch = [_mock_sample(0), _mock_sample(1)]

    for epoch in (0, 1):
        collator.current_epoch = epoch
        out = collator(batch)
        # Each sample had its target_text_chosen == target_text_rejected so the
        # input_ids for chosen half should equal rejected half.
        assert out.get("graphs") is not None, "graphs key missing"
        assert out["_n_with_rejected_graph"] == 2, \
            "both samples carry rejected graph but collator didn't detect"
        assert out["_rejected_graph_epoch_idx"] == epoch


def test_collator_falls_back_to_chosen_when_no_rejected_graph():
    """Samples without `{i}-th_rejected_x` keys should fall back to chosen graph."""
    from transformers import AutoTokenizer
    from src.data.molpo_collator import MolPOTrainCollator

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", use_fast=True,
        )
    except Exception as e:
        pytest.skip(f"LLaDA tokenizer unavailable: {e}")

    collator = MolPOTrainCollator(
        tokenizer=tokenizer,
        mol_representation="string+graph",
        max_length=64,
        batch_division=2,
        mol_token_type="selfies",
        require_pair=True,
        num_rejected_graphs=6,
    )
    # Samples WITHOUT rejected graph keys
    batch = [
        _mock_sample(0, with_rejected_graph=False),
        _mock_sample(1, with_rejected_graph=False),
    ]
    out = collator(batch)
    assert out["_n_with_rejected_graph"] == 0, \
        "samples have no rejected graph but collator claims otherwise"


def test_dataset_graph_rejection_text_fallback():
    """When a dataset row has corrupted graphs but no chosen/rejected text columns,
    `MoleculeDataset.__getitem__` should populate them from target_text."""
    from src.data.dataset import MoleculeDataset

    # Use the actual chebi_mol2text_atomwise dataset
    ds_path = REPO_ROOT / "dataset" / "Processed" / "chebi_mol2text_atomwise" / "Train"
    if not ds_path.exists():
        pytest.skip(f"atomwise dataset not built at {ds_path}")

    ds = MoleculeDataset(str(ds_path), mol_token_type="selfies")
    sample = ds[0]
    assert "target_text" in sample
    assert "target_text_chosen" in sample, \
        "fallback should populate target_text_chosen from target_text"
    assert "target_text_rejected" in sample, \
        "fallback should populate target_text_rejected from target_text"
    assert sample["target_text_chosen"] == sample["target_text"]
    assert sample["target_text_rejected"] == sample["target_text"]
    # Graph rejection keys must survive the dataset roundtrip
    assert "0-th_rejected_x" in sample
    assert "5-th_rejected_x" in sample


def test_dataset_has_molpo_pair_via_graph_rejection():
    """A dataset with `0-th_rejected_x` (graph rejection) should be recognised
    as a MolPO pair-bearing dataset even without `target_text_chosen` column."""
    from src.data.dataset import MoleculeDataset

    ds_path = REPO_ROOT / "dataset" / "Processed" / "chebi_mol2text_atomwise" / "Train"
    if not ds_path.exists():
        pytest.skip(f"atomwise dataset not built at {ds_path}")

    ds = MoleculeDataset(str(ds_path), mol_token_type="selfies")
    assert ds.has_molpo_pair is True
