"""Tests for validation/test GDR (rewards_accuracies) dataset-level aggregation.

Covers:
- MolPOTrainCollator emits `val_indices` aligned with kept chosen samples
- `_validation_step_molpo` dispatch path (mock model)
- `_process_validation_async` aggregates `<phase>/gdr/<task>` + `<phase>/gdr_mean`
  from per-sample JSONL records (with DDP padding dedup)
- val_dataloader returns 2 dataloaders when MolPO+eval_gdr enabled, 1 otherwise

These tests deliberately mock the heavy ML pieces (LLM, GNN, compute_elbo)
so they run on CPU only and stay fast.
"""

from __future__ import annotations

import json
import os
import types
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf


# ─────────────────────────────────────────
# 1. MolPOTrainCollator val_indices emission
# ─────────────────────────────────────────


class _MinimalTokenizer:
    """Tokenizer stub: encodes by char codes, EOS=0. Enough for the
    collator's _tokenize_one / _pad_one paths."""

    eos_token_id = 0
    pad_token_id = 0
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in (text or "")][:64] or [1]


def _make_sample(idx, chosen="A description.", rejected="B description."):
    return {
        "_val_idx": idx,
        "prompt_text": "Describe:",
        "target_text_chosen": chosen,
        "target_text_rejected": rejected,
        "task": "chebi-20-mol2text",
    }


def test_molpo_collator_emits_val_indices_aligned_with_kept_chosen():
    """val_indices length must equal B (chosen-side pair count) after filtering."""
    from src.data.molpo_collator import MolPOTrainCollator

    coll = MolPOTrainCollator(
        tokenizer=_MinimalTokenizer(),
        mol_representation="string_only",      # skip graph build path
        max_length=64,
        batch_division=2,
        mol_token_type="selfies",
        require_pair=True,
        num_rejected_graphs=1,
    )
    samples = [_make_sample(i) for i in (10, 11, 12)]
    out = coll(samples)

    assert "val_indices" in out, "collator must emit val_indices"
    assert out["val_indices"] == [10, 11, 12]
    assert len(out["val_indices"]) == int(out["molpo_batch_size"])


def test_molpo_collator_val_indices_skips_missing_pair():
    """A sample missing one side of the chosen/rejected pair is dropped when
    require_pair=False; remaining val_indices reflect the surviving subset.
    """
    from src.data.molpo_collator import MolPOTrainCollator

    coll = MolPOTrainCollator(
        tokenizer=_MinimalTokenizer(),
        mol_representation="string_only",
        max_length=64,
        batch_division=2,
        mol_token_type="selfies",
        require_pair=False,                    # skip drops silently
        num_rejected_graphs=1,
    )
    bad = _make_sample(101)
    del bad["target_text_rejected"]            # missing pair side → dropped
    samples = [_make_sample(100), bad, _make_sample(102)]
    out = coll(samples)
    assert out["val_indices"] == [100, 102]


# ─────────────────────────────────────────
# 2. Async GDR aggregation
# ─────────────────────────────────────────


def _write_jsonl_records(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _async_aggregate_gdr(gdr_data, phase="val"):
    """Mirror the dataset-level aggregation block in _process_validation_async
    so the same arithmetic can be tested without spinning up Lightning."""
    val_metrics = {}
    gdr_by_task = defaultdict(lambda: {"acc": [], "r_w": [], "r_l": [], "margin": []})
    for item in gdr_data:
        t = item["task"]
        gdr_by_task[t]["acc"].append(float(item["accuracy"]))
        gdr_by_task[t]["r_w"].append(float(item["r_w"]))
        gdr_by_task[t]["r_l"].append(float(item["r_l"]))
        gdr_by_task[t]["margin"].append(float(item["margin"]))
    all_acc = []
    for task, d in gdr_by_task.items():
        n = len(d["acc"])
        val_metrics[f"{phase}/gdr/{task}"] = sum(d["acc"]) / n
        val_metrics[f"{phase}/rewards_chosen/{task}"] = sum(d["r_w"]) / n
        val_metrics[f"{phase}/rewards_rejected/{task}"] = sum(d["r_l"]) / n
        val_metrics[f"{phase}/margin/{task}"] = sum(d["margin"]) / n
        val_metrics[f"{phase}/gdr_n/{task}"] = float(n)
        all_acc.extend(d["acc"])
    if all_acc:
        val_metrics[f"{phase}/gdr_mean"] = sum(all_acc) / len(all_acc)
        val_metrics[f"{phase}/gdr_n_total"] = float(len(all_acc))
    return val_metrics


def test_async_aggregates_gdr_per_task_and_overall():
    """Per-task mean(accuracy) and global gdr_mean must match arithmetic."""
    records = [
        {"val_idx": 0, "task": "A", "r_w": 0.5, "r_l": -0.1, "accuracy": 1.0, "margin": 0.6},
        {"val_idx": 1, "task": "A", "r_w": 0.2, "r_l": 0.3,  "accuracy": 0.0, "margin": -0.1},
        {"val_idx": 2, "task": "A", "r_w": 0.4, "r_l": -0.5, "accuracy": 1.0, "margin": 0.9},
        {"val_idx": 3, "task": "B", "r_w": 0.1, "r_l": -0.2, "accuracy": 1.0, "margin": 0.3},
    ]
    m = _async_aggregate_gdr(records, phase="val")
    assert m["val/gdr/A"] == pytest.approx(2 / 3)
    assert m["val/gdr/B"] == pytest.approx(1.0)
    assert m["val/gdr_n/A"] == 3
    assert m["val/gdr_n/B"] == 1
    assert m["val/gdr_n_total"] == 4
    assert m["val/gdr_mean"] == pytest.approx(3 / 4)
    # Reward / margin averages
    assert m["val/rewards_chosen/A"] == pytest.approx((0.5 + 0.2 + 0.4) / 3)
    assert m["val/margin/A"] == pytest.approx((0.6 - 0.1 + 0.9) / 3)


def test_async_phase_prefix_switches_to_test():
    records = [
        {"val_idx": 0, "task": "T", "r_w": 1.0, "r_l": 0.0, "accuracy": 1.0, "margin": 1.0},
    ]
    m = _async_aggregate_gdr(records, phase="test")
    assert "test/gdr/T" in m
    assert "test/gdr_mean" in m
    assert "val/gdr/T" not in m


def test_load_all_val_predictions_static_dedup_gdr(tmp_path):
    """`_load_all_val_predictions_static` dedups by (val_idx, strategy) — the
    gdr tag has no strategy field, so dedup key becomes (val_idx, '') which
    correctly de-duplicates DDP-padded duplicate val_idx records across ranks.
    """
    from src.training.validation import ValidationMixin

    epoch, step = 0, 100
    # Rank 0: idx 0,1,2
    _write_jsonl_records(
        tmp_path / f"val-epoch{epoch}-step{step}-rank0-gdr.jsonl",
        [
            {"val_idx": 0, "task": "A", "r_w": 0.1, "r_l": -0.1, "accuracy": 1.0, "margin": 0.2},
            {"val_idx": 1, "task": "A", "r_w": 0.2, "r_l": 0.1,  "accuracy": 1.0, "margin": 0.1},
            {"val_idx": 2, "task": "A", "r_w": 0.0, "r_l": 0.0,  "accuracy": 0.0, "margin": 0.0},
        ],
    )
    # Rank 1: idx 0 DUPE (DDP padding), idx 3 unique
    _write_jsonl_records(
        tmp_path / f"val-epoch{epoch}-step{step}-rank1-gdr.jsonl",
        [
            {"val_idx": 0, "task": "A", "r_w": 0.1, "r_l": -0.1, "accuracy": 1.0, "margin": 0.2},
            {"val_idx": 3, "task": "A", "r_w": 0.5, "r_l": -0.5, "accuracy": 1.0, "margin": 1.0},
        ],
    )

    records = ValidationMixin._load_all_val_predictions_static(
        str(tmp_path), world_size=2, tag="gdr", epoch=epoch, step=step
    )
    # 5 raw lines − 1 dup (val_idx=0) = 4 surviving
    assert len(records) == 4
    idxs = sorted(r["val_idx"] for r in records)
    assert idxs == [0, 1, 2, 3]


# ─────────────────────────────────────────
# 3. validation_step dispatch by dataloader_idx
# ─────────────────────────────────────────


class _FakeMolpoEvalModel:
    """Replaces the heavy MolDA model: returns canned per-pair rewards so
    validation_step_molpo can be exercised without LLM init."""

    def molpo_eval_forward(self, batch):
        B = int(batch["molpo_batch_size"])
        return {
            "tasks": list(batch["tasks"][:B]) if batch.get("tasks") else ["A"] * B,
            "v_molpo/rewards_chosen": torch.full((B,), 0.5),
            "v_molpo/rewards_rejected": torch.full((B,), -0.5),
            "v_molpo/margin": torch.full((B,), 1.0),
            "v_molpo/rewards_accuracies": torch.ones(B),
            "v_molpo/elbo_theta_w": torch.full((B,), -0.1),
            "v_molpo/elbo_theta_l": torch.full((B,), -0.9),
            "v_molpo/elbo_ref_w": torch.full((B,), -0.2),
            "v_molpo/elbo_ref_l": torch.full((B,), -0.4),
        }


def _make_stub_mixin_for_validation_step(tmp_path):
    """Construct a stub object with just enough state for
    ValidationMixin._validation_step_molpo to run."""
    from src.training.validation import ValidationMixin

    obj = types.SimpleNamespace()
    obj.model = _FakeMolpoEvalModel()
    obj.global_rank = 0
    obj.current_epoch = 0
    obj.global_step = 7
    obj._val_gdr_fh = None
    obj.trainer = types.SimpleNamespace(log_dir=str(tmp_path), world_size=1)

    # Bind production methods we need
    obj._val_jsonl_path = ValidationMixin._val_jsonl_path.__get__(obj)
    obj._open_val_jsonl = ValidationMixin._open_val_jsonl.__get__(obj)
    obj._write_jsonl = ValidationMixin._write_jsonl.__get__(obj)
    obj._validation_step_molpo = ValidationMixin._validation_step_molpo.__get__(obj)
    return obj


def test_validation_step_molpo_writes_per_sample_records(tmp_path):
    obj = _make_stub_mixin_for_validation_step(tmp_path)
    batch = {
        "molpo_batch_size": 3,
        "tasks": ["A", "A", "B"] * 2,         # 2B = 6 entries; only [:B]=[:3] used
        "val_indices": [10, 11, 12],
    }
    obj._validation_step_molpo(batch, batch_idx=0)
    # Close the file
    if obj._val_gdr_fh is not None:
        obj._val_gdr_fh.close()
    # Inspect written records
    path = obj._val_jsonl_path("gdr")
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 3
    assert [r["val_idx"] for r in lines] == [10, 11, 12]
    assert [r["task"] for r in lines] == ["A", "A", "B"]
    assert all(r["accuracy"] == 1.0 for r in lines)
    assert all(r["r_w"] == 0.5 for r in lines)
    assert all(r["margin"] == 1.0 for r in lines)


def test_validation_step_dispatches_by_dataloader_idx():
    """validation_step(..., dataloader_idx=1) routes to _validation_step_molpo,
    and a regular batch with dataloader_idx=0 does NOT route to molpo."""
    from src.training.validation import ValidationMixin

    called = {"molpo": 0}

    obj = types.SimpleNamespace()
    obj.validation_step = ValidationMixin.validation_step.__get__(obj)
    # Override the molpo path on the instance so the dispatch is observable
    # without needing a full LightningModule.
    obj._validation_step_molpo = lambda batch, batch_idx: called.__setitem__(
        "molpo", called["molpo"] + 1,
    )

    # idx=1 (with molpo_batch_size) → molpo path
    obj.validation_step({"molpo_batch_size": 2, "tasks": ["A", "A"]},
                         batch_idx=0, dataloader_idx=1)
    assert called["molpo"] == 1

    # idx=0 with non-molpo batch must not invoke the molpo path. The
    # generation path will fail downstream when accessing model state, but
    # we only care that molpo was NOT invoked beforehand.
    try:
        obj.validation_step(
            {"tasks": ["A"], "prompt_input_ids": None,
             "prompt_attention_mask": None, "target_texts": ["t"]},
            batch_idx=0, dataloader_idx=0,
        )
    except (AttributeError, KeyError, TypeError):
        pass
    assert called["molpo"] == 1, "dataloader_idx=0 must NOT route to molpo path"


# ─────────────────────────────────────────
# 4. Datamodule val_dataloader list shape
# ─────────────────────────────────────────


def _make_datamodule_cfg(molpo_enabled=True, eval_gdr=True, has_pair=True):
    """Minimal cfg + datamodule stub for testing val_dataloader plumbing.

    Bypasses HuggingFace dataset load by stubbing val_dataset/test_dataset
    with a fake object that has .has_molpo_pair.
    """
    cfg = OmegaConf.create({
        "data": {"max_length": 32, "root": "x", "splits": {"train": "T", "val": "V", "test": "Te"}},
        "model": {"mol_representation": "string_only"},
        "tokenizer": {"mol_token_type": "selfies"},
        "training": {"batch_size": 2},
        "validation": {"inference_batch_size": 2},
        "hardware": {"num_workers": 0},
        "molpo": {
            "enabled": molpo_enabled,
            "eval_gdr": eval_gdr,
            "eval_batch_size": None,
            "batch_division": 2,
            "require_pair": True,
            "num_rejected_graphs": 1,
        } if molpo_enabled else None,
    })
    return cfg, has_pair


class _FakeDataset:
    def __init__(self, has_pair):
        self.has_molpo_pair = has_pair
    def __len__(self):
        return 4
    def __getitem__(self, i):
        return {"_val_idx": i,
                "prompt_text": "x",
                "target_text_chosen": "a",
                "target_text_rejected": "b",
                "task": "A"}


def _build_datamodule(cfg, has_pair):
    from src.data.datamodule import MolDADataModule
    dm = MolDADataModule(tokenizer=_MinimalTokenizer(), cfg=cfg)
    dm.val_dataset = _FakeDataset(has_pair)
    dm.test_dataset = _FakeDataset(has_pair)
    return dm


def test_val_dataloader_returns_two_when_molpo_enabled():
    cfg, has_pair = _make_datamodule_cfg(molpo_enabled=True, eval_gdr=True, has_pair=True)
    dm = _build_datamodule(cfg, has_pair)
    loaders = dm.val_dataloader()
    assert isinstance(loaders, list) and len(loaders) == 2
    test_loaders = dm.test_dataloader()
    assert isinstance(test_loaders, list) and len(test_loaders) == 2


def test_val_dataloader_returns_one_when_molpo_disabled():
    cfg, has_pair = _make_datamodule_cfg(molpo_enabled=False, has_pair=True)
    dm = _build_datamodule(cfg, has_pair)
    loaders = dm.val_dataloader()
    assert isinstance(loaders, list) and len(loaders) == 1


def test_val_dataloader_returns_one_when_dataset_lacks_pair():
    cfg, has_pair = _make_datamodule_cfg(molpo_enabled=True, eval_gdr=True, has_pair=False)
    dm = _build_datamodule(cfg, has_pair)
    loaders = dm.val_dataloader()
    assert isinstance(loaders, list) and len(loaders) == 1


def test_val_dataloader_returns_one_when_eval_gdr_off():
    cfg, has_pair = _make_datamodule_cfg(molpo_enabled=True, eval_gdr=False, has_pair=True)
    dm = _build_datamodule(cfg, has_pair)
    loaders = dm.val_dataloader()
    assert isinstance(loaders, list) and len(loaders) == 1
