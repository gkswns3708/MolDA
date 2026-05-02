"""Tests for scripts/make_test_subset.py

Covers:
  - per-task stratification ratio correctness
  - min_per_task floor enforcement
  - seed reproducibility
  - 18-task preservation (no task dropped)
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_test_subset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("make_test_subset", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mts_module():
    return _load_module()


def _make_ds(task_counts: dict[str, int]) -> Dataset:
    tasks = []
    for t, n in task_counts.items():
        tasks.extend([t] * n)
    return Dataset.from_dict({"task": tasks, "x": list(range(len(tasks)))})


def test_ratio_ceil_and_min(mts_module):
    """각 task의 target = max(min_per_task, round(N/ratio))."""
    counts = {"A": 3297, "B": 100, "C": 11, "D": 9}
    ds = _make_ds(counts)
    sub = mts_module.stratified_subset(ds, ratio=100, min_per_task=10, seed=42)
    c = Counter(sub["task"])
    assert c["A"] == 33              # round(3297/100)
    assert c["B"] == max(10, 1)      # round(100/100)=1 → min floor 10
    assert c["C"] == 10              # min floor (round=0)
    assert c["D"] == 9               # N<min, can't exceed total


def test_all_tasks_preserved(mts_module):
    """18 task 전부 포함되는지."""
    tasks = [f"task_{i:02d}" for i in range(18)]
    counts = {t: 1000 for t in tasks}
    ds = _make_ds(counts)
    sub = mts_module.stratified_subset(ds, ratio=100, min_per_task=10, seed=42)
    assert set(sub["task"]) == set(tasks)


def test_seed_reproducible(mts_module):
    counts = {"A": 1000, "B": 500}
    ds = _make_ds(counts)
    s1 = mts_module.stratified_subset(ds, ratio=100, min_per_task=10, seed=42)
    s2 = mts_module.stratified_subset(ds, ratio=100, min_per_task=10, seed=42)
    assert list(s1["x"]) == list(s2["x"])


def test_seed_changes_output(mts_module):
    counts = {"A": 1000}
    ds = _make_ds(counts)
    s1 = mts_module.stratified_subset(ds, ratio=100, min_per_task=10, seed=42)
    s2 = mts_module.stratified_subset(ds, ratio=100, min_per_task=10, seed=7)
    # ratio=100 → target=10, with N=1000 → very likely different indices
    assert list(s1["x"]) != list(s2["x"])


def test_no_duplicates_within_task(mts_module):
    counts = {"A": 500, "B": 200}
    ds = _make_ds(counts)
    sub = mts_module.stratified_subset(ds, ratio=100, min_per_task=10, seed=42)
    idxs = list(sub["x"])
    assert len(idxs) == len(set(idxs))
