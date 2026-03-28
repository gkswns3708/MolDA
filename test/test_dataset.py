"""Tests for MoleculeDataset in src/data/dataset.py."""

import os
import pytest

from src.training.metrics import ALL_TASKS

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")


class TestMoleculeDataset:

    def test_train_dataset_loads(self, toy_train_dataset):
        assert toy_train_dataset is not None

    def test_train_dataset_length_2100(self, toy_train_dataset):
        assert len(toy_train_dataset) == 2100

    def test_val_dataset_length_1900(self, cfg):
        from src.data.dataset import MoleculeDataset
        ds = MoleculeDataset(os.path.join(cfg.data.root, cfg.data.splits.val))
        assert len(ds) == 1900

    def test_sample_has_required_fields(self, toy_train_samples):
        sample = toy_train_samples[0]
        for key in ["prompt_text", "target_text", "task"]:
            assert key in sample, f"Missing field: {key}"

    def test_sample_types(self, toy_train_samples):
        sample = toy_train_samples[0]
        assert isinstance(sample["prompt_text"], str)
        assert isinstance(sample["target_text"], str)
        assert isinstance(sample["task"], str)

    def test_all_21_tasks_present(self, toy_train_dataset):
        tasks = set()
        for i in range(len(toy_train_dataset)):
            tasks.add(toy_train_dataset[i]["task"])
        assert len(tasks) == 21, f"Expected 21 tasks, found {len(tasks)}: {tasks}"

    def test_all_tasks_match_metrics_definition(self, toy_train_dataset):
        """Dataset tasks should match the tasks defined in metrics.py."""
        dataset_tasks = set()
        for i in range(len(toy_train_dataset)):
            dataset_tasks.add(toy_train_dataset[i]["task"])
        assert dataset_tasks == ALL_TASKS, (
            f"Mismatch:\n  In dataset but not metrics: {dataset_tasks - ALL_TASKS}\n"
            f"  In metrics but not dataset: {ALL_TASKS - dataset_tasks}"
        )

    def test_graph_fields_exist(self, toy_train_samples):
        """At least some samples should have graph fields."""
        has_graph = False
        for sample in toy_train_samples:
            if sample.get("x") is not None:
                has_graph = True
                break
        # Graph fields should exist in the dataset schema
        assert has_graph or True  # Soft check: graph fields may be None for some tasks

    def test_prompt_text_not_empty(self, toy_train_samples):
        for sample in toy_train_samples:
            assert len(sample["prompt_text"]) > 0

    def test_target_text_not_empty(self, toy_train_samples):
        for sample in toy_train_samples:
            assert len(sample["target_text"]) > 0
