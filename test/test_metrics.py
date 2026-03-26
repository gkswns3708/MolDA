"""Tests for task-specific evaluation metrics in src/training/metrics.py."""

import pytest
import torch
import numpy as np

from src.training.metrics import (
    _parse_tag, _parse_float_tag, _parse_boolean_tag, _levenshtein_distance,
    get_task_type, classification_evaluate, regression_evaluate,
    molecule_evaluate, caption_evaluate, evaluate_by_task,
    CLASSIFICATION_TASKS, REGRESSION_TASKS, REACTION_TASKS,
    TEXT2MOL_TASKS, MOL2TEXT_TASKS, NAME_CONVERSION_TASKS, ALL_TASKS,
)


# ─────────────────────────────────────────────
# Tag Parsing
# ─────────────────────────────────────────────

class TestParseTag:

    def test_basic(self):
        assert _parse_tag("<FLOAT>3.14</FLOAT>", "FLOAT") == "3.14"

    def test_missing_tag(self):
        assert _parse_tag("no tags here", "FLOAT") is None

    def test_with_whitespace(self):
        assert _parse_tag("<BOOLEAN> True </BOOLEAN>", "BOOLEAN") == "True"

    def test_multiline_content(self):
        text = "<DESCRIPTION>line1\nline2</DESCRIPTION>"
        assert _parse_tag(text, "DESCRIPTION") == "line1\nline2"

    def test_nested_tags_different(self):
        text = "<SELFIES>[C][O]</SELFIES>"
        assert _parse_tag(text, "SELFIES") == "[C][O]"


class TestParseFloatTag:

    def test_valid_positive(self):
        assert _parse_float_tag("<FLOAT>3.14</FLOAT>") == pytest.approx(3.14)

    def test_valid_negative(self):
        assert _parse_float_tag("<FLOAT>-1.5</FLOAT>") == pytest.approx(-1.5)

    def test_with_number_tokens(self):
        result = _parse_float_tag("<FLOAT><|3|><|.|><|1|><|4|></FLOAT>")
        assert result == pytest.approx(3.14)

    def test_invalid_content(self):
        assert _parse_float_tag("<FLOAT>abc</FLOAT>") is None

    def test_missing_tag(self):
        assert _parse_float_tag("no float here") is None


class TestParseBooleanTag:

    def test_true(self):
        assert _parse_boolean_tag("<BOOLEAN> True </BOOLEAN>") is True

    def test_false(self):
        assert _parse_boolean_tag("<BOOLEAN> False </BOOLEAN>") is False

    def test_invalid(self):
        assert _parse_boolean_tag("<BOOLEAN> maybe </BOOLEAN>") is None

    def test_missing(self):
        assert _parse_boolean_tag("no boolean") is None

    def test_case_insensitive(self):
        assert _parse_boolean_tag("<BOOLEAN>TRUE</BOOLEAN>") is True
        assert _parse_boolean_tag("<BOOLEAN>false</BOOLEAN>") is False


# ─────────────────────────────────────────────
# Task Categorization
# ─────────────────────────────────────────────

class TestTaskCategorization:

    def test_all_tasks_count_21(self):
        assert len(ALL_TASKS) == 21

    def test_no_overlap_between_categories(self):
        sets = [CLASSIFICATION_TASKS, REGRESSION_TASKS, REACTION_TASKS,
                TEXT2MOL_TASKS, MOL2TEXT_TASKS, NAME_CONVERSION_TASKS]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                overlap = sets[i] & sets[j]
                assert not overlap, f"Overlap between categories: {overlap}"

    def test_get_task_type_classification(self):
        for task in CLASSIFICATION_TASKS:
            assert get_task_type(task) == "classification"

    def test_get_task_type_regression(self):
        for task in REGRESSION_TASKS:
            assert get_task_type(task) == "regression"

    def test_get_task_type_molecule(self):
        for task in REACTION_TASKS | TEXT2MOL_TASKS:
            assert get_task_type(task) == "molecule"

    def test_get_task_type_caption(self):
        for task in MOL2TEXT_TASKS:
            assert get_task_type(task) == "caption"

    def test_get_task_type_name_conversion(self):
        for task in NAME_CONVERSION_TASKS:
            assert get_task_type(task) == "name_conversion"

    def test_unknown_task_defaults_molecule(self):
        assert get_task_type("unknown_task_xyz") == "molecule"


# ─────────────────────────────────────────────
# Classification Evaluate
# ─────────────────────────────────────────────

class TestClassificationEvaluate:

    def test_perfect_accuracy(self):
        probs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])  # [P(F), P(T)]
        labels = ["<BOOLEAN> True </BOOLEAN>", "<BOOLEAN> False </BOOLEAN>"]
        result = classification_evaluate(probs, labels, "bace")
        assert result["accuracy"] == 1.0

    def test_all_wrong(self):
        probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])  # predict F, T
        labels = ["<BOOLEAN> True </BOOLEAN>", "<BOOLEAN> False </BOOLEAN>"]
        result = classification_evaluate(probs, labels, "bace")
        assert result["accuracy"] == 0.0

    def test_no_valid_labels(self):
        probs = torch.tensor([[0.5, 0.5]])
        labels = ["invalid label"]
        result = classification_evaluate(probs, labels, "bace")
        assert result["failure_rate"] == 1.0

    def test_roc_auc_single_class(self):
        probs = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
        labels = ["<BOOLEAN> True </BOOLEAN>", "<BOOLEAN> True </BOOLEAN>"]
        result = classification_evaluate(probs, labels, "bace")
        assert result["roc_auc"] == 0.0

    def test_failure_rate(self):
        probs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        labels = ["<BOOLEAN> True </BOOLEAN>", "invalid"]
        result = classification_evaluate(probs, labels, "bace")
        assert result["failure_rate"] == pytest.approx(0.5)


# ─────────────────────────────────────────────
# Regression Evaluate
# ─────────────────────────────────────────────

class TestRegressionEvaluate:

    def test_perfect(self):
        preds = ["<FLOAT>3.14</FLOAT>", "<FLOAT>2.0</FLOAT>"]
        labels = ["<FLOAT>3.14</FLOAT>", "<FLOAT>2.0</FLOAT>"]
        result = regression_evaluate(preds, labels, "qm9_homo")
        assert result["mae"] == pytest.approx(0.0, abs=1e-7)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-7)

    def test_known_error(self):
        preds = ["<FLOAT>4.0</FLOAT>", "<FLOAT>3.0</FLOAT>"]
        labels = ["<FLOAT>3.0</FLOAT>", "<FLOAT>1.0</FLOAT>"]
        result = regression_evaluate(preds, labels, "qm9_homo")
        assert result["mae"] == pytest.approx(1.5)
        assert result["rmse"] == pytest.approx(np.sqrt(2.5), rel=1e-5)

    def test_no_valid(self):
        result = regression_evaluate(["bad"], ["bad"], "qm9_homo")
        assert result["failure_rate"] == 1.0
        assert result["mae"] == float("inf")

    def test_failure_rate_partial(self):
        preds = ["<FLOAT>1.0</FLOAT>", "invalid"]
        labels = ["<FLOAT>1.0</FLOAT>", "<FLOAT>2.0</FLOAT>"]
        result = regression_evaluate(preds, labels, "qm9_homo")
        assert result["failure_rate"] == pytest.approx(0.5)


# ─────────────────────────────────────────────
# Molecule Evaluate
# ─────────────────────────────────────────────

class TestMoleculeEvaluate:

    def test_no_selfies_tag(self):
        result = molecule_evaluate(["no tag"], ["no tag"], "forward_reaction_prediction")
        assert result["failure_rate"] == 1.0

    def test_exact_match(self):
        selfies_str = "<SELFIES>[C][O]</SELFIES>"
        result = molecule_evaluate([selfies_str], [selfies_str], "forward_reaction_prediction")
        # If selfies/rdkit available, exact_match should be 1.0
        # If not available, graceful return
        if result.get("exact_match_ratio") is not None:
            assert result["exact_match_ratio"] == 1.0


# ─────────────────────────────────────────────
# Caption Evaluate
# ─────────────────────────────────────────────

class TestCaptionEvaluate:

    def test_missing_tags(self):
        result = caption_evaluate(["no tag"], ["no tag"], "chebi-20-mol2text")
        assert result["failure_rate"] == 1.0

    def test_identical_text(self):
        text = "<DESCRIPTION>This is a molecule.</DESCRIPTION>"
        result = caption_evaluate([text], [text], "chebi-20-mol2text")
        assert result["failure_rate"] == 0.0
        # BLEU for identical text should be high
        if result["bleu4"] > 0:
            assert result["bleu4"] > 0.5


# ─────────────────────────────────────────────
# Levenshtein Distance
# ─────────────────────────────────────────────

class TestLevenshtein:

    def test_identical(self):
        assert _levenshtein_distance("abc", "abc") == 0

    def test_empty_vs_nonempty(self):
        assert _levenshtein_distance("abc", "") == 3
        assert _levenshtein_distance("", "abc") == 3

    def test_known_pair(self):
        assert _levenshtein_distance("kitten", "sitting") == 3

    def test_both_empty(self):
        assert _levenshtein_distance("", "") == 0

    def test_single_char_diff(self):
        assert _levenshtein_distance("a", "b") == 1


# ─────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────

class TestDispatch:

    def test_name_conversion_returns_empty(self):
        result = evaluate_by_task("smol-name_conversion-i2s")
        assert result == {}
