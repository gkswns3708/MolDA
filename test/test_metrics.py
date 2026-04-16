"""Tests for task-specific evaluation metrics in src/training/metrics.py."""

import pytest
import torch
import numpy as np
import math

from src.training.metrics import (
    _parse_tag, _parse_tag_with_fallback, _parse_float_tag, _parse_boolean_tag,
    _levenshtein_distance,
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


class TestParseTagWithFallback:

    def test_dual_side(self):
        text = "<SELFIES>[C][O]</SELFIES>"
        assert _parse_tag_with_fallback(text, "SELFIES") == "[C][O]"

    def test_left_side_fallback(self):
        """닫는 태그 없으면 열린 태그부터 끝까지 추출"""
        text = "<SELFIES>[C][O]"
        result = _parse_tag_with_fallback(text, "SELFIES")
        assert result == "[C][O]"

    def test_no_tag(self):
        assert _parse_tag_with_fallback("no tags", "SELFIES") is None

    def test_description_tag(self):
        text = "<DESCRIPTION>A molecule</DESCRIPTION>"
        assert _parse_tag_with_fallback(text, "DESCRIPTION") == "A molecule"


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

    def test_all_tasks_count(self):
        assert len(ALL_TASKS) == 44

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

    EXPECTED_KEYS = {"accuracy", "f1", "precision", "recall", "roc_auc", "failure_rate"}

    def test_perfect_accuracy(self):
        probs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])  # [P(F), P(T)]
        labels = ["<BOOLEAN> True </BOOLEAN>", "<BOOLEAN> False </BOOLEAN>"]
        result = classification_evaluate(probs, labels, "bace")
        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_all_wrong(self):
        probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])  # predict F, T
        labels = ["<BOOLEAN> True </BOOLEAN>", "<BOOLEAN> False </BOOLEAN>"]
        result = classification_evaluate(probs, labels, "bace")
        assert result["accuracy"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_no_valid_labels(self):
        """전체 실패 시 NaN 반환 (Old_MolDA 동일)"""
        probs = torch.tensor([[0.5, 0.5]])
        labels = ["invalid label"]
        result = classification_evaluate(probs, labels, "bace")
        assert result["failure_rate"] == 1.0
        assert math.isnan(result["accuracy"])
        assert math.isnan(result["f1"])

    def test_roc_auc_single_class(self):
        """한 class만 있을 때 roc_auc = NaN (Old_MolDA 동일)"""
        probs = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
        labels = ["<BOOLEAN> True </BOOLEAN>", "<BOOLEAN> True </BOOLEAN>"]
        result = classification_evaluate(probs, labels, "bace")
        assert math.isnan(result["roc_auc"])

    def test_failure_rate(self):
        probs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        labels = ["<BOOLEAN> True </BOOLEAN>", "invalid"]
        result = classification_evaluate(probs, labels, "bace")
        assert result["failure_rate"] == pytest.approx(0.5)

    def test_argmax_prediction(self):
        """argmax 방식으로 prediction (Old_MolDA 동일)"""
        probs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        labels = ["<BOOLEAN> True </BOOLEAN>", "<BOOLEAN> False </BOOLEAN>"]
        result = classification_evaluate(probs, labels, "bace")
        assert result["accuracy"] == 1.0


# ─────────────────────────────────────────────
# Regression Evaluate
# ─────────────────────────────────────────────

class TestRegressionEvaluate:

    def test_perfect(self):
        preds = ["<FLOAT><|3|><|.|><|1|><|4|></FLOAT>", "<FLOAT><|2|><|.|><|0|></FLOAT>"]
        labels = ["<FLOAT>3.14</FLOAT>", "<FLOAT>2.0</FLOAT>"]
        result = regression_evaluate(preds, labels, "qm9_homo")
        assert result["mae"] == pytest.approx(0.0, abs=1e-7)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-7)

    def test_known_error(self):
        preds = ["<FLOAT><|4|><|.|><|0|></FLOAT>", "<FLOAT><|3|><|.|><|0|></FLOAT>"]
        labels = ["<FLOAT>3.0</FLOAT>", "<FLOAT>1.0</FLOAT>"]
        result = regression_evaluate(preds, labels, "qm9_homo")
        assert result["mae"] == pytest.approx(1.5)
        assert result["mse"] == pytest.approx(2.5)
        assert result["rmse"] == pytest.approx(np.sqrt(2.5), rel=1e-5)

    def test_no_valid(self):
        result = regression_evaluate(["bad"], ["bad"], "qm9_homo")
        assert result["failure_rate"] == 1.0
        assert result["mae"] == float("inf")

    def test_failure_rate_partial(self):
        """<|.|> 토큰 없는 prediction은 failure (Old_MolDA 동일)"""
        preds = ["<FLOAT><|1|><|.|><|0|></FLOAT>", "<FLOAT>2</FLOAT>"]  # 두번째는 <|.|> 없음
        labels = ["<FLOAT>1.0</FLOAT>", "<FLOAT>2.0</FLOAT>"]
        result = regression_evaluate(preds, labels, "qm9_homo")
        assert result["failure_rate"] == pytest.approx(0.5)

    def test_dot_token_required(self):
        """<|.|> 토큰 없으면 failure"""
        preds = ["<FLOAT>1</FLOAT>"]  # <|.|> 없음
        labels = ["<FLOAT>1.0</FLOAT>"]
        result = regression_evaluate(preds, labels, "qm9_homo")
        assert result["failure_rate"] == 1.0


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
        if result.get("exact_match_ratio") is not None:
            assert result["exact_match_ratio"] == 1.0

    def test_maccs_fts_identical(self):
        """동일 분자 → MACCS_FTS ≈ 1.0"""
        selfies_str = "<SELFIES>[C][C][O]</SELFIES>"  # ethanol
        result = molecule_evaluate([selfies_str], [selfies_str], "forward_reaction_prediction")
        if "MACCS_FTS" in result:
            assert result["MACCS_FTS"] == pytest.approx(1.0, abs=1e-5)

    def test_maccs_fts_different(self):
        """서로 다른 분자 → MACCS_FTS < 1.0"""
        pred = "<SELFIES>[C][C][O]</SELFIES>"      # ethanol
        gt = "<SELFIES>[C][=C][C][=C][C][=C][Ring1][=Branch1]</SELFIES>"  # benzene
        result = molecule_evaluate([pred], [gt], "forward_reaction_prediction")
        if "MACCS_FTS" in result:
            assert result["MACCS_FTS"] < 1.0

    def test_rdk_morgan_fts_identical(self):
        """동일 분자 → RDK/Morgan FTS ≈ 1.0"""
        selfies_str = "<SELFIES>[C][C][O]</SELFIES>"
        result = molecule_evaluate([selfies_str], [selfies_str], "forward_reaction_prediction")
        if "RDK_FTS" in result:
            assert result["RDK_FTS"] == pytest.approx(1.0, abs=1e-5)
        if "morgan_FTS" in result:
            assert result["morgan_FTS"] == pytest.approx(1.0, abs=1e-5)

    def test_rdk_morgan_fts_different(self):
        """서로 다른 분자 → RDK/Morgan FTS < 1.0"""
        pred = "<SELFIES>[C][C][O]</SELFIES>"
        gt = "<SELFIES>[C][=C][C][=C][C][=C][Ring1][=Branch1]</SELFIES>"
        result = molecule_evaluate([pred], [gt], "forward_reaction_prediction")
        if "RDK_FTS" in result:
            assert result["RDK_FTS"] < 1.0
        if "morgan_FTS" in result:
            assert result["morgan_FTS"] < 1.0

    def test_all_fingerprint_keys_present(self):
        """모든 fingerprint metric key가 반환에 포함 (Old_MolDA 키 이름)"""
        selfies_str = "<SELFIES>[C][C][O]</SELFIES>"
        result = molecule_evaluate([selfies_str], [selfies_str], "forward_reaction_prediction")
        for key in ["MACCS_FTS", "RDK_FTS", "morgan_FTS"]:
            assert key in result, f"Missing key: {key}"

    def test_bleu_keys_without_tokenizer(self):
        """tokenizer 없이 호출 → bleu_smiles/bleu_selfies = 0"""
        selfies_str = "<SELFIES>[C][C][O]</SELFIES>"
        result = molecule_evaluate([selfies_str], [selfies_str], "forward_reaction_prediction")
        assert "bleu_smiles" in result
        assert "bleu_selfies" in result
        assert result["bleu_smiles"] == 0.0
        assert result["bleu_selfies"] == 0.0

    def test_validity_includes_failures(self):
        """validity_ratio = 1 - failure/total (Old_MolDA 방식)"""
        selfies_str = "<SELFIES>[C][C][O]</SELFIES>"
        result = molecule_evaluate([selfies_str, "no tag"], [selfies_str, "no tag"],
                                   "forward_reaction_prediction")
        # 1 success, 1 failure → validity = 0.5
        assert result["validity_ratio"] == pytest.approx(0.5)
        assert result["failure_rate"] == pytest.approx(0.5)

    def test_levenshtein_raw_distance(self):
        """Levenshtein은 raw edit distance (정수 평균, Old_MolDA 동일)"""
        # 동일 분자 → distance = 0
        selfies_str = "<SELFIES>[C][C][O]</SELFIES>"
        result = molecule_evaluate([selfies_str], [selfies_str], "forward_reaction_prediction")
        assert result["levenshtein_score"] == pytest.approx(0.0)

    def test_left_side_fallback_selfies(self):
        """닫는 태그 없어도 left-side fallback으로 파싱 (Old_MolDA 동일)"""
        pred = "<SELFIES>[C][O]"  # 닫는 태그 없음
        gt = "<SELFIES>[C][O]</SELFIES>"
        result = molecule_evaluate([pred], [gt], "forward_reaction_prediction")
        # fallback으로 파싱 성공해야 함
        assert result["failure_rate"] == pytest.approx(0.0)

    def test_space_removal(self):
        """공백이 있어도 제거 후 파싱 (Old_MolDA 동일)"""
        pred = "<SELFIES> [C] [O] </SELFIES>"
        gt = "<SELFIES>[C][O]</SELFIES>"
        result = molecule_evaluate([pred], [gt], "forward_reaction_prediction")
        assert result["failure_rate"] == pytest.approx(0.0)

    # --- SMILES mode tests ---

    def test_smiles_mode_exact_match(self):
        """SMILES 태그로 입력 시 정상 평가"""
        smiles_str = "<SMILES>CCO</SMILES>"
        result = molecule_evaluate([smiles_str], [smiles_str], "forward_reaction_prediction")
        assert result["exact_match_ratio"] == 1.0
        assert result["failure_rate"] == pytest.approx(0.0)

    def test_smiles_mode_validity(self):
        """SMILES 모드에서 invalid SMILES → failure"""
        valid = "<SMILES>CCO</SMILES>"
        invalid_pred = "<SMILES>XXXINVALID</SMILES>"
        result = molecule_evaluate([invalid_pred], [valid], "forward_reaction_prediction")
        # CanonSmiles 또는 MolToInchi에서 실패 → failure
        assert result["failure_rate"] == 1.0

    def test_smiles_mode_fingerprints(self):
        """SMILES 모드에서 fingerprint 유사도 정상 계산"""
        smiles_str = "<SMILES>CCO</SMILES>"
        result = molecule_evaluate([smiles_str], [smiles_str], "forward_reaction_prediction")
        assert result["MACCS_FTS"] == pytest.approx(1.0, abs=1e-5)
        assert result["RDK_FTS"] == pytest.approx(1.0, abs=1e-5)
        assert result["morgan_FTS"] == pytest.approx(1.0, abs=1e-5)

    def test_smiles_mode_different_molecules(self):
        """SMILES 모드에서 다른 분자 → FTS < 1.0"""
        pred = "<SMILES>CCO</SMILES>"        # ethanol
        gt = "<SMILES>c1ccccc1</SMILES>"     # benzene
        result = molecule_evaluate([pred], [gt], "forward_reaction_prediction")
        assert result["MACCS_FTS"] < 1.0
        assert result["exact_match_ratio"] == 0.0

    def test_smiles_mode_levenshtein(self):
        """SMILES 모드에서 동일 분자 → Levenshtein = 0"""
        smiles_str = "<SMILES>CCO</SMILES>"
        result = molecule_evaluate([smiles_str], [smiles_str], "forward_reaction_prediction")
        assert result["levenshtein_score"] == pytest.approx(0.0)

    def test_smiles_mode_left_side_fallback(self):
        """SMILES 태그 닫는 태그 없어도 fallback"""
        pred = "<SMILES>CCO"  # 닫는 태그 없음
        gt = "<SMILES>CCO</SMILES>"
        result = molecule_evaluate([pred], [gt], "forward_reaction_prediction")
        assert result["failure_rate"] == pytest.approx(0.0)

    def test_smiles_selfies_auto_detect(self):
        """배치 내에서 SELFIES/SMILES 혼합 시 각각 자동 감지"""
        selfies_str = "<SELFIES>[C][C][O]</SELFIES>"
        smiles_str = "<SMILES>CCO</SMILES>"
        result = molecule_evaluate(
            [selfies_str, smiles_str],
            [selfies_str, smiles_str],
            "forward_reaction_prediction"
        )
        assert result["failure_rate"] == pytest.approx(0.0)
        assert result["exact_match_ratio"] == pytest.approx(1.0)

    def test_smiles_no_selfies_bleu(self):
        """SMILES 모드에서 SELFIES BLEU = 0 (SELFIES 변환 안 함)"""
        smiles_str = "<SMILES>CCO</SMILES>"
        result = molecule_evaluate([smiles_str], [smiles_str], "forward_reaction_prediction")
        assert result["bleu_selfies"] == 0.0


# ─────────────────────────────────────────────
# Caption Evaluate
# ─────────────────────────────────────────────

class TestCaptionEvaluate:

    def test_missing_tags(self):
        result = caption_evaluate(["no tag"], ["no tag"], "chebi-20-mol2text")
        assert result["meteor"] == 0.0

    def test_identical_text(self):
        text = "<DESCRIPTION>This is a molecule.</DESCRIPTION>"
        result = caption_evaluate([text], [text], "chebi-20-mol2text")
        assert result["failure_rate"] == 0.0
        # BLEU for identical text should be high (×100 스케일)
        if result["bleu4"] > 0:
            assert result["bleu4"] > 50

    def test_meteor_identical(self):
        """동일 텍스트 → METEOR > 90 (×100 스케일, Old_MolDA 동일)"""
        text = "<DESCRIPTION>This molecule contains a hydroxyl group attached to a benzene ring.</DESCRIPTION>"
        result = caption_evaluate([text], [text], "chebi-20-mol2text")
        if result.get("meteor", 0) > 0:
            assert result["meteor"] > 90

    def test_rouge2_key_present(self):
        """rouge2가 반환 dict에 포함"""
        text = "<DESCRIPTION>This is a test molecule description.</DESCRIPTION>"
        result = caption_evaluate([text], [text], "chebi-20-mol2text")
        assert "rouge2" in result

    def test_rouge2_identical(self):
        """동일 텍스트 → rouge2 > 90 (×100 스케일, Old_MolDA 동일)"""
        text = "<DESCRIPTION>This molecule contains a hydroxyl group attached to a benzene ring.</DESCRIPTION>"
        result = caption_evaluate([text], [text], "chebi-20-mol2text")
        if result.get("rouge2", 0) > 0:
            assert result["rouge2"] > 90

    def test_scale_is_100(self):
        """BLEU/METEOR/ROUGE 스케일이 0~100 (Old_MolDA 동일)"""
        text = "<DESCRIPTION>This molecule contains a hydroxyl group attached to a benzene ring.</DESCRIPTION>"
        result = caption_evaluate([text], [text], "chebi-20-mol2text")
        # 동일 텍스트이므로 모든 metric이 높아야 함 (×100 스케일)
        for key in ["bleu2", "bleu4", "rouge1", "rougeL", "meteor"]:
            if result.get(key, 0) > 0:
                assert result[key] > 10, f"{key} should be on 0-100 scale, got {result[key]}"

    def test_all_caption_keys(self):
        """모든 caption metric key가 반환에 포함 (meteor_wordnet, meteor_llada 포함)"""
        text = "<DESCRIPTION>A test molecule.</DESCRIPTION>"
        result = caption_evaluate([text], [text], "chebi-20-mol2text")
        expected_keys = {"bleu2", "bleu4", "meteor", "meteor_wordnet", "meteor_llada",
                         "rouge1", "rouge2", "rougeL", "failure_rate"}
        assert set(result.keys()) == expected_keys

    def test_auto_detect_description_tag(self):
        """target에서 DESCRIPTION 태그 자동 감지"""
        text = "<DESCRIPTION>A molecule description.</DESCRIPTION>"
        result = caption_evaluate([text], [text], "chebi-20-mol2text")
        assert result["failure_rate"] == 0.0

    def test_auto_detect_iupac_tag(self):
        """target에서 IUPAC 태그 자동 감지"""
        text = "<IUPAC>ethanol</IUPAC>"
        result = caption_evaluate([text], [text], "smol-name_conversion-s2i")
        assert result["failure_rate"] == 0.0

    def test_auto_detect_molformula_tag(self):
        """target에서 MOLFORMULA 태그 자동 감지"""
        text = "<MOLFORMULA>C2H6O</MOLFORMULA>"
        result = caption_evaluate([text], [text], "smol-name_conversion-s2f")
        assert result["failure_rate"] == 0.0

    def test_left_side_fallback(self):
        """prediction에 닫는 태그 없어도 파싱 (Old_MolDA 동일)"""
        pred = "<DESCRIPTION>A molecule"  # 닫는 태그 없음
        gt = "<DESCRIPTION>A molecule</DESCRIPTION>"
        result = caption_evaluate([pred], [gt], "chebi-20-mol2text")
        assert result["failure_rate"] == 0.0

    def test_wordnet_only_meteor(self):
        """WordNet만 사용 시 meteor_llada 키 없음"""
        text = "<DESCRIPTION>A test molecule.</DESCRIPTION>"
        result = caption_evaluate([text], [text], "chebi-20-mol2text",
                                  meteor_tokenizers=["wordnet"])
        assert "meteor_wordnet" in result
        assert "meteor_llada" not in result


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


# ─────────────────────────────────────────────
# JSONL Validation Helpers (DDP-safe I/O)
# ─────────────────────────────────────────────

import json
import os
import tempfile


class TestValJSONLHelpers:
    """JSONL write/read/cleanup 헬퍼 단위 테스트 (GPU 불필요)."""

    def _write_jsonl_file(self, path, records):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def test_jsonl_roundtrip(self):
        """단일 rank JSONL 쓰기 → 읽기 → 원본과 일치."""
        records = [
            {"task": "bace", "probs": [0.3, 0.7], "label": "<BOOLEAN> True </BOOLEAN>"},
            {"task": "smol-property_prediction-bbbp", "probs": [0.8, 0.2], "label": "<BOOLEAN> False </BOOLEAN>"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cls.jsonl")
            self._write_jsonl_file(path, records)

            loaded = []
            with open(path, "r") as f:
                for line in f:
                    loaded.append(json.loads(line.strip()))

            assert len(loaded) == 2
            assert loaded[0]["task"] == "bace"
            assert loaded[1]["probs"] == [0.8, 0.2]

    def test_multi_rank_merge(self):
        """다수 rank JSONL 파일 병합 시뮬레이션."""
        rank0 = [{"task": "bace", "probs": [0.3, 0.7], "label": "T"}]
        rank1 = [{"task": "smol-property_prediction-hiv", "probs": [0.9, 0.1], "label": "F"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_jsonl_file(os.path.join(tmpdir, "rank0.jsonl"), rank0)
            self._write_jsonl_file(os.path.join(tmpdir, "rank1.jsonl"), rank1)

            # Simulate loading all ranks
            all_records = []
            for rank in range(2):
                path = os.path.join(tmpdir, f"rank{rank}.jsonl")
                with open(path, "r") as f:
                    for line in f:
                        all_records.append(json.loads(line.strip()))

            assert len(all_records) == 2
            tasks = {r["task"] for r in all_records}
            assert "bace" in tasks
            assert "smol-property_prediction-hiv" in tasks

    def test_generation_jsonl_roundtrip(self):
        """Generation prediction JSONL 쓰기 → 읽기."""
        records = [
            {"task": "forward_reaction_prediction", "strategy": "low_confidence_random",
             "pred_text": "<SELFIES>[C][O]</SELFIES>", "label_text": "<SELFIES>[C][O]</SELFIES>"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gen.jsonl")
            self._write_jsonl_file(path, records)

            with open(path, "r") as f:
                loaded = [json.loads(line.strip()) for line in f]

            assert loaded[0]["strategy"] == "low_confidence_random"
            assert loaded[0]["pred_text"] == "<SELFIES>[C][O]</SELFIES>"

    def test_prediction_json_save(self):
        """영구 prediction JSON 저장 형식 검증."""
        cls_data = [{"task": "bace", "probs": [0.3, 0.7], "label": "True"}]
        gen_data = [{"task": "retrosynthesis", "strategy": "random_random",
                     "pred_text": "pred", "label_text": "label"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "predictions.json")
            payload = {
                "epoch": 0,
                "global_step": 100,
                "classification": cls_data,
                "generation": gen_data,
            }
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)

            with open(path, "r") as f:
                loaded = json.load(f)

            assert loaded["epoch"] == 0
            assert len(loaded["classification"]) == 1
            assert len(loaded["generation"]) == 1
            assert loaded["generation"][0]["strategy"] == "random_random"
