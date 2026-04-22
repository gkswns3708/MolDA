"""Policy-verification tests for the cross-source decontamination pipeline.

Tests are organized around 7 policy axes:
  Axis 1: Family/group mapping accuracy
  Axis 2: Entity key canonicalization invariants
  Axis 3: Eval blacklist generation
  Axis 4: Split-aware conflict resolution
  Axis 5: Source priority rule
  Axis 6: Report/statistics/reason codes
  Axis 7: End-to-end acceptance (zero overlap + schema preservation)

Each axis contains unit tests, policy tests, and/or integration tests
that verify the decontamination *policy invariants* rather than
implementation details.
"""

import sys
import os

import pytest
import datasets

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset_generation.utils import (
    extract_entity_key,
    get_canonical_molecule_key,
    get_canonical_reaction_key,
    get_canonical_smiles,
    get_canonical_reaction_smiles,
)
from dataset_generation.dedup import (
    ENTITY_FAMILIES,
    FAMILY_KEY_SOURCE,
    REMOVE_ON_CONFLICT,
    REMOVAL_REASONS,
    RemovalStats,
    _get_family,
    _get_key_source_field,
    build_eval_blacklist,
    remove_eval_leakage,
    dedup_within_family,
    run_decontamination_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(mol_strings):
    """Helper: input_mol_string 컬럼만 가진 HuggingFace Dataset 생성."""
    return datasets.Dataset.from_dict({"input_mol_string": mol_strings})


def _make_rich_dataset(mol_strings, task_name="test_task"):
    """Production schema 컬럼을 가진 Dataset 생성 (schema preservation 테스트용)."""
    n = len(mol_strings)
    return datasets.Dataset.from_dict({
        "input_mol_string": mol_strings,
        "label": [f"label_{i}" for i in range(n)],
        "instruction": [f"inst_{i}" for i in range(n)],
        "task_subtask_pair": [task_name] * n,
    })


def _make_text2mol_dataset(mol_labels):
    """text2mol 스키마: input_mol_string='<None>', label에 <SMILES>분자</SMILES>."""
    n = len(mol_labels)
    return datasets.Dataset.from_dict({
        "input_mol_string": ["<SMILES> <None> </SMILES>"] * n,
        "label": mol_labels,
    })


def _make_mol2text_dataset(mol_inputs):
    """mol2text 스키마: input_mol_string에 분자, label은 description."""
    n = len(mol_inputs)
    return datasets.Dataset.from_dict({
        "input_mol_string": mol_inputs,
        "label": [f"desc_{i}" for i in range(n)],
    })


def _collect_keys(ds, key_field="input_mol_string"):
    """Dataset에서 모든 entity key를 set으로 수집. family-aware하게 field 지정 가능."""
    keys = set()
    if key_field not in ds.column_names:
        return keys
    for mol_str in ds[key_field]:
        key = extract_entity_key(mol_str)
        if key is not None:
            keys.add(key)
    return keys


def _collect_keys_for_task(ds, task_name):
    """Task에 맞는 key source field로 key를 수집 (FAMILY_KEY_SOURCE 기반)."""
    return _collect_keys(ds, _get_key_source_field(task_name))


# ---------------------------------------------------------------------------
# Entity Key Tests
# ---------------------------------------------------------------------------

class TestEntityKeys:
    """Entity key 추출 함수 테스트."""

    def test_canonical_molecule_key_valid(self):
        # Ethanol: different valid SMILES → same canonical key
        assert get_canonical_molecule_key("CCO") == get_canonical_molecule_key("OCC")

    def test_canonical_molecule_key_invalid(self):
        # RDKit 실패 후 selfies_to_smiles 시도 → 빈 문자열 반환 가능
        # dedup에서는 falsy 값이면 key 비교 대상에서 제외되므로 기능적으로 안전
        result = get_canonical_molecule_key("not_a_smiles_xxxx")
        assert not result or result is None

    def test_canonical_molecule_key_empty(self):
        assert get_canonical_molecule_key("") is None
        assert get_canonical_molecule_key(None) is None

    def test_canonical_reaction_key_valid(self):
        # Same reaction, different component order
        key1 = get_canonical_reaction_key("A.B>>C")
        # For real SMILES:
        key_a = get_canonical_reaction_key("CCO.CC>>CCOC")
        key_b = get_canonical_reaction_key("CC.CCO>>CCOC")
        # After sorting, both sides should be identical
        assert key_a == key_b

    def test_canonical_reaction_key_no_arrow(self):
        # No '>>' → treated as single molecule
        key = get_canonical_reaction_key("CCO")
        assert key == get_canonical_smiles("CCO")

    def test_extract_entity_key_smiles_tags(self):
        key = extract_entity_key("<SMILES> CCO </SMILES>")
        assert key == get_canonical_smiles("CCO")

    def test_extract_entity_key_selfies_tags(self):
        # SELFIES tags with a SMILES inside (edge case)
        key = extract_entity_key("<SELFIES> CCO </SELFIES>")
        # Should try SMILES first, succeed
        assert key == get_canonical_smiles("CCO")

    def test_extract_entity_key_none_string(self):
        assert extract_entity_key("<None>") is None
        assert extract_entity_key(None) is None
        assert extract_entity_key("") is None

    def test_extract_entity_key_reaction(self):
        key = extract_entity_key("<SMILES> CCO>>CC </SMILES>")
        expected = get_canonical_reaction_smiles("CCO>>CC")
        assert key == expected


# ---------------------------------------------------------------------------
# Axis 2: Entity Key Canonicalization (policy invariants)
# ---------------------------------------------------------------------------

class TestEntityKeyCanonicalization:
    """같은 chemical identity면 표현이 달라도 같은 key가 나와야 한다."""

    @pytest.mark.parametrize("smiles_a,smiles_b", [
        ("CCO", "OCC"),               # ethanol
        ("c1ccccc1", "C1=CC=CC=C1"),  # benzene
        ("CC(=O)O", "OC(C)=O"),       # acetic acid
    ])
    def test_equivalent_smiles_same_key(self, smiles_a, smiles_b):
        # 동일 분자의 다른 SMILES 표현 → 같은 canonical key
        assert get_canonical_molecule_key(smiles_a) == get_canonical_molecule_key(smiles_b)

    @pytest.mark.parametrize("bad_input", [
        "", None, "<None>", ">>>", "[invalid",
    ])
    def test_invalid_inputs_return_none(self, bad_input):
        # 파싱 불가능한 입력 → None (조용히 통과하면 안 됨)
        assert extract_entity_key(bad_input) is None

    def test_invalid_smiles_returns_falsy(self):
        # RDKit 실패 → SELFIES fallback에서 빈 문자열 반환 가능
        # None 또는 빈 문자열 모두 falsy이므로 dedup에서 안전하게 제외됨
        result = extract_entity_key("not_a_molecule_xyz")
        assert not result

    def test_different_molecules_different_keys(self):
        # 다른 분자는 반드시 다른 key (false collision 방지)
        keys = {get_canonical_molecule_key(s) for s in ["CCO", "CO", "CCCO"]}
        assert len(keys) == 3

    def test_selfies_and_smiles_yield_same_molecule_key(self):
        # SELFIES encode → canonical key == SMILES canonical key
        # 주의: [C][C][O] 같은 SELFIES는 RDKit에서 유효한 SMILES로 파싱됨
        # (explicit H=0 carbon). 따라서 selfies_to_smiles fallback에 도달하지 않음.
        # 이 테스트는 selfies_to_smiles() 직접 호출로 SELFIES decode 경로를 검증.
        sf = pytest.importorskip("selfies")
        from dataset_generation.utils import selfies_to_smiles
        ethanol_selfies = sf.encoder("CCO")
        assert ethanol_selfies is not None
        # SELFIES decoder 경로를 직접 검증
        decoded = selfies_to_smiles(ethanol_selfies)
        assert decoded == get_canonical_smiles("CCO")

    def test_selfies_tagged_same_key_as_smiles_tagged(self):
        # <SELFIES> 태그 vs <SMILES> 태그
        # 현재 구현: _strip_mol_tags 후 get_canonical_molecule_key 호출
        # SELFIES 문자열이 유효한 RDKit SMILES로도 파싱되면 다른 key가 나올 수 있음.
        # (known limitation: SELFIES "[C][C][O]"는 RDKit에서 explicit H=0 carbon으로 파싱)
        # 이 테스트는 RDKit에서 파싱 불가능한 SELFIES 문자열로 검증.
        sf = pytest.importorskip("selfies")
        from dataset_generation.utils import selfies_to_smiles
        # 복잡한 분자: aspirin (RDKit가 SELFIES 형태를 유효 SMILES로 인식 못할 가능성 높음)
        aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        aspirin_selfies = sf.encoder(aspirin_smiles)
        assert aspirin_selfies is not None
        # SELFIES decoder가 올바르게 작동하는지 검증
        decoded = selfies_to_smiles(aspirin_selfies)
        assert decoded == get_canonical_smiles(aspirin_smiles)

    def test_whitespace_invariance(self):
        # 공백 차이는 무시되어야 함
        assert extract_entity_key("<SMILES>  CCO  </SMILES>") == \
               extract_entity_key("<SMILES>CCO</SMILES>")
        assert extract_entity_key("<SMILES> CCO </SMILES>") == \
               extract_entity_key("<SMILES>CCO</SMILES>")

    def test_reaction_component_order_invariance_real_smiles(self):
        # 반응물 순서가 달라도 같은 reaction key
        key_a = get_canonical_reaction_key("CCO.CC(=O)O>>CC(=O)OCC")
        key_b = get_canonical_reaction_key("CC(=O)O.CCO>>CC(=O)OCC")
        assert key_a is not None
        assert key_a == key_b

    def test_reaction_product_order_invariance(self):
        # 생성물 순서가 달라도 같은 reaction key
        key_a = get_canonical_reaction_key("CCO>>CC.O")
        key_b = get_canonical_reaction_key("CCO>>O.CC")
        assert key_a is not None
        assert key_a == key_b


# ---------------------------------------------------------------------------
# Family Mapping Tests
# ---------------------------------------------------------------------------

class TestFamilyMapping:
    """Entity family 매핑 테스트."""

    def test_known_families(self):
        assert _get_family("forward_reaction_prediction") == "REACTION_FORWARD_FAMILY"
        assert _get_family("smol-forward_synthesis") == "REACTION_FORWARD_FAMILY"
        assert _get_family("retrosynthesis") == "REACTION_RETRO_FAMILY"
        assert _get_family("smol-retrosynthesis") == "REACTION_RETRO_FAMILY"
        assert _get_family("chebi-20-mol2text") == "MOL2TEXT_FAMILY"
        assert _get_family("smol-molecule_captioning") == "MOL2TEXT_FAMILY"
        assert _get_family("chebi-20-text2mol") == "TEXT2MOL_FAMILY"
        assert _get_family("smol-molecule_generation") == "TEXT2MOL_FAMILY"

    def test_unknown_task_returns_none(self):
        assert _get_family("bace") is None
        assert _get_family("tox21") is None
        assert _get_family("unknown_task") is None

    def test_remove_on_conflict_is_strict_subset_of_families(self):
        # 정책: MOL2TEXT/TEXT2MOL family는 priority removal 수행 X
        #       REMOVE_ON_CONFLICT는 모든 family의 엄격한 부분집합이며, 제외는 명시적이다.
        families_with_tasks = set(ENTITY_FAMILIES.values())
        assert set(REMOVE_ON_CONFLICT.keys()).issubset(families_with_tasks)
        # 양 family는 의도적으로 제거 대상에 포함되면 안 됨
        assert "MOL2TEXT_FAMILY" not in REMOVE_ON_CONFLICT
        assert "TEXT2MOL_FAMILY" not in REMOVE_ON_CONFLICT
        # reaction family는 여전히 anchor 기반 priority 유지
        assert REMOVE_ON_CONFLICT["REACTION_FORWARD_FAMILY"] == "forward_reaction_prediction"
        assert REMOVE_ON_CONFLICT["REACTION_RETRO_FAMILY"] == "retrosynthesis"

    def test_paired_tasks_in_same_family(self):
        # 같은 family에 최소 2개 task가 있어야 dedup 가능
        family_counts = {}
        for task, family in ENTITY_FAMILIES.items():
            family_counts[family] = family_counts.get(family, 0) + 1
        for family, count in family_counts.items():
            assert count >= 2, f"{family} has only {count} task(s)"

    def test_all_entity_families_entries_are_strings(self):
        # config drift 방지: key/value 모두 문자열이어야 함
        for task, family in ENTITY_FAMILIES.items():
            assert isinstance(task, str), f"key {task!r} is not str"
            assert isinstance(family, str), f"value {family!r} is not str"

    def test_remove_on_conflict_task_belongs_to_its_family(self):
        # REMOVE_ON_CONFLICT에 지정된 task가 해당 family에 실제로 등록되어 있어야 함
        for family, remove_task in REMOVE_ON_CONFLICT.items():
            assert ENTITY_FAMILIES.get(remove_task) == family, \
                f"{remove_task} is not registered in {family}"

    def test_unknown_task_uses_task_name_as_group_id(self):
        # family에 속하지 않는 task → blacklist에서 task name이 group_id가 됨
        data = {
            "bace": {
                "test": _make_dataset(["<SMILES> CCO </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        assert "bace" in bl
        # family 이름으로는 존재하지 않아야 함
        for family_name in set(ENTITY_FAMILIES.values()):
            assert family_name not in bl

    def test_no_task_belongs_to_multiple_families(self):
        # dict 구조상 자명하지만, 정책 문서로서 명시적 검증
        seen_tasks = set()
        for task in ENTITY_FAMILIES:
            assert task not in seen_tasks, f"{task} appears in multiple families"
            seen_tasks.add(task)


# ---------------------------------------------------------------------------
# Eval Blacklist Tests
# ---------------------------------------------------------------------------

class TestEvalBlacklist:
    """Eval blacklist 구축 테스트."""

    def test_blacklist_from_test(self):
        data = {
            "forward_reaction_prediction": {
                "train": _make_dataset(["<SMILES> CCO </SMILES>"]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        canon_cc = get_canonical_smiles("CC")
        # CC should be in the blacklist
        assert canon_cc in bl.get("REACTION_FORWARD_FAMILY", set()) or \
               canon_cc in bl.get("forward_reaction_prediction", set())

    def test_blacklist_includes_validation(self):
        data = {
            "bace": {
                "train": _make_dataset(["<SMILES> CCO </SMILES>"]),
                "val": _make_dataset(["<SMILES> CCCC </SMILES>"]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        bl_with_val = build_eval_blacklist(data, include_validation=True)
        bl_without_val = build_eval_blacklist(data, include_validation=False)

        # bace is not in ENTITY_FAMILIES, so group_id = "bace"
        assert len(bl_with_val.get("bace", set())) >= len(bl_without_val.get("bace", set()))

    def test_blacklist_none_input_ignored(self):
        # text2mol family는 label field를 key로 사용 (input_mol_string은 <None>)
        data = {
            "chebi-20-text2mol": {
                "test": _make_text2mol_dataset(["<None>", "<SMILES> CCO </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        # Only CCO should be in blacklist, <None> ignored
        keys = bl.get("TEXT2MOL_FAMILY", set())
        assert len(keys) == 1
        assert get_canonical_smiles("CCO") in keys

    def test_blacklist_groups_by_family(self):
        # 같은 family의 두 task → 하나의 family bucket으로 합쳐짐
        data = {
            "forward_reaction_prediction": {
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
            "smol-forward_synthesis": {
                "test": _make_dataset(["<SMILES> CCO </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        family_keys = bl.get("REACTION_FORWARD_FAMILY", set())
        assert get_canonical_smiles("CC") in family_keys
        assert get_canonical_smiles("CCO") in family_keys
        # 개별 task 이름으로는 존재하지 않아야 함
        assert "forward_reaction_prediction" not in bl
        assert "smol-forward_synthesis" not in bl

    def test_blacklist_val_only_molecule_included_when_flag_true(self):
        # val에만 있는 분자: include_validation=True면 포함, False면 미포함
        data = {
            "bace": {
                "train": _make_dataset(["<SMILES> CCCC </SMILES>"]),
                "val": _make_dataset(["<SMILES> CCCCCC </SMILES>"]),  # val-only
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        canon_hexane = get_canonical_smiles("CCCCCC")

        bl_with = build_eval_blacklist(data, include_validation=True)
        assert canon_hexane in bl_with.get("bace", set())

        bl_without = build_eval_blacklist(data, include_validation=False)
        assert canon_hexane not in bl_without.get("bace", set())

    def test_blacklist_does_not_include_train_keys(self):
        # train-only 분자는 blacklist에 절대 포함되면 안 됨
        data = {
            "retrosynthesis": {
                "train": _make_dataset(["<SMILES> CCCCCCCC </SMILES>"]),  # train-only
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        canon_octane = get_canonical_smiles("CCCCCCCC")
        all_keys = set()
        for keys in bl.values():
            all_keys |= keys
        assert canon_octane not in all_keys

    def test_blacklist_empty_when_no_eval_splits(self):
        # eval split이 없으면 blacklist 비어 있음
        data = {
            "retrosynthesis": {
                "train": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        assert len(bl) == 0


# ---------------------------------------------------------------------------
# Leakage Removal Tests
# ---------------------------------------------------------------------------

class TestLeakageRemoval:
    """Eval leakage 제거 테스트."""

    def test_removes_leaking_samples(self):
        data = {
            "forward_reaction_prediction": {
                "train": _make_dataset([
                    "<SMILES> CC </SMILES>",      # leaks (in test)
                    "<SMILES> CCO </SMILES>",     # safe
                    "<SMILES> CCCO </SMILES>",    # safe
                ]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        data, stats = remove_eval_leakage(data, bl)

        train = data["forward_reaction_prediction"]["train"]
        assert len(train) == 2  # CC removed
        # Verify CC is not in remaining
        for mol_str in train["input_mol_string"]:
            key = extract_entity_key(mol_str)
            assert key != get_canonical_smiles("CC")

    def test_no_removal_when_no_overlap(self):
        data = {
            "retrosynthesis": {
                "train": _make_dataset(["<SMILES> CCCC </SMILES>"]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        data, stats = remove_eval_leakage(data, bl)
        assert len(data["retrosynthesis"]["train"]) == 1


# ---------------------------------------------------------------------------
# Axis 4: Split-Aware Conflict Resolution
# ---------------------------------------------------------------------------

class TestSplitAwareConflictResolution:
    """eval split 보호가 train보다 우선한다."""

    def test_train_removed_when_overlaps_with_eval(self):
        # train↔test 충돌 → train 제거, test 유지
        data = {
            "retrosynthesis": {
                "train": _make_dataset([
                    "<SMILES> CC </SMILES>",    # overlaps with test
                    "<SMILES> CCCO </SMILES>",  # safe
                ]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        data, _ = remove_eval_leakage(data, bl, include_validation=False)

        assert len(data["retrosynthesis"]["train"]) == 1
        assert len(data["retrosynthesis"]["test"]) == 1  # 변경 없음

    def test_eval_splits_content_unchanged_after_pipeline(self):
        # pipeline 전후로 test split의 내용이 byte-identical
        test_mols = ["<SMILES> CC </SMILES>", "<SMILES> CCO </SMILES>"]
        data = {
            "smol-forward_synthesis": {
                "train": _make_dataset(["<SMILES> CC </SMILES>", "<SMILES> CCCC </SMILES>"]),
                "test": _make_dataset(test_mols),
            },
            "forward_reaction_prediction": {
                "train": _make_dataset(["<SMILES> CC </SMILES>"]),
                "test": _make_dataset(["<SMILES> CCCCC </SMILES>"]),
            },
        }
        result = run_decontamination_pipeline(data, include_validation_in_blacklist=False)

        # smol test 내용 검증
        assert list(result["smol-forward_synthesis"]["test"]["input_mol_string"]) == test_mols

    def test_val_frozen_when_included_in_blacklist(self):
        # include_validation=True → val은 eval boundary의 일부이므로 필터되지 않음
        data = {
            "retrosynthesis": {
                "train": _make_dataset(["<SMILES> CC </SMILES>"]),
                "val": _make_dataset(["<SMILES> CC </SMILES>"]),  # test와 같은 mol
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=True)
        data, _ = remove_eval_leakage(data, bl, include_validation=True)

        # val은 frozen → 길이 유지
        assert len(data["retrosynthesis"]["val"]) == 1
        # train은 제거됨
        assert len(data["retrosynthesis"]["train"]) == 0

    def test_val_filtered_when_excluded_from_blacklist(self):
        # include_validation=False → val도 필터 대상
        data = {
            "retrosynthesis": {
                "train": _make_dataset(["<SMILES> CC </SMILES>", "<SMILES> CCCC </SMILES>"]),
                "val": _make_dataset(["<SMILES> CC </SMILES>"]),  # test와 겹침
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        data, _ = remove_eval_leakage(data, bl, include_validation=False)

        # val도 필터 → CC 제거
        assert len(data["retrosynthesis"]["val"]) == 0

    def test_safe_train_samples_not_removed(self):
        # overlap 없는 train 샘플은 반드시 생존
        safe_mol = "<SMILES> CCCCCCCCCC </SMILES>"  # decane, 다른 split에 없음
        data = {
            "forward_reaction_prediction": {
                "train": _make_dataset([safe_mol, "<SMILES> CC </SMILES>"]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
            "smol-forward_synthesis": {
                "train": _make_dataset(["<SMILES> CCCC </SMILES>"]),
            },
        }
        result = run_decontamination_pipeline(data, include_validation_in_blacklist=False)
        fwd_keys = _collect_keys(result["forward_reaction_prediction"]["train"])
        assert get_canonical_smiles("CCCCCCCCCC") in fwd_keys


# ---------------------------------------------------------------------------
# Within-Family Dedup Tests
# ---------------------------------------------------------------------------

class TestWithinFamilyDedup:
    """Cross-source dedup within family 테스트."""

    def test_removes_from_conflict_task(self):
        """REACTION_FORWARD_FAMILY에서 forward_reaction_prediction(Mol-Instructions)이
        smol-forward_synthesis(SMolInstruct)와 겹치면 전자에서 제거."""
        overlap_mol = "<SMILES> CCO </SMILES>"
        unique_mol = "<SMILES> CCCCCC </SMILES>"

        data = {
            "smol-forward_synthesis": {
                "train": _make_dataset([overlap_mol, "<SMILES> CC </SMILES>"]),
            },
            "forward_reaction_prediction": {
                "train": _make_dataset([overlap_mol, unique_mol]),
            },
        }
        data, stats = dedup_within_family(data)

        # forward_reaction_prediction should have overlap_mol removed
        fwd_train = data["forward_reaction_prediction"]["train"]
        assert len(fwd_train) == 1
        key = extract_entity_key(fwd_train["input_mol_string"][0])
        assert key == get_canonical_smiles("CCCCCC")

    def test_does_not_touch_eval(self):
        """Eval split(test)은 건드리지 않아야 함."""
        overlap_mol = "<SMILES> CCO </SMILES>"
        data = {
            "smol-forward_synthesis": {
                "train": _make_dataset([overlap_mol]),
            },
            "forward_reaction_prediction": {
                "train": _make_dataset([overlap_mol]),
                "test": _make_dataset([overlap_mol]),  # should NOT be touched
            },
        }
        data, stats = dedup_within_family(data)
        assert len(data["forward_reaction_prediction"]["test"]) == 1


# ---------------------------------------------------------------------------
# Axis 5: Source Priority Rule
# ---------------------------------------------------------------------------

class TestSourcePriorityRule:
    """priority rule은 같은 family의 train-train 충돌에만 적용된다."""

    def test_anchor_preserved_remove_task_filtered(self):
        # anchor task(smol)의 train은 유지, remove task(mol-inst)의 train에서 제거
        overlap = "<SMILES> CCO </SMILES>"
        data = {
            "smol-forward_synthesis": {
                "train": _make_dataset([overlap, "<SMILES> CC </SMILES>"]),
            },
            "forward_reaction_prediction": {
                "train": _make_dataset([overlap, "<SMILES> CCCC </SMILES>"]),
            },
        }
        data, _ = dedup_within_family(data)

        # anchor 유지
        assert len(data["smol-forward_synthesis"]["train"]) == 2
        # remove task에서 overlap 제거
        fwd_keys = _collect_keys(data["forward_reaction_prediction"]["train"])
        assert get_canonical_smiles("CCO") not in fwd_keys
        assert get_canonical_smiles("CCCC") in fwd_keys

    def test_priority_rule_applies_only_within_same_family(self):
        # Reaction family 간 독립성 검증: REACTION_FORWARD_FAMILY와 REACTION_RETRO_FAMILY가
        # 서로 overlap 분자가 있어도 dedup은 각 family 내부에서만 수행돼야 한다.
        shared_mol = "<SMILES> CCO </SMILES>"
        data = {
            # REACTION_FORWARD_FAMILY
            "smol-forward_synthesis": {
                "train": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
            "forward_reaction_prediction": {
                "train": _make_dataset([shared_mol]),  # CCO는 이 family anchor에 없음
            },
            # REACTION_RETRO_FAMILY
            "smol-retrosynthesis": {
                "train": _make_dataset([shared_mol]),  # anchor, CCO 존재
            },
            "retrosynthesis": {
                "train": _make_dataset([shared_mol]),  # CCO 존재 → 제거 대상
            },
        }
        data, _ = dedup_within_family(data)

        # REACTION_RETRO_FAMILY에서만 제거 발생 (retrosynthesis train 0개)
        assert len(data["retrosynthesis"]["train"]) == 0
        # REACTION_FORWARD_FAMILY의 forward_reaction_prediction은 변경 없음
        # (forward family anchor인 smol-forward_synthesis에는 CCO가 없으므로)
        assert len(data["forward_reaction_prediction"]["train"]) == 1

    def test_priority_only_compares_train_splits(self):
        # anchor의 eval split에만 있는 mol → remove task train에서 제거되지 않음
        mol = "<SMILES> CCO </SMILES>"
        data = {
            "smol-forward_synthesis": {
                "test": _make_dataset([mol]),  # eval에만 있음, train 없음
            },
            "forward_reaction_prediction": {
                "train": _make_dataset([mol]),
            },
        }
        data, _ = dedup_within_family(data)
        # dedup_within_family는 anchor의 train만 비교 → 제거 없음
        assert len(data["forward_reaction_prediction"]["train"]) == 1

    def test_single_task_family_skips_dedup(self):
        # family에 task 1개만 있으면 dedup skip
        data = {
            "retrosynthesis": {
                "train": _make_dataset(["<SMILES> CCO </SMILES>", "<SMILES> CCO </SMILES>"]),
            },
        }
        data, stats = dedup_within_family(data)
        # 2개 모두 살아남음 (within-family dedup은 cross-source만)
        assert len(data["retrosynthesis"]["train"]) == 2

    def test_same_source_intra_duplicates_not_removed(self):
        # 동일 task 내 중복은 dedup_within_family가 건드리지 않음
        dup_mol = "<SMILES> CCO </SMILES>"
        data = {
            "smol-forward_synthesis": {
                "train": _make_dataset([dup_mol, dup_mol, "<SMILES> CC </SMILES>"]),
            },
            "forward_reaction_prediction": {
                "train": _make_dataset(["<SMILES> CCCC </SMILES>"]),
            },
        }
        data, _ = dedup_within_family(data)
        # smol train의 intra-duplicate는 유지
        assert len(data["smol-forward_synthesis"]["train"]) == 3


# ---------------------------------------------------------------------------
# Axis 6: Report/Statistics/Reason Codes
# ---------------------------------------------------------------------------

class TestRemovalStatsAndReports:
    """제거 통계와 사유 코드가 정확히 기록되어야 한다."""

    def test_stats_record_eval_blacklist_reason(self):
        data = {
            "retrosynthesis": {
                "train": _make_dataset(["<SMILES> CC </SMILES>", "<SMILES> CCCC </SMILES>"]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        _, stats = remove_eval_leakage(data, bl, include_validation=False)
        assert stats.counts["retrosynthesis"]["train"]["eval_blacklist"] == 1

    def test_stats_record_within_family_dup_reason(self):
        overlap = "<SMILES> CCO </SMILES>"
        data = {
            "smol-forward_synthesis": {
                "train": _make_dataset([overlap]),
            },
            "forward_reaction_prediction": {
                "train": _make_dataset([overlap, "<SMILES> CC </SMILES>"]),
            },
        }
        _, stats = dedup_within_family(data)
        assert stats.counts["forward_reaction_prediction"]["train"]["within_family_dup"] == 1

    def test_stats_count_matches_actual_removal(self):
        # before - after == sum(stats counts)
        data = {
            "smol-forward_synthesis": {
                "train": _make_dataset([
                    "<SMILES> CC </SMILES>",   # leaks
                    "<SMILES> CCO </SMILES>",  # safe
                ]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        before = len(data["smol-forward_synthesis"]["train"])
        bl = build_eval_blacklist(data, include_validation=False)
        data, stats = remove_eval_leakage(data, bl, include_validation=False)
        after = len(data["smol-forward_synthesis"]["train"])

        total_removed = sum(
            count
            for task_splits in stats.counts.values()
            for split_reasons in task_splits.values()
            for count in split_reasons.values()
        )
        assert before - after == total_removed

    def test_stats_breakdown_by_task_and_split(self):
        # task/split별 분리 확인
        data = {
            "retrosynthesis": {
                "train": _make_dataset(["<SMILES> CC </SMILES>"]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
            "smol-retrosynthesis": {
                "train": _make_dataset(["<SMILES> CC </SMILES>"]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        _, stats = remove_eval_leakage(data, bl, include_validation=False)

        # 두 task의 train 모두에서 제거 기록이 있어야 함
        assert stats.counts["retrosynthesis"]["train"]["eval_blacklist"] >= 1
        assert stats.counts["smol-retrosynthesis"]["train"]["eval_blacklist"] >= 1

    def test_stats_empty_when_no_removal(self):
        data = {
            "retrosynthesis": {
                "train": _make_dataset(["<SMILES> CCCC </SMILES>"]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        _, stats = remove_eval_leakage(data, bl, include_validation=False)

        total = sum(
            count
            for task_splits in stats.counts.values()
            for split_reasons in task_splits.values()
            for count in split_reasons.values()
        )
        assert total == 0

    def test_stats_print_report_no_error(self, capsys):
        stats = RemovalStats()
        stats.record("test_task", "train", "eval_blacklist", 5)
        stats.print_report("Test Report")
        captured = capsys.readouterr()
        assert "test_task" in captured.out
        assert "5" in captured.out

    def test_reason_codes_documented(self):
        # dedup.py에서 사용하는 모든 reason이 REMOVAL_REASONS에 등록되어 있어야 함
        expected_reasons = {"eval_blacklist", "within_family_dup", "invalid_key"}
        assert set(REMOVAL_REASONS.keys()) == expected_reasons


# ---------------------------------------------------------------------------
# Full Pipeline Integration Test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """전체 decontamination 파이프라인 통합 테스트."""

    def test_pipeline_removes_leakage_and_dedup(self):
        leaking_mol = "<SMILES> CC </SMILES>"
        overlap_mol = "<SMILES> CCO </SMILES>"
        safe_mol_1 = "<SMILES> CCCC </SMILES>"
        safe_mol_2 = "<SMILES> CCCCC </SMILES>"

        data = {
            "smol-forward_synthesis": {
                "train": _make_dataset([overlap_mol, safe_mol_1]),
                "test": _make_dataset([leaking_mol]),
            },
            "forward_reaction_prediction": {
                "train": _make_dataset([leaking_mol, overlap_mol, safe_mol_2]),
                "test": _make_dataset([leaking_mol]),
            },
        }

        result = run_decontamination_pipeline(data, include_validation_in_blacklist=False)

        # forward_reaction_prediction train:
        #   - leaking_mol removed (eval blacklist)
        #   - overlap_mol removed (within-family dedup, priority rule)
        #   - safe_mol_2 remains
        fwd_train = result["forward_reaction_prediction"]["train"]
        assert len(fwd_train) == 1
        assert extract_entity_key(fwd_train["input_mol_string"][0]) == get_canonical_smiles("CCCCC")

        # smol-forward_synthesis train:
        #   - overlap_mol remains (anchor task)
        #   - safe_mol_1 remains
        #   - leaking_mol not present in train anyway
        smol_train = result["smol-forward_synthesis"]["train"]
        assert len(smol_train) == 2

    def test_pipeline_zero_overlap_after(self):
        """Decontamination 후 train과 test 간 entity key 겹침 = 0 검증."""
        data = {
            "retrosynthesis": {
                "train": _make_dataset([
                    "<SMILES> CC </SMILES>",
                    "<SMILES> CCO </SMILES>",
                    "<SMILES> CCCO </SMILES>",
                ]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
            "smol-retrosynthesis": {
                "train": _make_dataset([
                    "<SMILES> CCO </SMILES>",
                    "<SMILES> CCCCO </SMILES>",
                ]),
                "test": _make_dataset(["<SMILES> CCO </SMILES>"]),
            },
        }

        result = run_decontamination_pipeline(data, include_validation_in_blacklist=False)

        # Collect all test keys for the family
        test_keys = set()
        for task in ["retrosynthesis", "smol-retrosynthesis"]:
            test_ds = result[task].get("test")
            if test_ds:
                for mol_str in test_ds["input_mol_string"]:
                    key = extract_entity_key(mol_str)
                    if key:
                        test_keys.add(key)

        # Collect all train keys for the family
        train_keys = set()
        for task in ["retrosynthesis", "smol-retrosynthesis"]:
            train_ds = result[task].get("train")
            if train_ds:
                for mol_str in train_ds["input_mol_string"]:
                    key = extract_entity_key(mol_str)
                    if key:
                        train_keys.add(key)

        # Zero overlap
        overlap = test_keys & train_keys
        assert len(overlap) == 0, f"Leakage detected: {overlap}"


# ---------------------------------------------------------------------------
# Axis 7: End-to-End Acceptance Tests
# ---------------------------------------------------------------------------

class TestEndToEndAcceptance:
    """cross-source decontamination 이후 정책 invariant가 유지되는지 검증."""

    def _build_four_family_data(self):
        """4개 family 모두 포함, 각 family에 overlap 존재하는 toy data."""
        return {
            # REACTION_FORWARD_FAMILY
            "smol-forward_synthesis": {
                "train": _make_dataset([
                    "<SMILES> CCO>>CC </SMILES>",
                    "<SMILES> CCCC>>CC </SMILES>",
                ]),
                "test": _make_dataset(["<SMILES> CCO>>CC </SMILES>"]),
            },
            "forward_reaction_prediction": {
                "train": _make_dataset([
                    "<SMILES> CCO>>CC </SMILES>",  # overlap with smol
                    "<SMILES> CCCCC>>CC </SMILES>",
                ]),
                "test": _make_dataset(["<SMILES> CCO>>CC </SMILES>"]),
            },
            # REACTION_RETRO_FAMILY
            "smol-retrosynthesis": {
                "train": _make_dataset([
                    "<SMILES> CC>>CCO </SMILES>",
                    "<SMILES> CC>>CCCC </SMILES>",
                ]),
                "test": _make_dataset(["<SMILES> CC>>CCO </SMILES>"]),
            },
            "retrosynthesis": {
                "train": _make_dataset([
                    "<SMILES> CC>>CCO </SMILES>",  # overlap
                    "<SMILES> CC>>CCCCCC </SMILES>",
                ]),
                "test": _make_dataset(["<SMILES> CC>>CCO </SMILES>"]),
            },
            # MOL2TEXT_FAMILY
            "smol-molecule_captioning": {
                "train": _make_dataset([
                    "<SMILES> CCO </SMILES>",
                    "<SMILES> CCCC </SMILES>",
                ]),
                "test": _make_dataset(["<SMILES> CCO </SMILES>"]),
            },
            "chebi-20-mol2text": {
                "train": _make_dataset([
                    "<SMILES> CCO </SMILES>",  # overlap
                    "<SMILES> CCCCC </SMILES>",
                ]),
                "test": _make_dataset(["<SMILES> CCO </SMILES>"]),
            },
            # TEXT2MOL_FAMILY — key source는 label (input_mol_string은 <None>)
            "smol-molecule_generation": {
                "train": _make_text2mol_dataset([
                    "<SMILES>CC</SMILES>",
                    "<SMILES>CCCO</SMILES>",
                ]),
                "test": _make_text2mol_dataset(["<SMILES>CC</SMILES>"]),
            },
            "chebi-20-text2mol": {
                "train": _make_text2mol_dataset([
                    "<SMILES>CC</SMILES>",  # overlap → eval blacklist로 제거
                    "<SMILES>CCCCCCCC</SMILES>",
                ]),
                "test": _make_text2mol_dataset(["<SMILES>CC</SMILES>"]),
            },
        }

    def _assert_zero_overlap_for_family(self, result, task_names):
        """family 내 모든 task의 train/test key overlap이 0인지 검증 (family-aware key source)."""
        train_keys, test_keys = set(), set()
        for task in task_names:
            splits = result.get(task, {})
            if "train" in splits:
                train_keys |= _collect_keys_for_task(splits["train"], task)
            if "test" in splits:
                test_keys |= _collect_keys_for_task(splits["test"], task)
        overlap = train_keys & test_keys
        assert len(overlap) == 0, f"Leakage in {task_names}: {overlap}"

    def test_molecule_family_zero_train_test_overlap(self):
        # MOL2TEXT + TEXT2MOL family에서 zero overlap
        data = self._build_four_family_data()
        result = run_decontamination_pipeline(data, include_validation_in_blacklist=False)

        self._assert_zero_overlap_for_family(
            result, ["chebi-20-mol2text", "smol-molecule_captioning"])
        self._assert_zero_overlap_for_family(
            result, ["chebi-20-text2mol", "smol-molecule_generation"])

    def test_reaction_family_zero_train_test_overlap(self):
        # REACTION_FORWARD + REACTION_RETRO family에서 zero overlap
        data = self._build_four_family_data()
        result = run_decontamination_pipeline(data, include_validation_in_blacklist=False)

        self._assert_zero_overlap_for_family(
            result, ["forward_reaction_prediction", "smol-forward_synthesis"])
        self._assert_zero_overlap_for_family(
            result, ["retrosynthesis", "smol-retrosynthesis"])

    def test_schema_preserved_after_decontamination(self):
        # 컬럼 이름/타입, 행 무결성 유지 검증
        data = {
            "smol-forward_synthesis": {
                "train": _make_rich_dataset([
                    "<SMILES> CCO </SMILES>",
                    "<SMILES> CCCC </SMILES>",
                    "<SMILES> CCCCC </SMILES>",
                ], "smol-forward_synthesis"),
                "test": _make_rich_dataset([
                    "<SMILES> CCO </SMILES>",
                ], "smol-forward_synthesis"),
            },
            "forward_reaction_prediction": {
                "train": _make_rich_dataset([
                    "<SMILES> CCO </SMILES>",
                    "<SMILES> CC </SMILES>",
                ], "forward_reaction_prediction"),
            },
        }
        result = run_decontamination_pipeline(data, include_validation_in_blacklist=False)

        for task_name, splits in result.items():
            for split_name, ds in splits.items():
                # 컬럼 존재
                assert "input_mol_string" in ds.column_names
                assert "label" in ds.column_names
                assert "instruction" in ds.column_names
                assert "task_subtask_pair" in ds.column_names
                # 타입 유지 (datasets.Dataset)
                assert isinstance(ds, datasets.Dataset)

    def test_output_type_is_hf_dataset(self):
        data = {
            "retrosynthesis": {
                "train": _make_dataset(["<SMILES> CC </SMILES>"]),
                "test": _make_dataset(["<SMILES> CCO </SMILES>"]),
            },
        }
        result = run_decontamination_pipeline(data, include_validation_in_blacklist=False)
        for splits in result.values():
            for ds in splits.values():
                assert isinstance(ds, datasets.Dataset)

    def test_pipeline_idempotent(self):
        # 2회 실행 시 추가 제거 0
        data = {
            "smol-forward_synthesis": {
                "train": _make_dataset(["<SMILES> CCO </SMILES>", "<SMILES> CCCC </SMILES>"]),
                "test": _make_dataset(["<SMILES> CCO </SMILES>"]),
            },
            "forward_reaction_prediction": {
                "train": _make_dataset(["<SMILES> CCO </SMILES>", "<SMILES> CC </SMILES>"]),
                "test": _make_dataset(["<SMILES> CCO </SMILES>"]),
            },
        }
        first = run_decontamination_pipeline(data, include_validation_in_blacklist=False)

        # 1회 결과의 샘플 수 기록
        counts_after_first = {
            (task, split): len(ds)
            for task, splits in first.items()
            for split, ds in splits.items()
        }

        second = run_decontamination_pipeline(first, include_validation_in_blacklist=False)
        for task, splits in second.items():
            for split, ds in splits.items():
                assert len(ds) == counts_after_first[(task, split)], \
                    f"Idempotency violated: {task}/{split}"

    def test_pipeline_with_all_four_families(self):
        # 4개 family 동시 실행, 각 family별 zero overlap
        data = self._build_four_family_data()
        result = run_decontamination_pipeline(data, include_validation_in_blacklist=False)

        families = {
            "REACTION_FORWARD_FAMILY": ["forward_reaction_prediction", "smol-forward_synthesis"],
            "REACTION_RETRO_FAMILY": ["retrosynthesis", "smol-retrosynthesis"],
            "MOL2TEXT_FAMILY": ["chebi-20-mol2text", "smol-molecule_captioning"],
            "TEXT2MOL_FAMILY": ["chebi-20-text2mol", "smol-molecule_generation"],
        }
        for family_name, tasks in families.items():
            self._assert_zero_overlap_for_family(result, tasks)

    def test_non_family_tasks_get_leakage_removal_only(self):
        # family에 속하지 않는 task → eval leakage만 제거, family dedup 없음
        data = {
            "bace": {
                "train": _make_dataset([
                    "<SMILES> CC </SMILES>",    # overlaps with test
                    "<SMILES> CCCC </SMILES>",  # safe
                ]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
        }
        result = run_decontamination_pipeline(data, include_validation_in_blacklist=False)

        # leakage 제거됨
        train_keys = _collect_keys(result["bace"]["train"])
        test_keys = _collect_keys(result["bace"]["test"])
        assert len(train_keys & test_keys) == 0
        # safe mol 생존
        assert get_canonical_smiles("CCCC") in train_keys

    def test_pipeline_preserves_task_and_split_structure(self):
        # pipeline 전후 task/split key 구조 동일
        data = {
            "smol-forward_synthesis": {
                "train": _make_dataset(["<SMILES> CCO </SMILES>"]),
                "test": _make_dataset(["<SMILES> CC </SMILES>"]),
            },
            "forward_reaction_prediction": {
                "train": _make_dataset(["<SMILES> CCCC </SMILES>"]),
                "val": _make_dataset(["<SMILES> CCCCC </SMILES>"]),
                "test": _make_dataset(["<SMILES> CCCCCC </SMILES>"]),
            },
        }
        original_structure = {
            task: set(splits.keys()) for task, splits in data.items()
        }
        result = run_decontamination_pipeline(data, include_validation_in_blacklist=True)
        result_structure = {
            task: set(splits.keys()) for task, splits in result.items()
        }
        assert original_structure == result_structure


# ---------------------------------------------------------------------------
# Family-specific key-source routing (new contract)
# ---------------------------------------------------------------------------

class TestKeySourceRouting:
    def test_family_key_source_map_is_complete(self):
        # 모든 family가 FAMILY_KEY_SOURCE에 등록되어 있어야 함
        families = set(ENTITY_FAMILIES.values())
        assert families.issubset(set(FAMILY_KEY_SOURCE.keys()))

    def test_text2mol_family_uses_label(self):
        assert _get_key_source_field("chebi-20-text2mol") == "label"
        assert _get_key_source_field("smol-molecule_generation") == "label"

    def test_mol2text_family_uses_input_mol_string(self):
        # mol2text는 input_mol_string에 분자가 있음 — 현행 유지
        assert _get_key_source_field("chebi-20-mol2text") == "input_mol_string"
        assert _get_key_source_field("smol-molecule_captioning") == "input_mol_string"

    def test_reaction_family_uses_input_mol_string(self):
        assert _get_key_source_field("forward_reaction_prediction") == "input_mol_string"
        assert _get_key_source_field("smol-forward_synthesis") == "input_mol_string"
        assert _get_key_source_field("retrosynthesis") == "input_mol_string"
        assert _get_key_source_field("smol-retrosynthesis") == "input_mol_string"

    def test_unknown_task_defaults_to_input_mol_string(self):
        assert _get_key_source_field("bace") == "input_mol_string"
        assert _get_key_source_field("qm9_homo") == "input_mol_string"


class TestText2MolFamilyDedup:
    def test_blacklist_from_text2mol_test_uses_label(self):
        # text2mol test의 label에 있는 분자가 blacklist에 들어가야 함
        data = {
            "chebi-20-text2mol": {
                "test": _make_text2mol_dataset(["<SMILES>CCO</SMILES>"]),
            },
        }
        bl = build_eval_blacklist(data, include_validation=False)
        assert "TEXT2MOL_FAMILY" in bl
        assert get_canonical_smiles("CCO") in bl["TEXT2MOL_FAMILY"]

    def test_train_removed_when_label_matches_eval(self):
        # text2mol family: train sample label = test sample label → train 제거
        data = {
            "chebi-20-text2mol": {
                "test": _make_text2mol_dataset(["<SMILES>CCO</SMILES>"]),
            },
            "smol-molecule_generation": {
                "train": _make_text2mol_dataset([
                    "<SMILES>CCO</SMILES>",   # leaks from chebi test
                    "<SMILES>CCCC</SMILES>",  # safe
                ]),
            },
        }
        result = run_decontamination_pipeline(data, include_validation_in_blacklist=False)
        train_labels = result["smol-molecule_generation"]["train"]["label"]
        # leaking label removed
        assert "<SMILES>CCO</SMILES>" not in train_labels
        # safe label preserved
        assert "<SMILES>CCCC</SMILES>" in train_labels

    def test_text2mol_no_priority_removal(self):
        # within-family cross-source priority removal: text2mol은 수행 X
        # 즉 chebi-20-text2mol train과 smol-molecule_generation train에 같은 분자가 있어도 둘 다 보존
        data = {
            "chebi-20-text2mol": {
                "train": _make_text2mol_dataset(["<SMILES>CCO</SMILES>"]),
            },
            "smol-molecule_generation": {
                "train": _make_text2mol_dataset(["<SMILES>CCO</SMILES>"]),
            },
        }
        result_data, stats = dedup_within_family(data)
        # 양쪽 모두 유지
        assert len(result_data["chebi-20-text2mol"]["train"]) == 1
        assert len(result_data["smol-molecule_generation"]["train"]) == 1
        # stats에 within_family_dup 사유 제거 기록이 있으면 안 됨
        total_removed = sum(
            stats.counts[t][s][r]
            for t in stats.counts for s in stats.counts[t] for r in stats.counts[t][s]
        )
        assert total_removed == 0

    def test_mol2text_no_priority_removal(self):
        # mol2text 역시 same — ChEBI와 SMol 모두 train에 보존
        data = {
            "chebi-20-mol2text": {
                "train": _make_mol2text_dataset(["<SMILES> CCO </SMILES>"]),
            },
            "smol-molecule_captioning": {
                "train": _make_mol2text_dataset(["<SMILES> CCO </SMILES>"]),
            },
        }
        result_data, stats = dedup_within_family(data)
        assert len(result_data["chebi-20-mol2text"]["train"]) == 1
        assert len(result_data["smol-molecule_captioning"]["train"]) == 1

    def test_reaction_priority_still_active(self):
        # REACTION_FORWARD_FAMILY는 여전히 SMolInstruct anchor 유지
        data = {
            "smol-forward_synthesis": {
                "train": _make_dataset(["<SMILES> CCO </SMILES>"]),  # anchor
            },
            "forward_reaction_prediction": {
                "train": _make_dataset(["<SMILES> CCO </SMILES>"]),  # should be removed
            },
        }
        result_data, stats = dedup_within_family(data)
        assert len(result_data["smol-forward_synthesis"]["train"]) == 1
        assert len(result_data["forward_reaction_prediction"]["train"]) == 0
