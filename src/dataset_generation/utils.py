"""Shared utility functions for dataset generation.

Extracted from the original dataset_generator.py for reuse across
generator.py, dedup.py, and run.py.
"""

import os
import re

# NOTE: rdkit, selfies, torch, torch_geometric는 모두 함수 내부에서 lazy import한다.
# dataset_gen venv에는 이들이 설치되어 있지 않을 수 있으므로,
# lazy import로 실제 호출 시점까지 지연시켜 ImportError를 회피한다.


# ---------------------------------------------------------------------------
# Molecular string utilities
# ---------------------------------------------------------------------------

def clean_mol_string(mol_str):
    """분자 문자열에서 세미콜론(;)을 점(.)으로 변환 — 다중 분자 구분자 정규화"""
    return re.sub(r"\s*;\s*", ".", str(mol_str))


def get_canonical_smiles(smiles):
    """RDKit canonical SMILES 변환. 실패 시 None 반환."""
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        return None
    except Exception:
        return None


def selfies_to_smiles(selfies_str):
    """SELFIES → canonical SMILES 변환. 실패 시 None 반환."""
    import selfies as sf
    try:
        smiles = sf.decoder(selfies_str)
        if smiles is None:
            return None
        canonical = get_canonical_smiles(smiles)
        return canonical if canonical else smiles
    except Exception:
        return None


def smiles_to_selfies(smiles):
    """SMILES → SELFIES 변환. 실패 시 None 반환.

    내부적으로 canonical SMILES로 정규화한 뒤 SELFIES로 인코딩한다.
    """
    import selfies as sf
    try:
        canonical = get_canonical_smiles(smiles)
        if canonical is None:
            return None
        return sf.encoder(canonical)
    except Exception:
        return None


def convert_mol_representation(mol_string, target_repr):
    """분자 문자열을 target_repr('smiles' 또는 'selfies')로 변환.

    reaction SMILES ('>>' 포함)도 처리: 각 side의 분자를 개별 변환.
    실패 시 None 반환.
    """
    if not mol_string or mol_string == "<None>":
        return mol_string

    if target_repr == "smiles":
        # SELFIES일 수 있으므로 canonical SMILES로 시도, 실패 시 SELFIES decode
        if ">>" in mol_string:
            return get_canonical_reaction_smiles(mol_string)
        canonical = get_canonical_smiles(mol_string)
        if canonical is not None:
            return canonical
        return selfies_to_smiles(mol_string)

    elif target_repr == "selfies":
        if ">>" in mol_string:
            # reaction: 각 side 분자를 개별 SELFIES 변환
            sides = mol_string.split(">>")
            converted_sides = []
            for side in sides:
                molecules = side.split(".")
                converted = []
                for m in molecules:
                    m = m.strip()
                    if not m:
                        continue
                    sf_str = smiles_to_selfies(m)
                    if sf_str is None:
                        # SMILES 파싱 실패 → SELFIES→SMILES→SELFIES 시도
                        decoded = selfies_to_smiles(m)
                        if decoded:
                            sf_str = smiles_to_selfies(decoded)
                    if sf_str is None:
                        return None
                    converted.append(sf_str)
                converted_sides.append(".".join(converted))
            return ">>".join(converted_sides)
        else:
            # 단일 분자
            sf_str = smiles_to_selfies(mol_string)
            if sf_str is not None:
                return sf_str
            # SMILES 파싱 실패 → 이미 SELFIES일 수 있음 → decode해서 재변환
            decoded = selfies_to_smiles(mol_string)
            if decoded:
                return smiles_to_selfies(decoded)
            return None

    return mol_string


def get_canonical_reaction_smiles(reaction_smiles):
    """반응 SMILES를 canonical 형태로 변환.

    'A.B>>C.D' → 각 side의 분자를 개별 canonicalize 후 정렬하여 재조합.
    실패 시 None 반환.
    """
    try:
        if ">>" not in reaction_smiles:
            return get_canonical_smiles(reaction_smiles)

        sides = reaction_smiles.split(">>")
        canonical_sides = []
        for side in sides:
            molecules = side.split(".")
            canonical_mols = []
            for mol_str in molecules:
                mol_str = mol_str.strip()
                if not mol_str:
                    continue
                canonical = get_canonical_smiles(mol_str)
                if canonical is None:
                    return None
                canonical_mols.append(canonical)
            canonical_mols.sort()
            canonical_sides.append(".".join(canonical_mols))
        return ">>".join(canonical_sides)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Graph conversion
# ---------------------------------------------------------------------------

_DUMMY_GRAPH = None


def get_dummy_graph():
    """smiles2data('CC') 결과를 캐싱하여 반복 호출 방지"""
    global _DUMMY_GRAPH
    if _DUMMY_GRAPH is None:
        _DUMMY_GRAPH = smiles2data("CC")
    return _DUMMY_GRAPH


def smiles2data(smiles):
    """SMILES → PyG Data 변환 (OGB smiles2graph 사용).

    torch와 torch_geometric은 이 함수 호출 시에만 lazy import된다.
    dedup 등 graph가 불필요한 모듈에서는 torch를 로드하지 않는다.
    """
    import torch
    from ogb.utils import smiles2graph
    from torch_geometric.data import Data
    try:
        graph = smiles2graph(smiles)
        x = torch.from_numpy(graph["node_feat"])
        edge_index = torch.from_numpy(graph["edge_index"])
        edge_attr = torch.from_numpy(graph["edge_feat"])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    except Exception as e:
        raise ValueError(f"Failed to convert SMILES to graph: {smiles}. Error: {e}")


# ---------------------------------------------------------------------------
# Entity key extraction (for decontamination)
#
# 함수 계층:
#   extract_entity_key()          ← dedup.py에서 호출하는 최상위 진입점
#     ├── get_canonical_molecule_key()   ← 단일 분자
#     └── get_canonical_reaction_key()   ← 반응 SMILES
# ---------------------------------------------------------------------------

def get_canonical_molecule_key(raw_string):
    """단일 분자 문자열 → canonical SMILES key.

    SMILES 파싱을 먼저 시도하고, 실패하면 SELFIES decode를 시도한다.
    실패 시 None 반환.
    """
    if not raw_string:
        return None
    canonical = get_canonical_smiles(raw_string)
    if canonical is not None:
        return canonical
    # SMILES 파싱 실패 → SELFIES로 시도
    decoded = selfies_to_smiles(raw_string)
    return decoded


def get_canonical_reaction_key(rxn_string):
    """반응 SMILES → canonical reaction key.

    '>>' 기준 분리 → 각 side 분자를 개별 canonicalize → 정렬 → 재조합.

    NOTE: 현재 decontamination은 reaction string의 canonical component
    normalization 수준의 exact/near-exact matching이며, full reaction-role
    harmonization (agent/reagent 위치, atom mapping 차이 등)까지를 목표로
    하지는 않는다.
    """
    return get_canonical_reaction_smiles(rxn_string)


def _strip_mol_tags(s):
    """<SMILES>, </SMILES>, <SELFIES>, </SELFIES> 태그를 제거."""
    s = re.sub(r"</?SMILES>", "", s)
    s = re.sub(r"</?SELFIES>", "", s)
    return s.strip()


def extract_entity_key(input_mol_string):
    """input_mol_string에서 decontamination 비교용 canonical entity key를 추출.

    내부적으로 get_canonical_molecule_key() 또는 get_canonical_reaction_key()로 분기.
    dedup 비교 키는 항상 RDKit canonical SMILES로 통일되며,
    SMILES/SELFIES 표현 형식과 무관하게 동일 entity를 정확히 매칭한다.

    Args:
        input_mol_string: Arrow 데이터셋의 input_mol_string 컬럼 값

    Returns:
        canonical key 문자열, 또는 None (비교 불가능한 경우: <None>, 빈 문자열 등)
    """
    if input_mol_string is None:
        return None

    s = _strip_mol_tags(str(input_mol_string))

    if not s or s == "<None>":
        return None

    if ">>" in s:
        return get_canonical_reaction_key(s)
    else:
        return get_canonical_molecule_key(s)


# ---------------------------------------------------------------------------
# Config & task helpers
# ---------------------------------------------------------------------------

def from_dict(d):
    """dict를 attribute 접근 가능한 Struct 객체로 변환."""
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    return Struct(**d)


def get_task_subtask_info(target_benchmarks):
    """Config의 target_benchmarks 리스트를 task_subtask_dict와 pairs로 변환."""
    task_subtask_dict = {}
    for task in target_benchmarks:
        if isinstance(task, str):
            task_subtask_dict[task] = [0]
        else:
            task_subtask_dict.update(task)
    task_subtask_pairs = [
        (task, subtask)
        for task, subtasks in task_subtask_dict.items()
        for subtask in subtasks
    ]
    return task_subtask_dict, task_subtask_pairs


def get_num_workers(num_workers_arg):
    """num_workers CLI 인자를 실제 worker 수로 변환. 0이면 os.cpu_count() 사용."""
    if num_workers_arg <= 0:
        return min(os.cpu_count() or 1, 64)
    return min(num_workers_arg, 64)
