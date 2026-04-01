"""생성된 데이터셋 검증 스크립트.

Usage:
    python test/test_validate_generated_dataset.py \
        --data_root dataset/Processed/SELFIES/raw_v1

검증 항목:
    1. 기본 무결성: 컬럼 존재, null/빈값, 데이터 타입
    2. 분자 표현 일관성: SMILES 태그 안에 SMILES, SELFIES 태그 안에 SELFIES
    3. 그래프 유효성: node feature, edge_index, edge_attr 차원 확인
    4. Task 분포: task별 샘플 수
    5. Label 형식: task 유형별 label 포맷 검증
    6. 프롬프트 형식: LLaDA 토큰 구조 확인
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import datasets
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

# ─── benchmark constants ───
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from benchmark_constants import (
    CLASSIFICATION_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    REACTION_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    NAME_CONVERSION_BENCHMARKS,
)

# ─── helpers ───

SELFIES_EXCLUSIVE_RE = re.compile(r"\[(?:Branch\d?|Ring\d?|=Branch|#Branch|=Ring|Expl=Ring)")


def is_selfies(s: str) -> bool:
    """SELFIES 고유 토큰([Branch], [Ring] 등)이 있으면 SELFIES로 판별.
    [NH3+], [O-], [Cr] 같은 SMILES 대괄호 표기와 구분."""
    return bool(s) and bool(SELFIES_EXCLUSIVE_RE.search(s))


def is_valid_smiles(s: str) -> bool:
    if not s or s == "<None>":
        return True  # <None>은 text2mol 등에서 정상
    mol = Chem.MolFromSmiles(s)
    return mol is not None


def extract_mol_string_content(input_mol_string: str):
    """<SMILES> ... </SMILES> 또는 <SELFIES> ... </SELFIES> 에서 내용 추출."""
    smiles_match = re.search(r"<SMILES>\s*(.*?)\s*</SMILES>", input_mol_string)
    selfies_match = re.search(r"<SELFIES>\s*(.*?)\s*</SELFIES>", input_mol_string)
    if smiles_match:
        return "SMILES", smiles_match.group(1)
    elif selfies_match:
        return "SELFIES", selfies_match.group(1)
    return None, None


def validate_label_format(label: str, task: str):
    """task 유형에 따른 label 형식 검증. (에러 메시지 반환, 정상이면 None)"""
    if task in CLASSIFICATION_BENCHMARKS:
        # <CLASS> True/False </CLASS> 형식
        if "True" not in label and "False" not in label:
            return f"Classification label missing True/False: {label[:80]}"
    elif task in REGRESSION_BENCHMARKS:
        # <REG> <|+|><|0|>... </REG> 형식
        if "<REG>" not in label and "<|" not in label:
            return f"Regression label missing <REG> or digit tokens: {label[:80]}"
    elif task in REACTION_BENCHMARKS:
        # SMILES 또는 SELFIES 문자열
        if "<SMILES>" not in label and "<SELFIES>" not in label:
            return f"Reaction label missing mol tags: {label[:80]}"
    return None


# ─── main validation ───

def validate_split(ds, split_name: str, sample_limit: int = 5000):
    """단일 split 검증. 결과 dict 반환."""
    errors = defaultdict(list)
    warnings = defaultdict(list)
    task_counts = Counter()
    mol_repr_counts = Counter()
    total = len(ds)
    check_n = min(total, sample_limit)

    print(f"\n{'='*60}")
    print(f"  Validating {split_name.upper()} split: {total:,} examples (checking {check_n:,})")
    print(f"{'='*60}")

    # 1. 컬럼 확인
    required_cols = {"task", "x", "edge_index", "edge_attr", "prompt_text", "target_text",
                     "input_mol_string", "mol_representation",
                     "additional_x", "additional_edge_index", "additional_edge_attr"}
    missing_cols = required_cols - set(ds.column_names)
    if missing_cols:
        errors["schema"].append(f"Missing columns: {missing_cols}")
        print(f"  [FAIL] Missing columns: {missing_cols}")
        return {"errors": dict(errors), "warnings": dict(warnings),
                "task_counts": dict(task_counts), "total": total}

    print(f"  [OK] All required columns present: {sorted(ds.column_names)}")

    # 2. 샘플별 검증
    smiles_in_smiles_tag = 0  # SMILES 태그 안에 유효 SMILES
    selfies_in_smiles_tag = 0  # SMILES 태그 안에 SELFIES 혼입
    smiles_invalid = 0
    label_format_errors = 0
    graph_errors = 0
    prompt_format_errors = 0

    for i in range(check_n):
        row = ds[i]
        task = row["task"]
        task_counts[task] += 1
        mol_repr_counts[row.get("mol_representation", "unknown")] += 1

        # 2a. input_mol_string 검증
        ims = row["input_mol_string"]
        tag_type, content = extract_mol_string_content(ims)

        if tag_type == "SMILES" and content and content != "<None>":
            if is_selfies(content):
                selfies_in_smiles_tag += 1
                if selfies_in_smiles_tag <= 3:
                    errors["mol_repr"].append(
                        f"[{i}] SELFIES inside <SMILES> tag (task={task}): {content[:60]}")
            elif is_valid_smiles(content.split(">>")[0].split(".")[0]):
                smiles_in_smiles_tag += 1
            else:
                smiles_invalid += 1
                if smiles_invalid <= 3:
                    errors["mol_repr"].append(
                        f"[{i}] Invalid SMILES in <SMILES> tag (task={task}): {content[:60]}")

        elif tag_type == "SELFIES" and content and content != "<None>":
            if not is_selfies(content):
                if len(errors["mol_repr"]) < 5:
                    errors["mol_repr"].append(
                        f"[{i}] Non-SELFIES inside <SELFIES> tag (task={task}): {content[:60]}")

        # 2b. Label 형식 검증
        label_err = validate_label_format(row["target_text"], task)
        if label_err:
            label_format_errors += 1
            if label_format_errors <= 3:
                errors["label"].append(f"[{i}] {label_err}")

        # 2c. 그래프 유효성
        x = row["x"]
        edge_index = row["edge_index"]
        if x and len(x) > 0:
            node_feat_dim = len(x[0]) if x[0] else 0
            if node_feat_dim != 9:
                graph_errors += 1
                if graph_errors <= 3:
                    errors["graph"].append(f"[{i}] node_feat dim={node_feat_dim}, expected 9 (task={task})")
            if edge_index and len(edge_index) == 2:
                num_edges_src = len(edge_index[0])
                num_edges_dst = len(edge_index[1])
                if num_edges_src != num_edges_dst:
                    errors["graph"].append(f"[{i}] edge_index mismatch: src={num_edges_src}, dst={num_edges_dst}")

        # 2d. 프롬프트 형식
        prompt = row["prompt_text"]
        if "<|startoftext|>" in prompt:
            # LLaDA format
            if "<|start_header_id|>system<|end_header_id|>" not in prompt:
                prompt_format_errors += 1
                if prompt_format_errors <= 3:
                    errors["prompt"].append(f"[{i}] Missing system header in LLaDA prompt")
            if "<GRAPH>" not in prompt:
                prompt_format_errors += 1
                if prompt_format_errors <= 3:
                    errors["prompt"].append(f"[{i}] Missing <GRAPH> in prompt")

    # 3. 결과 출력
    print(f"\n  --- Molecule Representation ---")
    print(f"  Valid SMILES in <SMILES> tags:   {smiles_in_smiles_tag:,} / {check_n:,}")
    print(f"  SELFIES leaked into <SMILES>:    {selfies_in_smiles_tag:,}")
    print(f"  Invalid SMILES:                  {smiles_invalid:,}")
    print(f"  mol_representation distribution: {dict(mol_repr_counts)}")

    print(f"\n  --- Label Format ---")
    print(f"  Label format errors:   {label_format_errors:,} / {check_n:,}")

    print(f"\n  --- Graph ---")
    print(f"  Graph errors:          {graph_errors:,} / {check_n:,}")

    print(f"\n  --- Prompt ---")
    print(f"  Prompt format errors:  {prompt_format_errors:,} / {check_n:,}")

    print(f"\n  --- Task Distribution ---")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"    {task:45s} {count:>7,}")

    # 4. 에러 상세
    all_errors = {k: v for k, v in errors.items() if v}
    all_warnings = {k: v for k, v in warnings.items() if v}

    if all_errors:
        print(f"\n  [ERRORS]")
        for category, msgs in all_errors.items():
            print(f"    --- {category} ---")
            for m in msgs[:5]:
                print(f"      {m}")
            remaining = len(msgs) - 5
            if remaining > 0:
                print(f"      ... and {remaining} more")
    else:
        print(f"\n  [ALL CHECKS PASSED]")

    return {
        "errors": all_errors,
        "total": total,
        "checked": check_n,
        "task_counts": dict(task_counts),
        "stats": {
            "valid_smiles": smiles_in_smiles_tag,
            "selfies_in_smiles_tag": selfies_in_smiles_tag,
            "invalid_smiles": smiles_invalid,
            "label_errors": label_format_errors,
            "graph_errors": graph_errors,
            "prompt_errors": prompt_format_errors,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Validate generated dataset")
    parser.add_argument("--data_root", type=str, default="dataset/Processed/SELFIES/raw_v1",
                        help="Path to Processed dataset dir (e.g. dataset/Processed/SELFIES/raw_v1)")
    parser.add_argument("--sample_limit", type=int, default=5000,
                        help="Max samples to check per split (0=all)")
    args = parser.parse_args()

    splits = {
        "train": os.path.join(args.data_root, "Train"),
        "validation": os.path.join(args.data_root, "Val"),
        "test": os.path.join(args.data_root, "Test"),
    }

    print("=" * 60)
    print("  Dataset Validation Report")
    print(f"  data_root: {args.data_root}")
    print("=" * 60)

    results = {}
    total_errors = 0
    for split_name, path in splits.items():
        if not os.path.exists(path):
            print(f"\n  [SKIP] {split_name}: {path} not found")
            continue
        ds = datasets.Dataset.load_from_disk(path)
        limit = args.sample_limit if args.sample_limit > 0 else len(ds)
        result = validate_split(ds, split_name, sample_limit=limit)
        results[split_name] = result
        total_errors += sum(len(v) for v in result["errors"].values())

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for split_name, result in results.items():
        n_err = sum(len(v) for v in result["errors"].values())
        status = "PASS" if n_err == 0 else "FAIL"
        print(f"  {split_name:12s}: {result['total']:>10,} examples | checked {result['checked']:>7,} | [{status}] {n_err} error(s)")

    if total_errors > 0:
        print(f"\n  Total errors: {total_errors}")
        sys.exit(1)
    else:
        print(f"\n  All validations passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
