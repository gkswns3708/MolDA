"""Dual-column 생성 데이터셋 검증 (CLI 스크립트).

Usage:
    python test/validate_generated_dataset.py --data_root dataset/toy100

검증 항목:
    1. 기본 무결성: dual column 존재, null/빈값, 데이터 타입
    2. 분자 표현 일관성 (각 표현별):
       - <SMILES> 태그 안에 유효 SMILES
       - <SELFIES> 태그 안에 SELFIES 고유 토큰
    3. 그래프 유효성: node feature dim=9, edge_index 양쪽 길이 일치
    4. Task 분포: task별 샘플 수
    5. Label 형식: task 유형별 label 포맷 검증 (smiles/selfies 컬럼 각각)
    6. 프롬프트 형식: LLaDA 토큰 구조 확인 (smiles/selfies 컬럼 각각)
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import datasets
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from benchmark_constants import (
    CLASSIFICATION_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    REACTION_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    NAME_CONVERSION_BENCHMARKS,
)


# ─── helpers ─────────────────────────────────────────────────────────

SELFIES_EXCLUSIVE_RE = re.compile(r"\[(?:Branch\d?|Ring\d?|=Branch|#Branch|=Ring|Expl=Ring)")


def is_selfies(s: str) -> bool:
    """SELFIES 문자열이 정상 decode되면 True.

    Branch/Ring 토큰 regex는 false negative가 많음 ([Cr], [O-1], [O][=S][=O] 같은
    단순 분자는 Branch/Ring 없이도 valid SELFIES). selfies decoder로 실제 검증.
    """
    if not s:
        return False
    import selfies as sf
    try:
        decoded = sf.decoder(s)
    except Exception:
        return False
    if decoded is None or decoded == "":
        return False
    # decoder는 잘못된 입력에도 빈 문자열을 반환할 때가 있으므로,
    # 입력이 `[...]` 토큰 형태를 갖는지도 확인한다.
    return s.lstrip().startswith("[")


def is_valid_smiles(s: str) -> bool:
    if not s or s == "<None>":
        return True
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
        if "True" not in label and "False" not in label:
            return f"Classification label missing True/False: {label[:80]}"
    elif task in REGRESSION_BENCHMARKS:
        if "<REG>" not in label and "<|" not in label:
            return f"Regression label missing <REG> or digit tokens: {label[:80]}"
    elif task in REACTION_BENCHMARKS:
        if "<SMILES>" not in label and "<SELFIES>" not in label:
            return f"Reaction label missing mol tags: {label[:80]}"
    return None


# ─── main validation ─────────────────────────────────────────────────

DUAL_COLUMN_SUFFIXES = ("smiles", "selfies")
REQUIRED_COLS = {
    "task", "x", "edge_index", "edge_attr",
    "prompt_text_smiles", "prompt_text_selfies",
    "target_text_smiles", "target_text_selfies",
    "input_mol_string_smiles", "input_mol_string_selfies",
    "additional_x", "additional_edge_index", "additional_edge_attr",
}


def _validate_mol_string(repr_: str, ims: str, i: int, task: str, errors):
    """단일 표현 컬럼의 input_mol_string 값을 검증. (에러는 errors dict에 append)"""
    tag_type, content = extract_mol_string_content(ims)
    expected_tag = "SMILES" if repr_ == "smiles" else "SELFIES"
    if tag_type != expected_tag:
        if len(errors["tag_mismatch"]) < 10:
            errors["tag_mismatch"].append(
                f"[{i}/{repr_}] expected <{expected_tag}>, got <{tag_type}> (task={task})"
            )
        return "tag_mismatch"

    if content is None or content == "<None>":
        return "none_content"  # valid for text2mol etc.

    if repr_ == "smiles":
        # reaction-aware: each side's each molecule must be parseable
        ok = True
        for side in content.split(">>"):
            for m in side.split("."):
                m = m.strip()
                if m and not is_valid_smiles(m):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            if len(errors["mol_repr"]) < 10:
                errors["mol_repr"].append(
                    f"[{i}/smiles] invalid SMILES (task={task}): {content[:60]}"
                )
            return "invalid"
        return "valid"
    else:
        # selfies: each non-empty molecule should contain selfies-exclusive token
        ok = True
        for side in content.split(">>"):
            for m in side.split("."):
                m = m.strip()
                if m and not is_selfies(m):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            if len(errors["mol_repr"]) < 10:
                errors["mol_repr"].append(
                    f"[{i}/selfies] non-SELFIES content (task={task}): {content[:60]}"
                )
            return "invalid"
        return "valid"


def validate_split(ds, split_name: str, sample_limit: int = 5000):
    """단일 split 검증. 결과 dict 반환."""
    errors = defaultdict(list)
    task_counts = Counter()
    per_repr_counts = {"smiles": Counter(), "selfies": Counter()}
    total = len(ds)
    check_n = min(total, sample_limit)

    print(f"\n{'='*60}")
    print(f"  Validating {split_name.upper()} split: {total:,} examples (checking {check_n:,})")
    print(f"{'='*60}")

    # 1. 컬럼 확인
    missing_cols = REQUIRED_COLS - set(ds.column_names)
    if missing_cols:
        errors["schema"].append(f"Missing columns: {missing_cols}")
        print(f"  [FAIL] Missing columns: {missing_cols}")
        return {"errors": dict(errors), "total": total, "checked": 0, "task_counts": {}}

    if "mol_representation" in ds.column_names:
        errors["schema"].append(
            "Legacy single 'mol_representation' column should not exist in dual-column schema"
        )

    print(f"  [OK] All required columns present: {sorted(ds.column_names)}")

    label_format_errors = 0
    graph_errors = 0
    prompt_format_errors = 0

    for i in range(check_n):
        row = ds[i]
        task = row["task"]
        task_counts[task] += 1

        for repr_ in DUAL_COLUMN_SUFFIXES:
            # 2a. input_mol_string 검증
            status = _validate_mol_string(
                repr_, row[f"input_mol_string_{repr_}"], i, task, errors
            )
            per_repr_counts[repr_][status] += 1

            # 2b. Label 형식 검증 (per-representation)
            label_err = validate_label_format(row[f"target_text_{repr_}"], task)
            if label_err:
                label_format_errors += 1
                if len(errors["label"]) < 10:
                    errors["label"].append(f"[{i}/{repr_}] {label_err}")

            # 2d. 프롬프트 형식 (per-representation)
            prompt = row[f"prompt_text_{repr_}"]
            if "<|startoftext|>" in prompt:
                if "<|start_header_id|>system<|end_header_id|>" not in prompt:
                    prompt_format_errors += 1
                    if len(errors["prompt"]) < 10:
                        errors["prompt"].append(f"[{i}/{repr_}] Missing system header")
                if "<GRAPH>" not in prompt:
                    prompt_format_errors += 1
                    if len(errors["prompt"]) < 10:
                        errors["prompt"].append(f"[{i}/{repr_}] Missing <GRAPH>")

        # 2c. 그래프 유효성 (표현 무관하게 공유)
        x = row["x"]
        edge_index = row["edge_index"]
        if x and len(x) > 0:
            node_feat_dim = len(x[0]) if x[0] else 0
            if node_feat_dim != 9:
                graph_errors += 1
                if len(errors["graph"]) < 10:
                    errors["graph"].append(
                        f"[{i}] node_feat dim={node_feat_dim}, expected 9 (task={task})"
                    )
            if edge_index and len(edge_index) == 2:
                if len(edge_index[0]) != len(edge_index[1]):
                    errors["graph"].append(
                        f"[{i}] edge_index mismatch: "
                        f"src={len(edge_index[0])}, dst={len(edge_index[1])}"
                    )

    # 3. 결과 출력
    print(f"\n  --- Molecule Representation (smiles/selfies) ---")
    for repr_ in DUAL_COLUMN_SUFFIXES:
        c = per_repr_counts[repr_]
        print(f"  [{repr_}] valid={c['valid']:,}  "
              f"none_content={c['none_content']:,}  "
              f"invalid={c['invalid']:,}  "
              f"tag_mismatch={c['tag_mismatch']:,}")

    print(f"\n  --- Label / Graph / Prompt ---")
    print(f"  Label format errors:  {label_format_errors:,}")
    print(f"  Graph errors:         {graph_errors:,}")
    print(f"  Prompt format errors: {prompt_format_errors:,}")

    print(f"\n  --- Task Distribution ---")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"    {task:45s} {count:>7,}")

    all_errors = {k: v for k, v in errors.items() if v}
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
        "per_repr_counts": {k: dict(v) for k, v in per_repr_counts.items()},
        "stats": {
            "label_errors": label_format_errors,
            "graph_errors": graph_errors,
            "prompt_errors": prompt_format_errors,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Validate dual-column generated dataset")
    parser.add_argument("--data_root", type=str, default="dataset/toy100",
                        help="Path to dataset root (contains Train/Val/Test).")
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

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for split_name, result in results.items():
        n_err = sum(len(v) for v in result["errors"].values())
        status = "PASS" if n_err == 0 else "FAIL"
        print(f"  {split_name:12s}: {result['total']:>10,} examples | "
              f"checked {result['checked']:>7,} | [{status}] {n_err} error(s)")

    if total_errors > 0:
        print(f"\n  Total errors: {total_errors}")
        sys.exit(1)
    else:
        print(f"\n  All validations passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
