#!/usr/bin/env python3
"""Data augmentation: instruction-rephrase 10x for 4 target tasks.

Step 3 산출물(`dataset/Processed/{tag}/{split}`)을 입력으로 받아:
  1. 대상 4 task (chebi-20-mol2text, chebi-20-text2mol,
     smol-molecule_captioning, smol-molecule_generation)를 필터
  2. 각 row를 10회 복제, 복제마다 `instructions_smol.py`의 해당 pool에서
     새 instruction template을 random pick하여 prompt_text 재생성
  3. 나머지 원본 task(17개) + augmented 4 task를 concat해 저장

Tag suffix: `_10x_rephrase` — 원본 dir은 건드리지 않음.

Usage:
  python -m dataset_generation.data_aug \
      --input-tag raw_v1 \
      --split Train \
      --factor 10 \
      --seed 42

Outputs → `dataset/Processed/raw_v1_10x_rephrase/Train`

원본 분석 (generator.py 기준):
  - mol2text (chebi-20-mol2text, smol-molecule_captioning):
      instruction 에 `<INPUT>` 보존 상태로 stored. prepare_data_instance가
      `<INPUT>`을 input_mol_string + graph_sequence로 치환.
  - text2mol (chebi-20-text2mol, smol-molecule_generation):
      Step 1에서 instruction.replace("<INPUT>", "<DESCRIPTION>...</DESCRIPTION>")
      적용되어 stored. instruction 에 `<INPUT>` 없음, description embedded.
      Aug 시 기존 instruction 에서 description을 추출해 새 template에 재삽입.

기존 prompt_text_{smiles,selfies}의 header/system 구조는 완전 보존하고
user section 내용만 새 instruction 기반으로 교체.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk

# instructions_smol을 cwd/PYTHONPATH에 의존하지 않고 절대 경로로 직접 로드.
# (sys.path 조작은 실행 컨텍스트에 따라 flaky — importlib.util로 robust하게.)
_INSTRUCTIONS_SMOL_PATH = Path(__file__).resolve().parent / "instructions_smol.py"
_spec = importlib.util.spec_from_file_location("instructions_smol", _INSTRUCTIONS_SMOL_PATH)
instructions_smol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(instructions_smol)

DEFAULT_TARGET_TASKS = (
    "chebi-20-mol2text",
    "chebi-20-text2mol",
    "smol-molecule_captioning",
    "smol-molecule_generation",
)

# 4 task → instructions_smol 내 template pool 이름 매핑.
TASK_TEMPLATES = {
    "chebi-20-mol2text":        "molecule_captioning",
    "smol-molecule_captioning": "molecule_captioning",
    "chebi-20-text2mol":        "molecule_generation",
    "smol-molecule_generation": "molecule_generation",
}

# text2mol 계열: Step 1에서 이미 description이 치환됨
TEXT2MOL_TASKS = {"chebi-20-text2mol", "smol-molecule_generation"}
# mol2text 계열: instruction에 <INPUT> 보존
MOL2TEXT_TASKS = {"chebi-20-mol2text", "smol-molecule_captioning"}

# user section 내용 추출/교체용 (LLaDA/LLaMA-3 포맷)
USER_SECTION_RE = re.compile(
    r"(<\|start_header_id\|>user<\|end_header_id\|>\n\n)(.*?)(<\|eot_id\|>)",
    re.DOTALL,
)

# text2mol instruction 안에 embedded된 description 추출
DESCRIPTION_RE = re.compile(r"<DESCRIPTION>.*?</DESCRIPTION>", re.DOTALL)

# user section 안에서 graph_sequence 추출 (prepare_data_instance와 동일 포맷)
GRAPH_SEQ_RE = re.compile(r"<GRAPH>(?:<mol>)+</GRAPH>")


def _get_pool(task: str) -> list[str]:
    name = TASK_TEMPLATES[task]
    return list(getattr(instructions_smol, name))


def _build_new_instruction(
    row: dict, new_template: str
) -> Optional[str]:
    """task별 로직에 따라 새 instruction 문자열 생성.

    - mol2text: template 그대로 (<INPUT> 보존). prompt 재구성 시 치환됨.
    - text2mol: 기존 instruction에서 <DESCRIPTION>...</DESCRIPTION>을 추출해
                새 template의 <INPUT> 자리에 삽입.
    """
    task = row["task"]
    if task in MOL2TEXT_TASKS:
        return new_template
    # text2mol
    old_instr = row.get("instruction", "") or ""
    m = DESCRIPTION_RE.search(old_instr)
    if not m:
        return None  # description 추출 실패 → row skip
    return new_template.replace("<INPUT>", m.group(0))


def _rebuild_prompt_text(
    old_prompt: str, new_instruction: str, input_mol_string: str
) -> str:
    """기존 prompt_text의 header/system/assistant 구조는 그대로 두고
    user section 내용만 새 instruction 기반으로 교체.

    graph_sequence는 기존 user section에서 추출해 재사용.
    """
    user_m = USER_SECTION_RE.search(old_prompt)
    if not user_m:
        # 예상 밖 포맷이면 원본 prompt 유지 (안전)
        return old_prompt

    header, old_content, footer = user_m.group(1), user_m.group(2), user_m.group(3)

    # graph_sequence 추출
    g = GRAPH_SEQ_RE.search(old_content)
    graph_seq = g.group(0) if g else "<GRAPH></GRAPH>"

    if "<INPUT>" in new_instruction:
        # mol2text: input_mol_string + graph_sequence를 <INPUT> 자리에
        new_user = new_instruction.replace(
            "<INPUT>", input_mol_string + graph_seq
        )
    else:
        # text2mol: description이 embedded, graph_sequence만 append
        new_user = new_instruction + graph_seq

    return old_prompt[: user_m.start()] + header + new_user + footer + old_prompt[user_m.end():]


def augment_row(row: dict, rng: np.random.Generator) -> Optional[dict]:
    """단일 row를 새 instruction으로 재구성하여 반환.

    반환된 dict는 원본과 같은 key set. 실패 시 None.
    """
    task = row["task"]
    if task not in TASK_TEMPLATES:
        return row  # 대상 아니면 그대로 (safety)

    pool = _get_pool(task)
    new_template = str(rng.choice(pool))
    new_instruction = _build_new_instruction(row, new_template)
    if new_instruction is None:
        return None

    new_row = dict(row)
    new_row["instruction"] = new_instruction
    new_row["prompt_text_smiles"] = _rebuild_prompt_text(
        row["prompt_text_smiles"], new_instruction, row["input_mol_string_smiles"]
    )
    new_row["prompt_text_selfies"] = _rebuild_prompt_text(
        row["prompt_text_selfies"], new_instruction, row["input_mol_string_selfies"]
    )
    return new_row


def augment_dataset(
    ds: Dataset,
    target_tasks: tuple[str, ...],
    factor: int,
    seed: int,
) -> Dataset:
    """대상 task만 factor× 복제 + instruction 재생성, 나머지는 그대로 붙임."""
    targets_mask = [t in target_tasks for t in ds["task"]]
    target_idx = [i for i, m in enumerate(targets_mask) if m]
    other_idx = [i for i, m in enumerate(targets_mask) if not m]

    print(
        f"  target-task rows: {len(target_idx)}  /  other-task rows: {len(other_idx)}",
        flush=True,
    )

    if not target_idx:
        print("  [WARN] no target-task rows found — nothing to augment")
        return ds

    target_ds = ds.select(target_idx)
    other_ds = ds.select(other_idx)

    # factor개 copy 생성, 각 copy는 다른 seed로 instruction 재생성
    copies = []
    for k in range(factor):
        sub_seed = seed + k
        rng = np.random.default_rng(sub_seed)

        # HF Dataset.map은 row 단위 처리. 외부 rng 공유를 위해 클로저.
        def _apply(example):
            out = augment_row(example, rng)
            return out if out is not None else example

        c = target_ds.map(
            _apply,
            desc=f"augment copy {k+1}/{factor}",
            load_from_cache_file=False,
        )
        copies.append(c)

    augmented = concatenate_datasets(copies)
    print(
        f"  augmented {len(target_ds)} rows × {factor} = {len(augmented)} rows",
        flush=True,
    )

    # 순서: other + augmented (train shuffling은 dataloader에서 담당)
    return concatenate_datasets([other_ds, augmented])


def main():
    ap = argparse.ArgumentParser(description="MolDA instruction-rephrase augmentation")
    ap.add_argument("--input-tag", required=True,
                    help="e.g. 'raw_v1' — reads dataset/Processed/{input_tag}/{split}")
    ap.add_argument("--output-tag", default=None,
                    help="default: {input_tag}_10x_rephrase")
    ap.add_argument("--split", default="Train",
                    help="split name (Train/Val/Test). default: Train")
    ap.add_argument("--factor", type=int, default=10,
                    help="replication factor (5x sampled × 2x rephrase ≈ 10). default: 10")
    ap.add_argument("--target-tasks", nargs="+", default=list(DEFAULT_TARGET_TASKS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default="dataset/Processed",
                    help="root dir containing {tag}/{split}. default: dataset/Processed")
    args = ap.parse_args()

    input_tag = args.input_tag
    output_tag = args.output_tag or f"{input_tag}_10x_rephrase"

    input_path = Path(args.root) / input_tag / args.split
    output_path = Path(args.root) / output_tag / args.split

    print("=" * 70)
    print(f"  data_aug: instruction-rephrase 10x")
    print(f"  input  : {input_path}")
    print(f"  output : {output_path}")
    print(f"  target : {args.target_tasks}")
    print(f"  factor : {args.factor}   seed: {args.seed}")
    print("=" * 70)

    if not input_path.exists():
        raise SystemExit(f"ERROR: input path not found: {input_path}")

    t0 = time.time()
    ds = load_from_disk(str(input_path))
    print(f"loaded input: {len(ds)} rows, tasks={len(set(ds['task']))}")

    augmented = augment_dataset(
        ds,
        target_tasks=tuple(args.target_tasks),
        factor=args.factor,
        seed=args.seed,
    )

    print(f"saving → {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    augmented.save_to_disk(str(output_path))

    print("=" * 70)
    print(f"  DONE in {(time.time()-t0)/60:.2f} min")
    print(f"  input  rows: {len(ds):,}")
    print(f"  output rows: {len(augmented):,}")
    from collections import Counter
    in_cnt = Counter(ds["task"])
    out_cnt = Counter(augmented["task"])
    print("  per-task delta (target tasks only):")
    for t in args.target_tasks:
        print(f"    {t:40s}  {in_cnt.get(t, 0):>7,} → {out_cnt.get(t, 0):>7,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
