#!/usr/bin/env python3
"""Per-task stratified 1/100 Test subset for fast re-evaluation.

정책:
  - split: Test (fixed — 논문 eval 기준)
  - ratio: 1/100
  - per-task: 각 task에서 max(10, round(N_task / 100)) 개를 random pick
  - seed: 42 (재현성)
  - 18 task 전부 보존 (raw_v1_10x_rephrase/Test 기준 18개)
  - 저장: dataset/Processed/{output_tag}/Test (HF Arrow)

Usage:
  venvs/MolDA/bin/python scripts/make_test_subset.py \
      --input-tag raw_v1_10x_rephrase \
      --output-tag raw_v1_10x_rephrase_testsub100 \
      --ratio 100 \
      --min-per-task 10 \
      --seed 42
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk


def stratified_subset(ds: Dataset, ratio: int, min_per_task: int, seed: int) -> Dataset:
    tasks = ds["task"]
    counts = Counter(tasks)
    rng = np.random.default_rng(seed)

    # Map task → list of indices
    idx_by_task: dict[str, list[int]] = {}
    for i, t in enumerate(tasks):
        idx_by_task.setdefault(t, []).append(i)

    keep_idx: list[int] = []
    for task, total in sorted(counts.items()):
        n_target = max(min_per_task, round(total / ratio))
        n_target = min(n_target, total)
        # seed의 일관성 보장: task 별로 결정적 선택
        pool = np.asarray(idx_by_task[task])
        picked = rng.choice(pool, size=n_target, replace=False)
        keep_idx.extend(sorted(picked.tolist()))

    keep_idx.sort()
    return ds.select(keep_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-tag", default="raw_v1_10x_rephrase")
    ap.add_argument("--output-tag", default=None,
                    help="default: {input-tag}_testsub100")
    ap.add_argument("--split", default="Test",
                    help="split name (Test / Val). default: Test")
    ap.add_argument("--ratio", type=int, default=100,
                    help="1/ratio of each task. default: 100")
    ap.add_argument("--min-per-task", type=int, default=10,
                    help="min samples per task. default: 10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default="dataset/Processed")
    args = ap.parse_args()

    input_tag = args.input_tag
    output_tag = args.output_tag or f"{input_tag}_testsub100"

    input_path = Path(args.root) / input_tag / args.split
    output_path = Path(args.root) / output_tag / args.split

    if not input_path.exists():
        raise SystemExit(f"ERROR: input path not found: {input_path}")

    print(f"[load] {input_path}", flush=True)
    ds = load_from_disk(str(input_path))
    before_counts = Counter(ds["task"])
    print(f"  n_rows: {len(ds):,}  n_tasks: {len(before_counts)}", flush=True)

    subset = stratified_subset(ds, args.ratio, args.min_per_task, args.seed)
    after_counts = Counter(subset["task"])

    print(f"\n[subset] ratio=1/{args.ratio}, min_per_task={args.min_per_task}, seed={args.seed}")
    print(f"{'Task':<55} {'Total':>8} {'Picked':>8}  {'Picked/Total':>12}")
    print("-" * 90)
    for task in sorted(before_counts):
        tot = before_counts[task]
        pick = after_counts.get(task, 0)
        pct = 100.0 * pick / tot if tot else 0
        print(f"{task:<55} {tot:>8} {pick:>8}  {pct:>11.2f}%")
    print("-" * 90)
    print(f"{'TOTAL':<55} {len(ds):>8} {len(subset):>8}  {100.0*len(subset)/len(ds):>11.2f}%")

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(output_path))
    print(f"\n[save] {output_path}", flush=True)

    # Test 외 split도 hard-link (재평가용 train/val은 쓰지 않지만 data config 호환성 위해)
    for other in ("Train", "Val"):
        src = Path(args.root) / input_tag / other
        dst = Path(args.root) / output_tag / other
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
            print(f"[link] {dst} → {src}", flush=True)


if __name__ == "__main__":
    main()
