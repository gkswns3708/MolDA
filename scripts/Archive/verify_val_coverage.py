#!/usr/bin/env python3
"""
Verify whether the observed per-task validation coverage loss is fully
explained by DDP DistributedSampler + drop_last=True, or if additional
drops happen elsewhere.

Usage:
  # 예측 JSON에서 자동으로 observed loss 산출 + DDP 시뮬레이션과 비교
  python scripts/verify_val_coverage.py \
      --val-dataset dataset/Processed/toy100/Val \
      --predictions lightning_logs/version_N/val_predictions/predictions_epoch0_step21.json \
      --world-size 6 \
      --batch-size 22

  # 수동 observed 입력
  python scripts/verify_val_coverage.py \
      --val-dataset dataset/Processed/toy100/Val \
      --observed "smol-retrosynthesis=60,smol-property_prediction-sider=2"
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk


def simulate_ddp_drops(total: int, world_size: int, batch_size: int):
    """Lightning의 DistributedSampler(shuffle=False, drop_last=False) +
    DataLoader(drop_last=True)의 동등 동작을 재현하여 "어느 원본 index가
    단 한 번도 처리되지 않았는가"를 집합으로 반환.

    DistributedSampler는 dataset을 world_size의 배수로 pad한 후 indices를
    rank별로 stripe (indices[rank::world_size]). padding된 index는 원본을
    wrap-around하여 채우므로 단순 drop은 아니고 duplicate일 수 있다.

    DataLoader(drop_last=True, batch_size=B)는 각 rank의 indices에서 마지막
    partial batch (n mod B) 만큼을 떨어뜨린다. 떨어뜨려진 idx가 다른 rank에
서 커버되면 unique coverage 관점에선 loss 아님. 반대로 어떤 unique 원본
    sample이 모든 rank에서 drop되거나 아예 포함되지 않으면 unique miss.
    """
    pad = (world_size - total % world_size) % world_size
    padded_total = total + pad

    # rank별 indices (padded 범위의 stripe)
    per_rank_indices = {
        r: list(range(r, padded_total, world_size))
        for r in range(world_size)
    }

    # 각 rank에서 drop되는 (padded) indices
    dropped_per_rank = {}
    for r, idxs in per_rank_indices.items():
        n = len(idxs)
        keep = (n // batch_size) * batch_size
        dropped_per_rank[r] = idxs[keep:]

    # unique 원본 sample이 처리되었는지 여부
    processed = set()
    for r, idxs in per_rank_indices.items():
        dropped = set(dropped_per_rank[r])
        for padded_idx in idxs:
            if padded_idx in dropped:
                continue
            orig = padded_idx % total  # pad wrap-around
            processed.add(orig)

    all_idx = set(range(total))
    missed = all_idx - processed
    return missed, dropped_per_rank


def extract_observed_from_predictions(preds_path: Path, val_ds, expected_per_task):
    """predictions JSON에서 per-task unique covered를 계산하여 loss dict 반환."""
    with open(preds_path) as f:
        data = json.load(f)

    cls_per_task = Counter(r["task"] for r in data.get("classification", []))

    # generation: (task, strategy) → count. unique per task = max over strategies.
    gen_records = Counter(
        (r["task"], r.get("strategy", "?"))
        for r in data.get("generation", [])
    )
    gen_unique_per_task = {}
    for (task, _), cnt in gen_records.items():
        gen_unique_per_task[task] = max(gen_unique_per_task.get(task, 0), cnt)

    covered = {}
    for t in expected_per_task:
        covered[t] = cls_per_task.get(t, 0) + gen_unique_per_task.get(t, 0)

    loss = {t: expected_per_task[t] - covered[t] for t in expected_per_task
            if expected_per_task[t] - covered[t] > 0}
    return covered, loss


def parse_kv_string(s: str) -> dict:
    """'task=count,task=count' → dict."""
    out = {}
    if not s:
        return out
    for part in s.split(","):
        k, v = part.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dataset", required=True, type=Path)
    ap.add_argument("--world-size", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=22)
    ap.add_argument("--predictions", type=Path, default=None,
                    help="predictions_epoch0_stepN.json — from this, observed loss is auto-computed")
    ap.add_argument("--observed", type=str, default="",
                    help="manual override: task=count,task=count,...")
    args = ap.parse_args()

    ds = load_from_disk(str(args.val_dataset))
    tasks = ds["task"]
    total = len(tasks)
    expected_per_task = Counter(tasks)

    print(f"Val dataset:         {total} samples")
    print(f"Unique tasks:        {len(expected_per_task)}")
    print(f"DDP world_size:      {args.world_size}")
    print(f"val batch_size:      {args.batch_size}")
    print(f"First 5 task labels: {tasks[:5]}")
    print(f"Last  5 task labels: {tasks[-5:]}")

    # task 순서가 contiguous block인지 확인
    def is_contiguous_blocks(seq):
        seen, last = set(), None
        for t in seq:
            if t != last:
                if t in seen:
                    return False
                seen.add(t)
                last = t
        return True

    if is_contiguous_blocks(tasks):
        print("Task ordering:       contiguous blocks (per-task chunks)")
    else:
        print("Task ordering:       INTERLEAVED (not contiguous)")

    # DDP + drop_last 시뮬레이션
    missed, dropped_per_rank = simulate_ddp_drops(
        total=total,
        world_size=args.world_size,
        batch_size=args.batch_size,
    )

    print(f"\nPredicted unique samples missed: {len(missed)}")
    print("Dropped count per rank (pre-unique): "
          f"{ {r: len(d) for r, d in dropped_per_rank.items()} }")

    predicted_loss = Counter(tasks[i] for i in missed)
    print("\n── Predicted loss per task (DDP+drop_last simulation) ──")
    for t, c in sorted(predicted_loss.items(), key=lambda x: -x[1]):
        print(f"  {t:40s}  {c}")

    # Observed loss source
    observed = {}
    if args.predictions:
        covered, observed = extract_observed_from_predictions(
            args.predictions, ds, expected_per_task
        )
        print(f"\nObserved from predictions JSON: {args.predictions}")
    if args.observed:
        manual = parse_kv_string(args.observed)
        # 수동 입력이 있으면 덮어씀
        observed.update(manual)
        print(f"\nObserved (manual override): {manual}")

    if not observed:
        print("\n[INFO] no --predictions or --observed given; printing predicted only.")
        return

    print("\n── Observed loss per task ──")
    for t, c in sorted(observed.items(), key=lambda x: -x[1]):
        print(f"  {t:40s}  {c}")

    # Compare
    print("\n── Compare (predicted vs observed) ──")
    all_tasks = set(predicted_loss) | set(observed)
    diff = False
    for t in sorted(all_tasks):
        p = predicted_loss.get(t, 0)
        o = observed.get(t, 0)
        tag = "OK" if p == o else "DIFF"
        if p != o:
            diff = True
        print(f"  {t:40s}  predicted={p:4d}  observed={o:4d}  [{tag}]")

    total_pred = sum(predicted_loss.values())
    total_obs = sum(observed.values())
    print(f"\n  TOTAL                                     "
          f"predicted={total_pred:4d}  observed={total_obs:4d}  "
          f"[{'OK' if total_pred == total_obs else 'DIFF'}]")

    print()
    if not diff:
        print("✅ CONCLUSION: Observed loss is *fully explained* by DDP + "
              "drop_last=True on val_dataloader.")
        print("   → No additional bug path. Behavior is expected for the "
              "current DDP config.")
        print("   → To get 100% coverage, set val_dataloader drop_last=False "
              "or run on 1 GPU.")
    else:
        print("⚠️  CONCLUSION: Mismatch. Some samples are dropped beyond "
              "DDP + drop_last.")
        print("   → Investigate: validation.py task filters (cls_idx/gen_idx), "
              "collator skip paths, or generate() failures.")
        print("   → Difference indicates bugs/paths NOT caused by DDP alone.")


if __name__ == "__main__":
    main()
