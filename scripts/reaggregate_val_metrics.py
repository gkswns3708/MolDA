#!/usr/bin/env python3
"""Re-aggregate per-task × strategy metrics from a saved predictions JSON.

입력: lightning_logs/version_N/val_predictions/predictions_epoch{E}_step{S}.json
출력: JSON + Markdown summary (per-task × strategy validity/bleu/rouge/exact-match 등)

64 CPU 병렬 (multiprocessing.Pool) + 각 task×strategy 단위 작업 분배.
tokenizer는 metrics.py의 BLEU/METEOR에서 선택적이므로 wordnet 모드만 사용 (tokenizer=None).

Usage:
  cd /opt/EMNLP_MolDA/New_MolDA
  venvs/MolDA/bin/python scripts/reaggregate_val_metrics.py \
      --input lightning_logs/version_21/val_predictions/predictions_epoch1_step8526.json \
      --output-json docs/analysis/metrics_v21_epoch1.json \
      --output-md   docs/analysis/metrics_v21_epoch1.md \
      --num-workers 32
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from src.training.metrics import (  # noqa: E402
    CLASSIFICATION_TASKS,
    NAME_CONVERSION_TASKS,
    get_task_type,
    classification_evaluate,
    regression_evaluate,
    molecule_evaluate,
    caption_evaluate,
)


def _eval_bucket(args):
    """Worker: (task, strategy, records) → metric dict."""
    task, strategy, records = args
    task_type = get_task_type(task)
    preds = [r["pred_text"] for r in records]
    labels = [r["label_text"] for r in records]

    try:
        if task_type == "regression":
            metrics = regression_evaluate(preds, labels, task)
        elif task_type == "molecule":
            metrics = molecule_evaluate(preds, labels, task, tokenizer=None)
        elif task_type == "caption":
            metrics = caption_evaluate(preds, labels, task, tokenizer=None,
                                       meteor_tokenizers=["wordnet"])
        else:
            return (task, strategy, {"skipped_task_type": task_type, "n": len(records)})
    except Exception as e:
        return (task, strategy, {"error": f"{type(e).__name__}: {e}", "n": len(records)})

    failure_idxs = metrics.pop("_failure_indices", [])
    result = {k: (float(v) if not isinstance(v, (list, dict)) else v)
              for k, v in metrics.items()}
    result["n"] = len(records)
    result["num_failed"] = len(failure_idxs)
    return (task, strategy, result)


def _eval_cls(args):
    import torch
    task, records = args
    probs = torch.tensor([r["probs"] for r in records])
    labels = [r["label"] for r in records]
    try:
        metrics = classification_evaluate(probs, labels, task)
    except Exception as e:
        return (task, {"error": f"{type(e).__name__}: {e}", "n": len(records)})
    failure_idxs = metrics.pop("_failure_indices", [])
    result = {k: (float(v) if not isinstance(v, (list, dict)) else v)
              for k, v in metrics.items()}
    result["n"] = len(records)
    result["num_failed"] = len(failure_idxs)
    return (task, result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="predictions_epoch*_step*.json path")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--num-workers", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] {args.input}", flush=True)
    with open(args.input, "r") as f:
        data = json.load(f)

    cls_items = data.get("classification", [])
    gen_items = data.get("generation", [])
    epoch = data.get("epoch")
    step = data.get("global_step")
    print(f"[load] epoch={epoch} step={step} "
          f"cls={len(cls_items)} gen={len(gen_items)}  "
          f"({(time.time()-t0):.1f}s)", flush=True)

    # Bucket gen by (task, strategy)
    gen_buckets = defaultdict(list)
    for g in gen_items:
        gen_buckets[(g["task"], g["strategy"])].append(g)
    cls_buckets = defaultdict(list)
    for c in cls_items:
        cls_buckets[c["task"]].append(c)

    gen_args = [(t, s, rs) for (t, s), rs in gen_buckets.items()]
    cls_args = [(t, rs) for t, rs in cls_buckets.items()]
    print(f"[bucket] gen {len(gen_args)} (task,strategy) groups, "
          f"cls {len(cls_args)} task groups", flush=True)

    results = {"epoch": epoch, "global_step": step,
               "source": os.path.abspath(args.input),
               "generation": {}, "classification": {}}

    # Generation metrics (parallel)
    print(f"[eval] generation via Pool({args.num_workers})...", flush=True)
    t1 = time.time()
    with mp.Pool(args.num_workers) as pool:
        for i, (task, strategy, res) in enumerate(
            pool.imap_unordered(_eval_bucket, gen_args), 1
        ):
            results["generation"].setdefault(task, {})[strategy] = res
            print(f"  [{i:2d}/{len(gen_args)}] {task} / {strategy}  "
                  f"n={res.get('n')} fail={res.get('num_failed')}", flush=True)
    print(f"[eval] gen done in {(time.time()-t1):.1f}s", flush=True)

    # Classification metrics (serial is fine — only ~4 tasks)
    print(f"[eval] classification serial...", flush=True)
    for (task, res) in map(_eval_cls, cls_args):
        results["classification"][task] = res
        print(f"  {task} n={res.get('n')} fail={res.get('num_failed')}", flush=True)

    # Write JSON
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[save] {args.output_json}", flush=True)

    # Write Markdown
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write(f"# Val Metric Re-aggregation\n\n")
        f.write(f"- Source: `{args.input}`\n")
        f.write(f"- epoch={epoch} step={step}\n\n")

        # Caption tasks (mol2text family)
        f.write("## Caption tasks (mol2text family)\n\n")
        f.write("| task | strategy | n | bleu4 | bleu2 | rouge1 | rouge2 | rougeL | meteor_wordnet | failure_rate |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for task in sorted(results["generation"]):
            if get_task_type(task) != "caption":
                continue
            for strategy in sorted(results["generation"][task]):
                r = results["generation"][task][strategy]
                f.write(f"| {task} | {strategy} | {r.get('n',0)} |"
                        f" {r.get('bleu4', float('nan')):.3f} |"
                        f" {r.get('bleu2', float('nan')):.3f} |"
                        f" {r.get('rouge1', float('nan')):.3f} |"
                        f" {r.get('rouge2', float('nan')):.3f} |"
                        f" {r.get('rougeL', float('nan')):.3f} |"
                        f" {r.get('meteor_wordnet', float('nan')):.3f} |"
                        f" {r.get('failure_rate', float('nan')):.4f} |\n")

        # Molecule tasks (text2mol + reactions)
        f.write("\n## Molecule tasks (text2mol + reactions)\n\n")
        f.write("| task | strategy | n | validity | exact_match | MACCS | RDK | morgan | bleu_smiles | failure_rate |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for task in sorted(results["generation"]):
            if get_task_type(task) != "molecule":
                continue
            for strategy in sorted(results["generation"][task]):
                r = results["generation"][task][strategy]
                f.write(f"| {task} | {strategy} | {r.get('n',0)} |"
                        f" {r.get('validity_ratio', float('nan')):.4f} |"
                        f" {r.get('exact_match_ratio', float('nan')):.4f} |"
                        f" {r.get('MACCS_FTS', float('nan')):.4f} |"
                        f" {r.get('RDK_FTS', float('nan')):.4f} |"
                        f" {r.get('morgan_FTS', float('nan')):.4f} |"
                        f" {r.get('bleu_smiles', float('nan')):.4f} |"
                        f" {r.get('failure_rate', float('nan')):.4f} |\n")

        # Regression tasks
        f.write("\n## Regression tasks\n\n")
        f.write("| task | strategy | n | mae | rmse | r2 | failure_rate |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for task in sorted(results["generation"]):
            if get_task_type(task) != "regression":
                continue
            for strategy in sorted(results["generation"][task]):
                r = results["generation"][task][strategy]
                f.write(f"| {task} | {strategy} | {r.get('n',0)} |"
                        f" {r.get('mae', float('nan')):.4f} |"
                        f" {r.get('rmse', float('nan')):.4f} |"
                        f" {r.get('r2', float('nan')):.4f} |"
                        f" {r.get('failure_rate', float('nan')):.4f} |\n")

        # Classification tasks
        f.write("\n## Classification tasks\n\n")
        f.write("| task | n | "
                + " | ".join(sorted({k for res in results["classification"].values()
                                    for k in res if k not in ("n","num_failed","error")}))
                + " |\n")
        keys = sorted({k for res in results["classification"].values()
                       for k in res if k not in ("n","num_failed","error")})
        f.write("|---|---:|" + "|".join([" ---: "] * len(keys)) + "|\n")
        for task in sorted(results["classification"]):
            r = results["classification"][task]
            row = [f"{task}", f"{r.get('n',0)}"]
            for k in keys:
                v = r.get(k)
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                elif v is None:
                    row.append("")
                else:
                    row.append(str(v))
            f.write("| " + " | ".join(row) + " |\n")

    print(f"[save] {args.output_md}", flush=True)
    print(f"[done] total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
