"""Build chebi-20-mol2text augmented dataset with {i}-th_rejected_* keys.

Filters raw_v1_10x_rephrase/Train for `task == "chebi-20-mol2text"`,
then applies `map_by_substructure_replacement_atomwise` to populate
`{i}-th_rejected_x`, `{i}-th_rejected_edge_index`, `{i}-th_rejected_edge_attr`
and the `additional_rejected_*` counterparts for i in [0, num_rejected_graphs).

Uses fts-targeted (0.7 mode) by default — Mol-LLM §B.4 implied evaluation
metric. Falls back to batch (0.55 mode) if `--target-fts 0` is passed.

Run:
    source /opt/MolDA/venv/MolDA/bin/activate
    PYTHONPATH=/opt/MolDA/src python scripts/build_chebi_mol2text_atomwise.py \\
        --num-proc 48 --target-fts 0.7 --num-rejected-graphs 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from datasets import load_from_disk  # noqa: E402

from dataset_generation.graph_aug_atomwise import (  # noqa: E402
    DEFAULT_FTS_MAX_ATTEMPTS,
    DEFAULT_TARGET_FTS,
    DEFAULT_TARGET_FTS_TOLERANCE,
    map_by_substructure_replacement_atomwise,
)


DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "Processed" / "raw_v1_10x_rephrase" / "Train"
DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "Processed" / "chebi_mol2text_atomwise" / "Train"
TARGET_TASK = "chebi-20-mol2text"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--num-proc", type=int, default=48)
    p.add_argument("--task-filter", type=str, default=TARGET_TASK,
                   help="keep only rows whose task == this value. "
                        "Empty string ('') disables filtering (process all rows).")
    p.add_argument("--exclude-tasks", type=str, default="",
                   help="comma-separated list of tasks to exclude (applied AFTER "
                        "task-filter). Use with --task-filter '' for multi-task builds. "
                        "Example: 'smol-name_conversion-i2s,smol-name_conversion-i2f,smol-name_conversion-s2i'")
    p.add_argument("--n-limit", type=int, default=None,
                   help="limit rows after filter (for dry-run)")
    p.add_argument("--replace-ratio", type=float, default=0.05)
    p.add_argument("--num-rejected-graphs", type=int, default=6)
    # fts-targeted mode params; pass --target-fts 0 to disable
    p.add_argument("--target-fts", type=float, default=DEFAULT_TARGET_FTS)
    p.add_argument("--target-fts-tolerance", type=float,
                   default=DEFAULT_TARGET_FTS_TOLERANCE)
    p.add_argument("--max-attempts", type=int, default=DEFAULT_FTS_MAX_ATTEMPTS)
    p.add_argument("--writer-batch-size", type=int, default=1000)
    p.add_argument("--skip-additional", action="store_true",
                   help="skip {i}-th_additional_rejected_* keys for non-reaction tasks "
                        "(halves the work; chebi-20-mol2text doesn't use additional)")
    args = p.parse_args()

    target_fts = args.target_fts if args.target_fts > 0 else None
    print(f"[build] loading {args.data_dir}", file=sys.stderr)
    ds = load_from_disk(str(args.data_dir))
    print(f"[build] full dataset: {len(ds):,}", file=sys.stderr)

    # Task filter: empty string ("") or None means "no filter — process all rows".
    # Useful when running on the entire raw_v1_10x_rephrase (mixed-task) dataset.
    # When non-empty, only rows whose `task` exactly matches are kept.
    if args.task_filter:
        print(f"[build] filtering task == {args.task_filter!r}", file=sys.stderr)
        t0 = time.time()
        filtered = ds.filter(
            lambda r: r.get("task") == args.task_filter,
            num_proc=args.num_proc,
        )
        print(f"[build] filter done in {time.time()-t0:.1f}s, "
              f"{len(filtered):,} rows", file=sys.stderr)
    else:
        print(f"[build] no task filter — processing all {len(ds):,} rows",
              file=sys.stderr)
        filtered = ds

    # Exclude tasks (applied after include-filter)
    if args.exclude_tasks:
        exclude_set = {t.strip() for t in args.exclude_tasks.split(",") if t.strip()}
        print(f"[build] excluding tasks: {sorted(exclude_set)}", file=sys.stderr)
        before = len(filtered)
        t0 = time.time()
        filtered = filtered.filter(
            lambda r: r.get("task") not in exclude_set,
            num_proc=args.num_proc,
        )
        print(f"[build] exclude done in {time.time()-t0:.1f}s, "
              f"{before:,} → {len(filtered):,} rows", file=sys.stderr)

    if args.n_limit is not None:
        n = min(args.n_limit, len(filtered))
        filtered = filtered.select(range(n))
        print(f"[build] limiting to first {n} rows for dry-run", file=sys.stderr)

    mode_desc = (f"fts-targeted (target={target_fts}, "
                 f"tol={args.target_fts_tolerance}, "
                 f"max_attempts={args.max_attempts})"
                 if target_fts is not None else "batch")
    print(f"[build] mode: {mode_desc}", file=sys.stderr)
    print(f"[build] replace_ratio={args.replace_ratio}, "
          f"num_rejected_graphs={args.num_rejected_graphs}", file=sys.stderr)

    def _apply(row):
        return map_by_substructure_replacement_atomwise(
            row,
            replace_ratio=args.replace_ratio,
            num_rejected_graphs=args.num_rejected_graphs,
            target_fts=target_fts,
            target_fts_tolerance=args.target_fts_tolerance,
            max_attempts=args.max_attempts,
            skip_additional_rejected=args.skip_additional,
        )

    print(f"[build] mapping with num_proc={args.num_proc}", file=sys.stderr)
    t0 = time.time()
    mapped = filtered.map(
        _apply,
        num_proc=args.num_proc,
        writer_batch_size=args.writer_batch_size,
        desc="atomwise-augment",
    )
    print(f"[build] map done in {time.time()-t0:.1f}s "
          f"({(time.time()-t0)/max(len(filtered),1)*1000:.1f} ms/row)",
          file=sys.stderr)

    print(f"[build] saving to {args.output}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mapped.save_to_disk(str(args.output))
    print(f"[build] done. final size: {len(mapped):,} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
