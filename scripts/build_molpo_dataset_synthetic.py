"""Build a synthetic MolPO chosen/rejected dataset from an existing Arrow dataset.

Strategy (toy / Phase 3 sanity only — NOT a learning signal):
  - chosen   = original target_text
  - rejected = target_text from another sample of the SAME task (random pairing)

This produces structurally valid chosen/rejected pairs so V-MolPO forward path
can be exercised end-to-end. Real preference learning needs a proper rejection
strategy (e.g., low-confidence model generations from Stage 1/2 ckpt).

Usage:
    python scripts/build_molpo_dataset_synthetic.py \\
        --src dataset/Processed/raw_v1_10x_rephrase \\
        --dst dataset/Processed/raw_v1_10x_rephrase_molpo_synthetic \\
        --task chebi-20-mol2text \\
        --max-samples 1000 \\
        --seed 42

    # → dst/Train, dst/Val, dst/Test 에 chosen/rejected 컬럼 추가된 Arrow dataset
"""
import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import load_from_disk, Dataset


def synthesize_pairs(ds: Dataset, task_filter: str | None, mol_token_type: str,
                     max_samples: int | None, seed: int,
                     max_target_token_length: int | None = None,
                     tokenizer=None) -> Dataset:
    """Add target_text_chosen / target_text_rejected columns by random pairing.

    Args:
        max_target_token_length: filter out samples whose tokenized target exceeds
            this. Useful when generation cap (gen_max_len) is set (e.g., 256) — train/val
            consistency: model can never generate an answer longer than gen_max_len, so
            ground-truth target longer than gen_max_len is unreachable → exact_match
            forced to 0 for those samples regardless of model quality.
        tokenizer: HF tokenizer instance (required if max_target_token_length set)
    """
    rng = random.Random(seed)

    # Filter by task if requested
    if task_filter:
        keep_mask = [t == task_filter for t in ds["task"]]
        ds = ds.filter(lambda _, idx: keep_mask[idx], with_indices=True)

    # Determine target column name
    chosen_col = f"target_text_{mol_token_type}" if f"target_text_{mol_token_type}" in ds.column_names else "target_text"
    if chosen_col not in ds.column_names:
        raise RuntimeError(f"Neither 'target_text' nor 'target_text_{mol_token_type}' in dataset")

    # Length filter (before max_samples sub-sampling)
    if max_target_token_length is not None and max_target_token_length > 0:
        if tokenizer is None:
            raise ValueError("max_target_token_length requires tokenizer")
        n_before = len(ds)
        keep_mask = [
            len(tokenizer.encode(t, add_special_tokens=False)) <= max_target_token_length
            for t in ds[chosen_col]
        ]
        ds = ds.filter(lambda _, idx: keep_mask[idx], with_indices=True)
        n_after = len(ds)
        print(f"  Length filter (target ≤ {max_target_token_length} tokens): "
              f"{n_before} → {n_after} ({100*(n_before-n_after)/n_before:.1f}% removed)")

    if max_samples and len(ds) > max_samples:
        idx = rng.sample(range(len(ds)), max_samples)
        ds = ds.select(idx)

    n = len(ds)
    if n < 2:
        raise RuntimeError(f"Need ≥2 samples for synthetic pairing; got {n}.")

    targets = ds[chosen_col]

    # Random rejection: each sample i pairs with a different j ≠ i (same dataset)
    rejected_idx = list(range(n))
    rng.shuffle(rejected_idx)
    # Avoid self-pairing
    for i in range(n):
        if rejected_idx[i] == i:
            rejected_idx[i] = (i + 1) % n

    rejected_targets = [targets[j] for j in rejected_idx]

    # Add columns (single + dual variant for compat)
    new_cols = {
        "target_text_chosen": list(targets),
        "target_text_rejected": rejected_targets,
        f"target_text_chosen_{mol_token_type}": list(targets),
        f"target_text_rejected_{mol_token_type}": rejected_targets,
    }

    for col, vals in new_cols.items():
        if col in ds.column_names:
            ds = ds.remove_columns([col])
        ds = ds.add_column(col, vals)

    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="source Arrow dataset root (has Train/Val/Test/)")
    ap.add_argument("--dst", required=True, help="destination root")
    ap.add_argument("--task", default=None, help="filter by task (e.g. chebi-20-mol2text)")
    ap.add_argument("--mol-token-type", default="selfies", choices=["selfies", "smiles"])
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap per-split sample count (Phase 3 sanity)")
    ap.add_argument("--max-target-token-length", type=int, default=None,
                    help="filter out samples whose tokenized target exceeds this length "
                         "(e.g., 256 = generation cap). NaN exact_match defense.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", nargs="+", default=["Train", "Val", "Test"])
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Lazy load tokenizer only when length filter requested
    tokenizer = None
    if args.max_target_token_length is not None:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer for length filter (max={args.max_target_token_length}) ...")
        tokenizer = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
        )

    for split in args.splits:
        src_split = src / split
        if not src_split.exists():
            print(f"[skip] {src_split} not found")
            continue
        ds = load_from_disk(str(src_split))
        n_in = len(ds)
        ds_out = synthesize_pairs(
            ds, task_filter=args.task,
            mol_token_type=args.mol_token_type,
            max_samples=args.max_samples,
            seed=args.seed,
            max_target_token_length=args.max_target_token_length,
            tokenizer=tokenizer,
        )
        n_out = len(ds_out)
        out_split = dst / split
        ds_out.save_to_disk(str(out_split))
        print(f"[{split}] {n_in} → {n_out} samples (task={args.task}) → {out_split}")

    print("\nDone. To use:")
    print(f"  bash scripts/train_stage3_v_molpo_chebi.sh  # default reads dataset/Processed/...")
    print(f"  # or override: data.root={dst.relative_to('dataset/Processed') if 'dataset/Processed' in str(dst) else dst}")


if __name__ == "__main__":
    main()
