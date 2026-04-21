"""Full dataset에서 task별 N개씩 sampling하여 toy dataset 생성.

Usage:
    cd /opt/11-MolDA/New_MolDA
    source venvs/MolDA/bin/activate

    # SELFIES + SMILES 둘 다 (기본 100개)
    python scripts/make_toy_dataset.py

    # task별 50개, SELFIES만
    python scripts/make_toy_dataset.py --n 50 --repr selfies
"""

import argparse
from pathlib import Path

from datasets import Dataset, load_from_disk


def sample_by_task(ds, n: int, seed: int = 42) -> Dataset:
    """task 컬럼 기준으로 task별 최대 n개 sampling."""
    df = ds.to_pandas()
    sampled = (
        df.groupby("task", group_keys=False)
        .apply(lambda g: g.sample(min(n, len(g)), random_state=seed))
        .reset_index(drop=True)
    )
    return Dataset.from_pandas(sampled)


def main():
    parser = argparse.ArgumentParser(description="Make toy dataset from full dataset")
    parser.add_argument("--n", type=int, default=100, help="task별 sample 수 (default: 100)")
    parser.add_argument("--repr", choices=["selfies", "smiles", "both"], default="both",
                        help="표현 방식 (default: both)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    representations = {
        "selfies": ("SELFIES", "raw_v1"),
        "smiles": ("SMILES", "raw_v1"),
    }
    if args.repr == "both":
        targets = list(representations.values())
    else:
        targets = [representations[args.repr]]

    for repr_name, data_tag in targets:
        src_root = project_root / "dataset" / "Raw" / repr_name / data_tag
        dst_root = project_root / "dataset" / "Processed" / repr_name / f"toy{args.n}"

        print(f"\n{'='*60}")
        print(f"  {repr_name}: {src_root} → {dst_root}")
        print(f"{'='*60}")

        for split in ["Train", "Val", "Test"]:
            src_path = src_root / split
            if not src_path.exists():
                print(f"  [SKIP] {src_path} not found")
                continue

            ds = load_from_disk(str(src_path))
            sampled = sample_by_task(ds, args.n, args.seed)

            dst_path = dst_root / split
            dst_path.mkdir(parents=True, exist_ok=True)
            sampled.save_to_disk(str(dst_path))

            n_tasks = sampled.to_pandas()["task"].nunique()
            print(f"  {split}: {len(ds):>8,} → {len(sampled):>5,}  ({n_tasks} tasks)")

    print(f"\nDone! Config에서 사용할 경로:")
    for repr_name, _ in targets:
        print(f"  data.root=dataset/Processed/{repr_name}/toy{args.n}")


if __name__ == "__main__":
    main()
