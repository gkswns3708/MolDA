"""Run atomwise graph augmentation on N chebi-20-mol2text samples and
report distributions of n_removed / n_added / atom_change_ratio, plus
CCC fallback rate and fallback_reason breakdown.

Reads from `raw_v1_10x_rephrase/Train`, filters `task == "chebi-20-mol2text"`,
seed-samples N rows, and feeds each row's `input_mol_string_selfies`
through `extract_and_modify_atomwise(replace_ratio)`.

Usage:
    source /opt/MolDA/venv/MolDA/bin/activate
    PYTHONPATH=/opt/MolDA/src python scripts/stats_graph_aug_atomwise_chebi.py \\
        --n 2000 --replace-ratio 0.3 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from datasets import load_from_disk  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import DataStructs, MACCSkeys  # noqa: E402

from dataset_generation.graph_aug_atomwise import (  # noqa: E402
    CCC_FALLBACK_SMILES,
    DEFAULT_FTS_MAX_ATTEMPTS,
    DEFAULT_TARGET_FTS,
    DEFAULT_TARGET_FTS_TOLERANCE,
    extract_and_modify_atomwise,
    extract_and_modify_fts_targeted,
)


def _maccs_fts(orig_smiles: str, mod_smiles: str):
    """Return (fts, n_bits_flipped) or (None, None) if either fails to parse."""
    orig = Chem.MolFromSmiles(orig_smiles)
    mod = Chem.MolFromSmiles(mod_smiles)
    if orig is None or mod is None:
        return None, None
    orig_fp = MACCSkeys.GenMACCSKeys(orig)
    mod_fp = MACCSkeys.GenMACCSKeys(mod)
    fts = DataStructs.TanimotoSimilarity(orig_fp, mod_fp)
    n_flipped = sum(1 for i in range(orig_fp.GetNumBits())
                    if orig_fp.GetBit(i) != mod_fp.GetBit(i))
    return fts, n_flipped


def _rdkit_fts(orig_smiles: str, mod_smiles: str):
    orig = Chem.MolFromSmiles(orig_smiles)
    mod = Chem.MolFromSmiles(mod_smiles)
    if orig is None or mod is None:
        return None
    return DataStructs.TanimotoSimilarity(
        Chem.RDKFingerprint(orig),
        Chem.RDKFingerprint(mod),
    )


DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "Processed" / "raw_v1_10x_rephrase" / "Train"
DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "stats" / "graph_aug_atomwise_chebi.json"
TARGET_TASK = "chebi-20-mol2text"


def _strip_selfies_wrapper(s: str) -> str:
    return s.replace("<SELFIES>", "").replace("</SELFIES>", "").replace(" ", "")


def percentile(values, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def summarize(name: str, values: list) -> dict:
    if not values:
        return {"name": name, "count": 0}
    return {
        "name": name,
        "count": len(values),
        "mean": round(mean(values), 4),
        "std": round(pstdev(values), 4) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "p25": round(percentile(values, 0.25), 4),
        "p50": round(median(values), 4),
        "p75": round(percentile(values, 0.75), 4),
        "p95": round(percentile(values, 0.95), 4),
        "max": float(max(values)),
    }


def histogram(values: list, bins: list) -> dict:
    """Bins: sorted list of right-edges. Returns {bin_label: count}."""
    out = {}
    edges = list(bins) + [float("inf")]
    for lo, hi in zip([float("-inf")] + bins, edges):
        if hi == float("inf"):
            label = f">={lo:.2f}"
        elif lo == float("-inf"):
            label = f"<{hi:.2f}"
        else:
            label = f"[{lo:.2f},{hi:.2f})"
        out[label] = 0
    for v in values:
        for lo, hi, label in zip([float("-inf")] + bins, edges, list(out.keys())):
            if lo <= v < hi:
                out[label] += 1
                break
    return out


def format_md_table(rows: list[dict], cols: list[str]) -> str:
    """rows = list of dicts; cols = list of keys to render."""
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def run(n: int, replace_ratio: float, seed: int, data_dir: Path, output: Path,
        verbose: bool, num_filter_proc: int,
        target_fts: float | None = None,
        target_fts_tolerance: float = DEFAULT_TARGET_FTS_TOLERANCE,
        max_attempts: int = DEFAULT_FTS_MAX_ATTEMPTS) -> dict:
    print(f"[stats] loading dataset from {data_dir}", file=sys.stderr)
    ds = load_from_disk(str(data_dir))
    print(f"[stats] full dataset size: {len(ds):,}", file=sys.stderr)

    print(f"[stats] filtering task == {TARGET_TASK!r} (this scans all rows)", file=sys.stderr)
    t0 = time.time()
    filtered = ds.filter(lambda r: r.get("task") == TARGET_TASK, num_proc=num_filter_proc)
    print(f"[stats] filter done in {time.time()-t0:.1f}s, {len(filtered):,} chebi rows",
          file=sys.stderr)

    if len(filtered) < n:
        raise SystemExit(f"only {len(filtered)} chebi-20-mol2text rows available, need {n}")

    rng = random.Random(seed)
    indices = rng.sample(range(len(filtered)), n)
    indices.sort()

    n_removed, n_added, atom_change_ratios = [], [], []
    original_atom_counts = []
    maccs_fts_vals, maccs_dist_vals, maccs_bits_flipped = [], [], []
    rdkit_fts_vals = []
    fallback_counter: Counter[str] = Counter()
    ccc_count = 0
    exception_count = 0
    reparse_fail_count = 0
    elapsed = 0.0
    fts_elapsed = 0.0

    for k, idx in enumerate(indices):
        row = filtered[idx]
        selfies = _strip_selfies_wrapper(row.get("input_mol_string_selfies") or "")
        if not selfies:
            fallback_counter["empty_selfies"] += 1
            continue

        random.seed(seed * 7919 + k)
        t_start = time.time()
        try:
            if target_fts is None:
                out = extract_and_modify_atomwise(
                    selfies, replace_ratio=replace_ratio, verbose=verbose,
                )
            else:
                out = extract_and_modify_fts_targeted(
                    selfies,
                    target_fts=target_fts,
                    target_fts_tolerance=target_fts_tolerance,
                    replace_ratio=replace_ratio,
                    max_attempts=max_attempts,
                    verbose=verbose,
                )
        except Exception as e:
            exception_count += 1
            fallback_counter[f"exception:{type(e).__name__}"] += 1
            elapsed += time.time() - t_start
            continue
        elapsed += time.time() - t_start

        if out.get("modified_smiles") == CCC_FALLBACK_SMILES:
            ccc_count += 1
        reason = out.get("fallback_reason")
        fallback_counter[reason if reason is not None else "ok"] += 1

        n_removed.append(out["n_removed"])
        n_added.append(out["n_added"])
        atom_change_ratios.append(out["atom_change_ratio"])
        ratio = out["atom_change_ratio"]
        if ratio > 0:
            original_atom_counts.append(round((out["n_removed"] + out["n_added"]) / ratio))

        # MACCS / RDKit fingerprint Tanimoto — primary Mol-LLM metric
        if out.get("fallback_reason") is None:
            t_fts = time.time()
            fts, n_flipped = _maccs_fts(out["original_smiles"], out["modified_smiles"])
            rdfts = _rdkit_fts(out["original_smiles"], out["modified_smiles"])
            fts_elapsed += time.time() - t_fts
            if fts is None:
                reparse_fail_count += 1
            else:
                maccs_fts_vals.append(fts)
                maccs_dist_vals.append(1.0 - fts)
                maccs_bits_flipped.append(n_flipped)
                if rdfts is not None:
                    rdkit_fts_vals.append(rdfts)

        if (k + 1) % 200 == 0:
            print(f"[stats] processed {k+1}/{n} (elapsed {elapsed:.1f}s, "
                  f"fts_elapsed {fts_elapsed:.1f}s)", file=sys.stderr)

    summary = {
        "n_samples_requested": n,
        "n_samples_processed": len(atom_change_ratios) + ccc_count + exception_count,
        "n_samples_with_valid_result": len(atom_change_ratios),
        "ccc_fallback_count": ccc_count,
        "ccc_fallback_rate": round(ccc_count / n, 4) if n else 0.0,
        "exception_count": exception_count,
        "fallback_reason_counter": dict(fallback_counter),
        "replace_ratio": replace_ratio,
        "seed": seed,
        "target_task": TARGET_TASK,
        "elapsed_seconds": round(elapsed, 2),
        "fts_elapsed_seconds": round(fts_elapsed, 2),
        "mean_seconds_per_sample": round(elapsed / max(n, 1), 4),
        "smiles_reparse_fail_count": reparse_fail_count,
        "smiles_reparse_fail_rate": round(reparse_fail_count / max(n, 1), 4),
        "primary_metrics": {
            "maccs_fts": summarize("maccs_fts", maccs_fts_vals),
            "maccs_dist": summarize("maccs_dist", maccs_dist_vals),
            "maccs_bits_flipped": summarize("maccs_bits_flipped", maccs_bits_flipped),
            "rdkit_fts": summarize("rdkit_fts", rdkit_fts_vals),
        },
        "maccs_fts_target_band": {
            "band": "[0.60, 0.80]",
            "count_in_band": sum(1 for v in maccs_fts_vals if 0.60 <= v <= 0.80),
            "rate_in_band": round(
                sum(1 for v in maccs_fts_vals if 0.60 <= v <= 0.80)
                / max(len(maccs_fts_vals), 1), 4),
        },
        "maccs_fts_histogram": histogram(
            maccs_fts_vals, [0.10, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        ),
        "atom_distributions": {
            "n_removed": summarize("n_removed", n_removed),
            "n_added": summarize("n_added", n_added),
            "atom_change_ratio": summarize("atom_change_ratio", atom_change_ratios),
            "original_atom_count": summarize("original_atom_count", original_atom_counts),
        },
        "atom_change_ratio_histogram": histogram(
            atom_change_ratios, [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0],
        ),
        "n_added_histogram": histogram(
            n_added, [1, 2, 4, 8, 16, 32],
        ),
    }

    # markdown report
    mean_fts = summary["primary_metrics"]["maccs_fts"].get("mean", 0.0)
    print()
    print(f"# Atomwise augmentation stats — {TARGET_TASK} × {n} samples")
    print()
    print(f"- replace_ratio: **{replace_ratio}**, seed: **{seed}**")
    print(f"- elapsed: {elapsed:.1f}s ({summary['mean_seconds_per_sample']*1000:.2f} ms/sample)"
          f"  + {fts_elapsed:.1f}s for FTS computation")
    print(f"- valid results: **{summary['n_samples_with_valid_result']}/{n}**")
    print(f"- CCC fallback: **{ccc_count}/{n} ({summary['ccc_fallback_rate']*100:.2f}%)**")
    print(f"- SMILES re-parse failures: {reparse_fail_count}/{n} "
          f"({summary['smiles_reparse_fail_rate']*100:.2f}%)")
    print(f"- exceptions: {exception_count}")
    print()
    print(f"## ★ Primary metric — MACCS Tanimoto (target ≈ 0.70)")
    print()
    print(f"- **mean MACCS FTS: {mean_fts:.3f}** (target ~0.70)")
    print(f"- samples in target band [0.60, 0.80]: "
          f"{summary['maccs_fts_target_band']['count_in_band']}/"
          f"{len(maccs_fts_vals)} "
          f"({summary['maccs_fts_target_band']['rate_in_band']*100:.1f}%)")
    print()
    primary_rows = [summary["primary_metrics"][k] for k in (
        "maccs_fts", "maccs_dist", "maccs_bits_flipped", "rdkit_fts")]
    print(format_md_table(
        primary_rows,
        ["name", "count", "mean", "std", "min", "p25", "p50", "p75", "p95", "max"],
    ))
    print()
    print("### MACCS FTS histogram")
    print()
    for label, c in summary["maccs_fts_histogram"].items():
        print(f"- `{label}`: {c}")
    print()
    print("## fallback_reason breakdown")
    print()
    for reason, count in sorted(fallback_counter.items(), key=lambda x: -x[1]):
        print(f"- `{reason}`: {count}")
    print()
    print("## Atom-level distributions (secondary — Mol-LLM target is FTS, not atom counts)")
    print()
    dist_rows = [summary["atom_distributions"][k] for k in (
        "n_removed", "n_added", "atom_change_ratio", "original_atom_count")]
    print(format_md_table(
        dist_rows,
        ["name", "count", "mean", "std", "min", "p25", "p50", "p75", "p95", "max"],
    ))
    print()
    print("## atom_change_ratio histogram")
    print()
    for label, c in summary["atom_change_ratio_histogram"].items():
        print(f"- `{label}`: {c}")
    print()
    print("## n_added histogram")
    print()
    for label, c in summary["n_added_histogram"].items():
        print(f"- `{label}`: {c}")
    print()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[stats] wrote JSON → {output}", file=sys.stderr)

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--replace-ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--num-filter-proc", type=int, default=8)
    p.add_argument("--target-fts", type=float, default=None,
                   help="if set, switch to fts-targeted algorithm (best-of-N attempts)")
    p.add_argument("--target-fts-tolerance", type=float,
                   default=DEFAULT_TARGET_FTS_TOLERANCE)
    p.add_argument("--max-attempts", type=int, default=DEFAULT_FTS_MAX_ATTEMPTS)
    args = p.parse_args()
    run(args.n, args.replace_ratio, args.seed, args.data_dir, args.output,
        args.verbose, args.num_filter_proc,
        target_fts=args.target_fts,
        target_fts_tolerance=args.target_fts_tolerance,
        max_attempts=args.max_attempts)


if __name__ == "__main__":
    main()
