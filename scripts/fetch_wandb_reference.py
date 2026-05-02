#!/usr/bin/env python3
"""Pull reference run metric history + config from wandb.

Reads runs by URL (e.g. https://wandb.ai/{entity}/{project}/runs/{run_id}).
Saves:
  - {out_dir}/{run_id}_config.json     (run.config as dict)
  - {out_dir}/{run_id}_history.csv     (full metric time-series via run.scan_history)
  - {out_dir}/{run_id}_summary.json    (run.summary)

Usage:
  venvs/MolDA/bin/python scripts/fetch_wandb_reference.py \
      --url https://wandb.ai/hj_ai/mol-llm_llada/runs/krozx9yp \
      --url https://wandb.ai/hj_ai/MolDA_pro6000/runs/7xpk8frx \
      --out-dir docs/analysis/wandb
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import wandb

URL_RE = re.compile(
    r"^https?://(?:[^/]+/)?wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)"
)


def parse_run_url(url: str):
    u = urlparse(url)
    parts = [p for p in u.path.strip("/").split("/") if p]
    # path: {entity}/{project}/runs/{run_id}
    if len(parts) >= 4 and parts[-2] == "runs":
        return parts[-4], parts[-3], parts[-1]
    raise ValueError(f"could not parse wandb run url: {url}")


def fetch_run(api, entity: str, project: str, run_id: str, out_dir: Path):
    run_path = f"{entity}/{project}/{run_id}"
    print(f"[fetch] {run_path}", flush=True)
    run = api.run(run_path)

    # config
    cfg_path = out_dir / f"{run_id}_config.json"
    try:
        cfg = {k: v for k, v in run.config.items() if not k.startswith("_")}
    except Exception as e:
        cfg = {"_error": str(e)}
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [save] {cfg_path}", flush=True)

    # summary
    sum_path = out_dir / f"{run_id}_summary.json"
    with open(sum_path, "w") as f:
        json.dump(dict(run.summary), f, ensure_ascii=False, indent=2, default=str)
    print(f"  [save] {sum_path}", flush=True)

    # history — page-by-page with retry (scan_history can time out on large runs)
    hist_path = out_dir / f"{run_id}_history.csv"
    # .history(pandas=False, samples=N) returns metric series; default samples=500
    # Use run.history(samples=100000) to cap at full resolution for long runs.
    try:
        hist = run.history(samples=100000, pandas=False)
    except Exception as e:
        print(f"  [warn] history() failed ({e}); falling back to scan_history", flush=True)
        try:
            hist = list(run.scan_history(page_size=500))
        except Exception as e2:
            print(f"  [error] scan_history failed: {e2}", flush=True)
            hist = []

    if not hist:
        print(f"  [warn] empty history for {run_path}", flush=True)
        return
    keys = sorted({k for r in hist for k in r.keys()})
    with open(hist_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in hist:
            w.writerow(r)
    print(f"  [save] {hist_path}  (rows={len(hist)}, cols={len(keys)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", required=True,
                    help="wandb run URL (may repeat)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("[warn] WANDB_API_KEY not set — relying on cached login", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    for url in args.url:
        entity, project, run_id = parse_run_url(url)
        fetch_run(api, entity, project, run_id, out_dir)


if __name__ == "__main__":
    main()
