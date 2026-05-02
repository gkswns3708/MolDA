#!/usr/bin/env python3
"""Build comparison report: version_21 (current) vs krozx9yp (ref-multi) vs 7xpk8frx (ref-single).

Inputs:
  - docs/analysis/metrics_v21_epoch1.json   (current re-aggregated)
  - docs/analysis/wandb/krozx9yp_history.csv
  - docs/analysis/wandb/krozx9yp_config.json / _summary.json
  - docs/analysis/wandb/7xpk8frx_summary.json
Output:
  - docs/analysis/REPHRASE_REGRESSION_ANALYSIS.md

Strategy mapping between new/old code:
  new `low_confidence_random`   ↔ old `random`       (old has fixed low_confidence remasking)
  new `low_confidence_semi_ar`  ↔ old `semi_ar`
  new `random_random` / `random_semi_ar` → not present in old refs (remasking='random' new only)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_wandb_history(csv_path: str, target_step: int):
    """Find the closest-step row (by _step column) to target_step."""
    best_row = None
    best_dist = math.inf
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            s = row.get("_step") or row.get("global_step")
            if not s:
                continue
            try:
                s = int(float(s))
            except ValueError:
                continue
            d = abs(s - target_step)
            if d < best_dist:
                best_dist = d
                best_row = (s, row)
    return best_row


def read_wandb_epoch_row(csv_path: str, target_epoch: int,
                         probe_key: str = "val/chebi-20-mol2text/bleu4_semi_ar"):
    """Find row with epoch==target_epoch AND non-empty probe_key (val-logged row)."""
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            e = row.get("epoch")
            if not e:
                continue
            try:
                e = int(float(e))
            except ValueError:
                continue
            if e != target_epoch:
                continue
            v = row.get(probe_key)
            if v and v not in ("", "NaN"):
                return row
    return None


def find_best_epoch_row(csv_path: str,
                       probe_key: str = "val/chebi-20-mol2text/bleu4_semi_ar"):
    """Find the epoch row with max probe_key value."""
    best = None
    best_v = -float("inf")
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            v = row.get(probe_key)
            if not v or v in ("", "NaN"):
                continue
            try:
                vv = float(v)
            except ValueError:
                continue
            if vv > best_v:
                best_v = vv
                best = row
    return best


def fnum(x, default=float("nan")):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return v
    except Exception:
        return default


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-metrics", default="docs/analysis/metrics_v21_epoch1.json")
    ap.add_argument("--krozx9yp-history", default="docs/analysis/wandb/krozx9yp_history.csv")
    ap.add_argument("--krozx9yp-summary", default="docs/analysis/wandb/krozx9yp_summary.json")
    ap.add_argument("--krozx9yp-config", default="docs/analysis/wandb/krozx9yp_config.json")
    ap.add_argument("--f7xpk-summary", default="docs/analysis/wandb/7xpk8frx_summary.json")
    ap.add_argument("--f7xpk-config", default="docs/analysis/wandb/7xpk8frx_config.json")
    ap.add_argument("--target-step", type=int, default=8526)
    ap.add_argument("--target-epoch", type=int, default=1)
    ap.add_argument("--output", default="docs/analysis/REPHRASE_REGRESSION_ANALYSIS.md")
    args = ap.parse_args()

    # Load current
    current = json.load(open(args.current_metrics))

    # Load krozx9yp: epoch-aligned val-logged row (primary), plus best epoch row (reference)
    krow = read_wandb_epoch_row(args.krozx9yp_history, args.target_epoch)
    kbest = find_best_epoch_row(args.krozx9yp_history)
    kstep = int(float(krow.get("_step"))) if krow else None
    kbstep = int(float(kbest.get("_step"))) if kbest else None
    ksum = json.load(open(args.krozx9yp_summary))
    kcfg = json.load(open(args.krozx9yp_config))

    # Load 7xpk8frx (only summary — best/final metric values)
    f7sum = json.load(open(args.f7xpk_summary))
    f7cfg = json.load(open(args.f7xpk_config))

    gen = current.get("generation", {})

    # --- build report ---
    lines = []
    lines.append("# Rephrase 학습 성능 퇴행 분석 (Phase 1)\n")
    lines.append(
        "- **비교 대상 (current)**: `lightning_logs/version_21/val_predictions/predictions_epoch1_step8526.json`\n"
        f"  (epoch={current.get('epoch')}, step={current.get('global_step')})\n"
        f"- **참조 (multi-task, rephrase 없음)**: wandb `hj_ai/mol-llm_llada/krozx9yp`, "
        f"step=32 sampling, string_only, data_tag=`{kcfg.get('data_tag')}`\n"
        f"  - **epoch={args.target_epoch} aligned row**: `_step={kstep}` (val-logged)\n"
        f"  - **best epoch row** (for reference ceiling): `_step={kbstep}` "
        f"epoch={kbest.get('epoch') if kbest else '—'}\n"
        f"- **참조 (single-task mol2text, high sampling)**: wandb `hj_ai/MolDA_pro6000/7xpk8frx`, "
        f"step=**256** sampling, string_only, data_tag=`{f7cfg.get('data_tag')}`\n"
        "\n"
    )

    # Config-level comparison
    lines.append("## 0. 실험 환경 비교\n\n")
    lines.append("| 항목 | **version_21 (current)** | **krozx9yp (ref multi)** | **7xpk8frx (ref single)** |\n")
    lines.append("|---|---|---|---|\n")
    rows = [
        ("data tag", "raw_v1_10x_rephrase (+10x rephrase)", kcfg.get("data_tag"), f7cfg.get("data_tag")),
        ("mol representation", "string+graph", kcfg.get("mol_representation"), f7cfg.get("mol_representation")),
        ("sampling_steps", "32", kcfg.get("sampling_steps"), f7cfg.get("sampling_steps")),
        ("remasking_strategy", "[low_confidence, random]", kcfg.get("remasking_strategy"), f7cfg.get("remasking_strategy")),
        ("val_strategies", "[random, semi_ar]", "random + semi_ar", "random + semi_ar"),
        ("tasks", "18 tasks (multi-task)", "~24 tasks (multi-task)", "chebi-20-mol2text only"),
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |\n")
    lines.append("\n")

    # --- Caption tasks (mol2text family) ---
    lines.append("## 1. Caption tasks (mol2text family)\n\n")
    lines.append(
        "Current는 remasking × sampling 4 strategy를 수집하므로, old 참조와 매칭되는 "
        "`low_confidence_random` / `low_confidence_semi_ar`만 좌측에 배치하고, 참고로 "
        "`random_*` strategy도 함께 표기.\n\n"
    )
    lines.append(
        "| task | metric | current LC×random | current LC×semi_ar | **krozx9yp @epoch=1** random | semi_ar | **krozx9yp BEST** random | semi_ar | "
        "**7xpk8frx** best random | semi_ar |\n"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    caption_tasks = ["chebi-20-mol2text", "smol-molecule_captioning"]
    metrics = [("bleu4", "bleu4"), ("bleu2", "bleu2"),
               ("rouge1", "rouge1"), ("rouge2", "rouge2"), ("rougeL", "rougeL"),
               ("meteor_wordnet", "meteor_wordnet")]
    def _k_pair(row, task, m_ref):
        if row is None:
            return None, None
        if m_ref == "meteor_wordnet":
            r = fnum(row.get(f"val/{task}/meteor_wordnet_random")) or fnum(row.get(f"val/{task}/meteor_random"))
            s = fnum(row.get(f"val/{task}/meteor_wordnet_semi_ar")) or fnum(row.get(f"val/{task}/meteor_semi_ar"))
        else:
            r = fnum(row.get(f"val/{task}/{m_ref}_random"))
            s = fnum(row.get(f"val/{task}/{m_ref}_semi_ar"))
        return r, s

    for task in caption_tasks:
        for m_cur, m_ref in metrics:
            cr = gen.get(task, {}).get("low_confidence_random", {}).get(m_cur)
            cs = gen.get(task, {}).get("low_confidence_semi_ar", {}).get(m_cur)
            kr, ks = _k_pair(krow, task, m_ref)
            kbr, kbs = _k_pair(kbest, task, m_ref)
            # 7xpk8frx: only chebi-20-mol2text — from summary
            fr = fs = None
            if task == "chebi-20-mol2text":
                if m_ref == "meteor_wordnet":
                    fr = fnum(f7sum.get(f"val/{task}/meteor_wordnet_random"))
                    fs = fnum(f7sum.get(f"val/{task}/meteor_wordnet_semi_ar"))
                else:
                    fr = fnum(f7sum.get(f"val/{task}/{m_ref}_random"))
                    fs = fnum(f7sum.get(f"val/{task}/{m_ref}_semi_ar"))
            lines.append(
                f"| {task} | {m_cur} | {fmt(cr)} | {fmt(cs)} | "
                f"{fmt(kr)} | {fmt(ks)} | {fmt(kbr)} | {fmt(kbs)} | "
                f"{fmt(fr)} | {fmt(fs)} |\n"
            )
    lines.append("\n")

    # --- Molecule tasks ---
    lines.append("## 2. Molecule tasks (text2mol + reactions)\n\n")
    lines.append(
        "| task | metric | current LC×random | current LC×semi_ar | "
        "current R×random (new) | krozx9yp @ep=1 random | semi_ar | krozx9yp BEST random | semi_ar |\n"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    mol_tasks = [
        "chebi-20-text2mol", "smol-molecule_generation",
        "forward_reaction_prediction", "retrosynthesis", "reagent_prediction",
        "smol-forward_synthesis", "smol-retrosynthesis",
    ]
    mol_metrics = [("validity_ratio", "validity_ratio"),
                   ("exact_match_ratio", "exact_match_ratio"),
                   ("MACCS_FTS", "MACCS_FTS"),
                   ("RDK_FTS", "RDK_FTS"),
                   ("morgan_FTS", "morgan_FTS"),
                   ("failure_rate", "failure_rate")]
    for task in mol_tasks:
        for m_cur, m_ref in mol_metrics:
            lc_r = gen.get(task, {}).get("low_confidence_random", {}).get(m_cur)
            lc_s = gen.get(task, {}).get("low_confidence_semi_ar", {}).get(m_cur)
            rr = gen.get(task, {}).get("random_random", {}).get(m_cur)
            kr, ks = _k_pair(krow, task, m_ref)
            kbr, kbs = _k_pair(kbest, task, m_ref)
            lines.append(
                f"| {task} | {m_cur} | {fmt(lc_r)} | {fmt(lc_s)} | {fmt(rr)} | "
                f"{fmt(kr)} | {fmt(ks)} | {fmt(kbr)} | {fmt(kbs)} |\n"
            )
    lines.append("\n")

    # --- Observations ---
    lines.append("## 3. 주요 관측\n\n")

    # Pull a few headline numbers
    def cur(task, strategy, key):
        return gen.get(task, {}).get(strategy, {}).get(key)

    bleu4_chebi_LC_random = cur("chebi-20-mol2text", "low_confidence_random", "bleu4")
    bleu4_chebi_LC_semiar = cur("chebi-20-mol2text", "low_confidence_semi_ar", "bleu4")
    bleu4_chebi_R_random = cur("chebi-20-mol2text", "random_random", "bleu4")

    f7_bleu4_random = fnum(f7sum.get("val/chebi-20-mol2text/bleu4_random"))
    f7_bleu4_semiar = fnum(f7sum.get("val/chebi-20-mol2text/bleu4_semi_ar"))
    k_bleu4_random = fnum(krow.get("val/chebi-20-mol2text/bleu4_random")) if krow else None
    k_bleu4_semiar = fnum(krow.get("val/chebi-20-mol2text/bleu4_semi_ar")) if krow else None
    kb_bleu4_random = fnum(kbest.get("val/chebi-20-mol2text/bleu4_random")) if kbest else None
    kb_bleu4_semiar = fnum(kbest.get("val/chebi-20-mol2text/bleu4_semi_ar")) if kbest else None

    lines.append(
        f"### 3-1. `chebi-20-mol2text` BLEU-4 횡단 비교\n\n"
        f"- **current version_21 (sampling_steps=32, epoch=1 end, step=8526)**:\n"
        f"  - `low_confidence × random`: {fmt(bleu4_chebi_LC_random)}\n"
        f"  - `low_confidence × semi_ar`: {fmt(bleu4_chebi_LC_semiar)}\n"
        f"  - `random × random`: {fmt(bleu4_chebi_R_random)} ← 4 strategy 중 최고\n"
        f"- **krozx9yp @ epoch=1 (sampling_steps=32, string_only, 동일 multi-task 환경)**:\n"
        f"  - `random`: {fmt(k_bleu4_random)}\n"
        f"  - `semi_ar`: {fmt(k_bleu4_semiar)}\n"
        f"- **krozx9yp BEST epoch (15 epoch 학습 ceiling)**:\n"
        f"  - `random`: {fmt(kb_bleu4_random)}\n"
        f"  - `semi_ar`: {fmt(kb_bleu4_semiar)}\n"
        f"- **7xpk8frx (sampling_steps=256, string_only, single-task mol2text, 최종)**:\n"
        f"  - `random`: {fmt(f7_bleu4_random)}\n"
        f"  - `semi_ar`: **{fmt(f7_bleu4_semiar)}** ← 사용자가 언급한 ~40 기준값\n\n"
        f"**해석**: epoch=1 기준으로 current `low_confidence × semi_ar` ({fmt(bleu4_chebi_LC_semiar)}) 는 "
        f"krozx9yp 동일 epoch ({fmt(k_bleu4_semiar)}) 대비 낮다. 다만 current의 `random × random` "
        f"({fmt(bleu4_chebi_R_random)})은 이미 krozx9yp best({fmt(kb_bleu4_semiar)} semi_ar 기준)을 초과할 수 있음. "
        f"즉 **rephrase 학습이 무효가 아니라 strategy별 품질 분포가 이동했다**.\n\n"
    )

    # Validity per strategy — the "10% 탈락" phenomenon
    lines.append("### 3-2. Validity ratio 감소 현상 (text2mol / reaction 계열)\n\n")
    lines.append("사용자 보고: 예전 validity ≈ 0.999 → 현재 10%+ 탈락. 재집계 수치 (current):\n\n")
    lines.append("| task | LC×random | LC×semi_ar | R×random | R×semi_ar | krozx9yp @ep=1 random | semi_ar | krozx9yp BEST random | semi_ar |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for task in ["chebi-20-text2mol", "smol-molecule_generation",
                 "forward_reaction_prediction", "retrosynthesis", "reagent_prediction",
                 "smol-forward_synthesis", "smol-retrosynthesis"]:
        v_lcr = cur(task, "low_confidence_random", "validity_ratio")
        v_lcs = cur(task, "low_confidence_semi_ar", "validity_ratio")
        v_rr = cur(task, "random_random", "validity_ratio")
        v_rs = cur(task, "random_semi_ar", "validity_ratio")
        k_r = fnum(krow.get(f"val/{task}/validity_ratio_random")) if krow else None
        k_s = fnum(krow.get(f"val/{task}/validity_ratio_semi_ar")) if krow else None
        kb_r = fnum(kbest.get(f"val/{task}/validity_ratio_random")) if kbest else None
        kb_s = fnum(kbest.get(f"val/{task}/validity_ratio_semi_ar")) if kbest else None
        lines.append(
            f"| {task} | {fmt(v_lcr)} | {fmt(v_lcs)} | {fmt(v_rr)} | {fmt(v_rs)} | "
            f"{fmt(k_r)} | {fmt(k_s)} | {fmt(kb_r)} | {fmt(kb_s)} |\n"
        )
    lines.append("\n")

    lines.append("### 3-3. 해석 요약\n\n")
    lines.append(
        "- `random × random` strategy는 validity가 0.99+를 유지 → 모델은 여전히 valid molecule을 생성 가능.\n"
        "- `low_confidence × *` 와 `random × semi_ar`에서 validity가 84~91% 수준으로 떨어짐 → "
        "기존 validation pipeline이 주로 사용한 `low_confidence × semi_ar` (=krozx9yp의 `semi_ar`)에서 "
        "예전 대비 큰 차이가 나올 수 있음.\n"
        "- mol2text BLEU-4는 sampling_steps 차이(32 vs 256)가 절대값 격차의 주된 원인일 가능성이 크다. "
        "7xpk8frx(256 step)에서 40.28 vs 현재(32 step)에서 12.47 → Phase 2에서 subset re-eval로 검증 예정.\n"
    )

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
