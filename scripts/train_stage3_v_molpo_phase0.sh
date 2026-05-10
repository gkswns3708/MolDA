#!/bin/bash
# Phase 0 — VRPO 분산 감소 가설 검증 (1 GPU, ~30분)
#
# 검증 항목:
#   Theorem 2: V[B̂(y; n_t)] ∝ 1/n_t (n_t 늘리면 분산 감소)
#   Theorem 3: V[ŝ] with shared (T,M) < V[ŝ] with independent seeds
#
# 결과 보고 후 Phase 1 (RefLLaDAWrapper, MolPO collator) 착수 결정.
#
# 실행 위치: /opt/EMNLP_MolDA/New_MolDA
#   bash scripts/train_stage3_v_molpo_phase0.sh

set -e
cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Stage 1 ckpt — π_θ 의 출발점 (보수적 선택; π_ref 는 weight perturb 로 합성)
# 사용자 환경에 맞춰 override 가능
STAGE1_CKPT="${STAGE1_CKPT:-./checkpoint/selfies_dict_rephrase/stage1/epoch=epoch=11-step=step=46686.ckpt}"

# Phase 0 grid (필요시 증감)
N_PAIRS="${N_PAIRS:-8}"
N_TRIALS="${N_TRIALS:-100}"
WEIGHT_PERTURB="${WEIGHT_PERTURB:-0.001}"
GPU="${GPU:-0}"

echo "============================================================"
echo "Phase 0 — VRPO Variance Measurement"
echo "============================================================"
echo "  STAGE1_CKPT     = $STAGE1_CKPT"
echo "  N_PAIRS         = $N_PAIRS"
echo "  N_TRIALS        = $N_TRIALS"
echo "  WEIGHT_PERTURB  = $WEIGHT_PERTURB"
echo "  GPU             = $GPU"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python scripts/measure_vrpo_variance.py \
    +experiment=selfies_dict_rephrase \
    trainer=stage1 \
    hardware.devices="'0'" \
    pretrained_ckpt_path="$STAGE1_CKPT" \
    +phase0.n_pairs=$N_PAIRS \
    +phase0.n_trials=$N_TRIALS \
    +phase0.weight_perturb=$WEIGHT_PERTURB
