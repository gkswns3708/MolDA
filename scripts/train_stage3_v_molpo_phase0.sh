#!/bin/bash
# Phase 0 — VRPO 분산 감소 가설 검증 (DDP, ~3-5분)
#
# 검증 항목:
#   Theorem 2: V[B̂(y; n_t)] ∝ 1/n_t (n_t 늘리면 분산 감소)
#   Theorem 3: V[ŝ] with shared (T,M) < V[ŝ] with independent seeds
#
# Stage 3 V-MolPO 는 Stage 2 ckpt 에서 시작하므로, π_θ 는 Stage 2 ckpt 로
# 로드하고 π_ref 는 동일 ckpt + 작은 weight perturb (Stage 2 → Stage 3 한 step
# 갱신을 흉내) 로 합성. 이는 학습 초반의 실제 πθ ≈ πref 상황을 모방.
#
# DDP: trial 들을 GPU 별로 분산 처리 (각 rank 가 독립 모델 복사본 보유).
# 100 trials 기준 6 GPU 면 rank 당 ~17 trials → 약 1/6 시간으로 단축.
#
# 실행 위치: /opt/EMNLP_MolDA/New_MolDA
#   bash scripts/train_stage3_v_molpo_phase0.sh

set -e
cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Stage 2 ckpt — π_θ 출발점
STAGE2_CKPT="${STAGE2_CKPT:-./checkpoint/selfies_dict_rephrase/stage2/last.ckpt}"

# GPU 설정 (default: 6 GPU). 단일 GPU 로 돌리려면 GPUS=0 nproc=1
GPUS="${GPUS:-0,1,2,3,4,5}"
NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"

# Phase 0 grid
N_PAIRS="${N_PAIRS:-8}"
N_TRIALS="${N_TRIALS:-80}"           # DDP 시 nproc 의 배수가 균등 분배에 좋음
WEIGHT_PERTURB="${WEIGHT_PERTURB:-0.001}"

echo "============================================================"
echo "Phase 0 — VRPO Variance Measurement (DDP)"
echo "============================================================"
echo "  STAGE2_CKPT     = $STAGE2_CKPT"
echo "  GPUS            = $GPUS  (nproc=$NPROC)"
echo "  N_PAIRS         = $N_PAIRS"
echo "  N_TRIALS        = $N_TRIALS"
echo "  WEIGHT_PERTURB  = $WEIGHT_PERTURB"
echo "============================================================"

if [ ! -f "$STAGE2_CKPT" ]; then
    echo "ERROR: Stage 2 ckpt not found at $STAGE2_CKPT"
    echo "  Override with: STAGE2_CKPT=/path/to/stage2.ckpt $0"
    exit 1
fi

: "${TRAINER_NAME:=stage1}"   # default: stage1 (LLM-only, no GNN/Q-Former needed for ELBO variance)
: "${EXPERIMENT_NAME:=selfies_dict_rephrase}"
# 비고: Phase 0 는 LLaDA forward 의 ELBO 분산만 측정 → GNN/Q-Former 불필요.
# stage1 trainer 사용 + stage 2 ckpt 의 LLM/LoRA 가중치를 strict=False 로 로드.
# 이는 flash_attn 미설치 환경에서 GNN 모듈 import 우회 효과도 있음.
# 정식 Phase 1+ 학습은 stage2 trainer 필요.

CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC \
    scripts/measure_vrpo_variance.py \
    "+experiment=$EXPERIMENT_NAME" \
    "trainer=$TRAINER_NAME" \
    "hardware.devices='$GPUS'" \
    "pretrained_ckpt_path='$STAGE2_CKPT'" \
    +phase0.n_pairs=$N_PAIRS \
    +phase0.n_trials=$N_TRIALS \
    +phase0.weight_perturb=$WEIGHT_PERTURB
