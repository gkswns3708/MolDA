#!/bin/bash
# Phase 3 — Stage 3 V-MolPO 단일 task (ChEBI captioning) 학습
#
# 목적: V-MolPO loss 가 실제 학습에서 발산 없이 도는지 검증.
#   - train/loss      0.x ~ 5 안정 (이전 MICCAI MolPO LLaDA 0.x → 60 → 1500 폭주 X)
#   - train/v_molpo/* 메트릭 정상 로깅
#   - ChEBI exact_match val metric 향상
#
# 전제:
#   1) Stage 2 ckpt 존재 (πθ 출발점 + πref 양쪽 모두에 사용)
#   2) chosen/rejected pair 가 있는 데이터셋
#      (없으면 scripts/build_molpo_dataset_synthetic.py 로 합성 가능)
#
# 실행 위치: /opt/EMNLP_MolDA/New_MolDA
#   bash scripts/train_stage3_v_molpo_chebi.sh

set -e
cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# venv 선택 — flash_attn (Q-Former의 BertEncoder 의존) 이 venvs/MolDA_stage2 에만 설치됨.
# 다른 venv 활성 상태에서 실행되어도 올바른 python 자동 선택.
PY="${PY:-./venvs/MolDA_stage2/bin/python}"
if [ ! -x "$PY" ]; then
    echo "WARN: $PY not found, falling back to system python (flash_attn 미설치 가능성)"
    PY="python"
fi

# ── 입력 ckpt ─────────────────────────────────────────
# πθ 출발점 + πref 모두 Stage 2 ckpt (사용자 결정: stage 3 = stage 2 에서 시작)
STAGE2_CKPT="${STAGE2_CKPT:-./checkpoint/selfies_dict_rephrase/stage2/last.ckpt}"

# ── GPU ──────────────────────────────────────────────
GPUS="${GPUS:-0,1,2,3,4,5}"
N_GPUS=$(echo $GPUS | tr ',' '\n' | grep -c .)

# ── 학습 옵션 ────────────────────────────────────────
N_T="${N_T:-2}"
BETA="${BETA:-0.1}"
# 기본 200 epoch — 단일 task 검증 시 trend 관찰용. 사용자가 Ctrl+C 로 manual stop.
# 짧게 돌리려면 MAX_EPOCHS=20 등 env override.
MAX_EPOCHS="${MAX_EPOCHS:-200}"
# Per-GPU forward batch. V-MolPO collator 가 mol_div=2 → 2× expansion,
# n_t=2 stack → 추가 2× → effective tensor 첫차원 = BATCH_SIZE × 4.
# RTX PRO 6000 (97GB) 환경: BATCH_SIZE=4 권장 (effective 16 per GPU, ~50GB GPU 사용).
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM="${ACCUM:-10}"
GLOBAL_BS="${GLOBAL_BS:-$((N_GPUS * BATCH_SIZE * ACCUM))}"

echo "============================================================"
echo "Phase 3 — Stage 3 V-MolPO single task (ChEBI captioning)"
echo "============================================================"
echo "  STAGE2_CKPT (πθ + πref) = $STAGE2_CKPT"
echo "  GPUs                    = $GPUS  (N_GPUS=$N_GPUS)"
echo "  batch_size (per-GPU)    = $BATCH_SIZE  (V-MolPO effective ×4 = $((BATCH_SIZE*4)))"
echo "  n_t                     = $N_T"
echo "  beta                    = $BETA"
echo "  max_epochs              = $MAX_EPOCHS"
echo "  global_batch_size       = $GLOBAL_BS  (= N_GPUS × BATCH × accum=$ACCUM)"
echo "============================================================"

if [ ! -f "$STAGE2_CKPT" ]; then
    echo "ERROR: Stage 2 ckpt not found at $STAGE2_CKPT"
    echo "  Override with: STAGE2_CKPT=/path/to/stage2.ckpt $0"
    exit 1
fi

"$PY" scripts/train.py \
    +experiment=stage3_v_molpo_chebi_only \
    trainer=stage3 \
    "hardware.devices='$GPUS'" \
    "pretrained_ckpt_path='$STAGE2_CKPT'" \
    "molpo.ref_ckpt_path='$STAGE2_CKPT'" \
    molpo.n_t=$N_T \
    molpo.beta=$BETA \
    training.batch_size=$BATCH_SIZE \
    training.max_epochs=$MAX_EPOCHS \
    training.global_batch_size=$GLOBAL_BS
