#!/bin/bash
# Stage 3 V-MolPO — chebi-20-mol2text with atomwise 0.55 mode (강한 변형) 비교 실험.
#
# vs train_stage3_v_molpo_chebi_atomwise.sh (0.7 mode, FTS≈0.66) — 통제 변수 동일.
# 차이점: 사용 데이터셋이 chebi_mol2text_atomwise_055 (mean FTS≈0.32, atom 1.4 배).
#
# Usage:
#   bash scripts/train_stage3_v_molpo_chebi_atomwise_055.sh
# Env overrides: STAGE2_CKPT, GPUS, MAX_EPOCHS, BATCH_SIZE.

set -e
cd "$(dirname "$0")/.."

source venv/MolDA/bin/activate
set -a
[ -f .env ] && . ./.env
set +a

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

STAGE2_CKPT="${STAGE2_CKPT:-./checkpoint/10xRephrase/stage2/stage2_1epoch_checkpoint.ckpt}"
GPUS="${GPUS:-0,1,2,3,4,5}"
N_GPUS=$(echo $GPUS | tr ',' '\n' | grep -c .)

MAX_EPOCHS="${MAX_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-3}"
N_T="${N_T:-2}"
BETA="${BETA:-2.0}"

ACCUM="${ACCUM:-57}"
DEFAULT_GLOBAL_BS=$((N_GPUS * BATCH_SIZE * ACCUM))
GLOBAL_BS="${GLOBAL_BS:-$DEFAULT_GLOBAL_BS}"

echo "============================================================"
echo "Stage 3 V-MolPO — chebi-20-mol2text atomwise 0.55 mode (strong negative)"
echo "============================================================"
echo "  STAGE2_CKPT             = $STAGE2_CKPT"
echo "  GPUs                    = $GPUS  (N_GPUS=$N_GPUS)"
echo "  max_epochs              = $MAX_EPOCHS"
echo "  batch_size (per-GPU)    = $BATCH_SIZE"
echo "  accum                   = $ACCUM"
echo "  global_batch_size       = $GLOBAL_BS  (= $N_GPUS × $BATCH_SIZE × $ACCUM)"
echo "  n_t                     = $N_T"
echo "  beta                    = $BETA"
echo "  dataset                 = chebi_mol2text_atomwise_055 (FTS≈0.32)"
echo "============================================================"

if [ ! -f "$STAGE2_CKPT" ]; then
    echo "ERROR: Stage 2 ckpt not found at $STAGE2_CKPT"
    exit 1
fi

python scripts/train.py \
    +experiment=stage3_v_molpo_chebi_mol2text_atomwise_055 \
    trainer=stage3 \
    "hardware.devices='$GPUS'" \
    "pretrained_ckpt_path='$STAGE2_CKPT'" \
    "molpo.ref_ckpt_path='$STAGE2_CKPT'" \
    training.max_epochs=$MAX_EPOCHS \
    training.batch_size=$BATCH_SIZE \
    training.global_batch_size=$GLOBAL_BS \
    molpo.n_t=$N_T \
    molpo.beta=$BETA
