#!/bin/bash
# Stage 3 V-MolPO — chebi-20-mol2text with atomwise graph-rejection (2 epoch test).
#
# Mirrors cpjreoz6 wandb run config (project: MolDA_pro6000), with graph rejection
# now wired to consume {i}-th_rejected_* keys from chebi_mol2text_atomwise dataset.
#
# Usage:
#   bash scripts/train_stage3_v_molpo_chebi_atomwise.sh
# Env overrides: STAGE2_CKPT, GPUS, MAX_EPOCHS, BATCH_SIZE.

set -e
cd "$(dirname "$0")/.."

source venv/MolDA/bin/activate
set -a
[ -f .env ] && . ./.env
set +a

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

STAGE2_CKPT="${STAGE2_CKPT:-./checkpoint/10xRephrase/stage2/last.ckpt}"
GPUS="${GPUS:-0,1,2,3,4,5}"
N_GPUS=$(echo $GPUS | tr ',' '\n' | grep -c .)

MAX_EPOCHS="${MAX_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-3}"
N_T="${N_T:-2}"
BETA="${BETA:-2.0}"

# Lightning global_batch_size must equal N_GPUS × batch_size × accum.
# 6 GPU × batch=3 × accum=57 = 1026.
# batch_division=2 (yaml) with FREE SFT: SFT loss is computed from the chosen
# half of the pair forward (Old_MolDA / mol-llm_official pattern) — no extra
# forward, ~87 GB peak (wivo budget). Same memory + same speed as wivo, but
# now L_total = sft_w·L_SFT + molpo_w·L_pref is active.
ACCUM="${ACCUM:-57}"
DEFAULT_GLOBAL_BS=$((N_GPUS * BATCH_SIZE * ACCUM))
GLOBAL_BS="${GLOBAL_BS:-$DEFAULT_GLOBAL_BS}"

echo "============================================================"
echo "Stage 3 V-MolPO — chebi-20-mol2text atomwise (graph-rejection)"
echo "============================================================"
echo "  STAGE2_CKPT             = $STAGE2_CKPT"
echo "  GPUs                    = $GPUS  (N_GPUS=$N_GPUS)"
echo "  max_epochs              = $MAX_EPOCHS"
echo "  batch_size (per-GPU)    = $BATCH_SIZE"
echo "  accum                   = $ACCUM"
echo "  global_batch_size       = $GLOBAL_BS  (= $N_GPUS × $BATCH_SIZE × $ACCUM)"
echo "  n_t                     = $N_T"
echo "  beta                    = $BETA"
echo "============================================================"

if [ ! -f "$STAGE2_CKPT" ]; then
    echo "ERROR: Stage 2 ckpt not found at $STAGE2_CKPT"
    exit 1
fi

python scripts/train.py \
    +experiment=stage3_v_molpo_chebi_mol2text_atomwise \
    trainer=stage3 \
    "hardware.devices='$GPUS'" \
    "pretrained_ckpt_path='$STAGE2_CKPT'" \
    "molpo.ref_ckpt_path='$STAGE2_CKPT'" \
    training.max_epochs=$MAX_EPOCHS \
    training.batch_size=$BATCH_SIZE \
    training.global_batch_size=$GLOBAL_BS \
    molpo.n_t=$N_T \
    molpo.beta=$BETA
