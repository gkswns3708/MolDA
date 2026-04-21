#!/usr/bin/env bash
# 3 variant full-epoch smoke: SELFIES+dict, SELFIES-dict, SMILES.
# 각 variant에 대해 toy100 fixture로 1 epoch 학습 + 전체 validation 실행.
# 성공 = training backward 완료 + validation 전체 순회 + metric 계산까지 정상 종료.
#
# Usage:
#   bash scripts/smoke_variants.sh                     # 모두 실행
#   bash scripts/smoke_variants.sh toy_SELFIES         # 단일 variant

set -euo pipefail

cd "$(dirname "$0")/.."

# Fixture 존재 체크
if [[ ! -d "dataset/Processed/toy100/Train" ]]; then
    echo "ERROR: toy100 fixture missing at dataset/Processed/toy100/Train"
    echo "       Run first: bash scripts/make_toy100_fixture.sh"
    exit 2
fi

# venv 활성화 (이미 활성화돼있으면 스킵)
if [[ -z "${VIRTUAL_ENV:-}" ]] || [[ "${VIRTUAL_ENV}" != *"MolDA"* ]]; then
    source venvs/MolDA/bin/activate
fi

VARIANTS=(
    "toy_SELFIES"
    "toy_SELFIES_no_dict"
    "toy_SMILES"
)

# 사용자가 단일 variant 지정한 경우
if [[ $# -gt 0 ]]; then
    VARIANTS=("$@")
fi

TMP_LOG_DIR="/tmp/smoke_variants_$(date +%Y%m%d_%H%M%S)"
PERSIST_LOG_DIR="logs/smoke"
mkdir -p "$TMP_LOG_DIR" "$PERSIST_LOG_DIR"

declare -A RESULTS

for cfg in "${VARIANTS[@]}"; do
    echo "============================================================"
    echo "  variant: $cfg"
    echo "============================================================"
    LOG_FILE="$TMP_LOG_DIR/${cfg}.log"

    # 1 epoch 학습 + 전체 validation (metric 계산까지), 6 GPU DDP.
    # `script`로 pseudo-TTY를 생성해 python이 isatty()==True로 판단하게 함.
    # 이 덕에 Lightning/tqdm progress bar가 정상 표시됨.
    # -q quiet(메타 메시지 억제), -e exit code 전파, -f flush after each write,
    # -c command. LOG_FILE에 터미널 출력과 동일한 내용이 저장됨.
    #
    # batch_size=16 (per GPU) × 6 GPU = global 96, accum=1
    set +e
    script -qefc "python scripts/train.py --config-name $cfg \
        training.max_epochs=1 \
        training.max_steps=-1 \
        training.batch_size=16 \
        training.global_batch_size=96 \
        validation.inference_batch_size=22 \
        validation.num_sanity_val_steps=0 \
        validation.limit_val_batches=1.0 \
        validation.check_val_every_n_epoch=1 \
        generation.sampling_steps=8 \
        'generation.val_strategies=[\"random\"]' \
        wandb.enabled=false \
        'hardware.devices=\"0,1,2,3,4,5\"'" \
        "$LOG_FILE"
    EXIT=$?
    set -e

    # 사후 감사용 영구 로그
    cp "$LOG_FILE" "$PERSIST_LOG_DIR/${cfg}.log"

    if [[ $EXIT -eq 0 ]]; then
        # metric 계산 흔적 체크
        if grep -q "computing metrics done" "$LOG_FILE" \
           && grep -q "ALL DONE (predictions saved" "$LOG_FILE" \
           && grep -q "epoch_end: START" "$LOG_FILE"; then
            RESULTS[$cfg]="PASS"
        else
            RESULTS[$cfg]="FAIL (exit=0 but metrics not computed — check $PERSIST_LOG_DIR/${cfg}.log)"
        fi
    else
        RESULTS[$cfg]="FAIL (exit=$EXIT, log: $PERSIST_LOG_DIR/${cfg}.log)"
    fi
done

echo ""
echo "============================================================"
echo "  Smoke Test Summary"
echo "  Logs persisted: $PERSIST_LOG_DIR/"
echo "============================================================"
FAIL_COUNT=0
for cfg in "${VARIANTS[@]}"; do
    status="${RESULTS[$cfg]}"
    printf "  %-25s %s\n" "$cfg" "$status"
    [[ "$status" == "PASS" ]] || FAIL_COUNT=$((FAIL_COUNT+1))
done

exit $FAIL_COUNT
