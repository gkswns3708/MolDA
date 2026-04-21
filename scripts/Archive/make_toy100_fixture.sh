#!/usr/bin/env bash
# toy100 fixture 재현 스크립트.
# 한 번 실행하면 dataset/Processed/toy100/{Train,Val,Test} 완성.
#
# 단계:
#   [1/3] BACE CSV 생성 (dataset/Raw/raw/BioT5_bace_*.csv 미존재 시)
#   [2/3] dataset_generation.run --toy 100 --num_workers ${NUM_WORKERS:-4}
#   [3/3] dataset/Raw/raw_v1_toy100/{Train,Val,Test} → dataset/Processed/toy100/
#
# 환경:
#   - venvs/dataset_gen (deepchem / rdkit / selfies / datasets 설치됨)
#
# Override:
#   NUM_WORKERS=2 bash scripts/make_toy100_fixture.sh
#
# 출력:
#   dataset/Raw/raw/BioT5_bace_*.csv     (이후 재사용, 생성됐으면 skip)
#   dataset/Raw/raw_v1_toy100/           (중간 산출물, git ignored)
#   dataset/Processed/toy100/            (최종, git tracked via .gitignore negation)

set -euo pipefail
cd "$(dirname "$0")/.."

PY="venvs/dataset_gen/bin/python"
NUM_WORKERS="${NUM_WORKERS:-4}"

RAW_BACE_TRAIN="dataset/Raw/raw/BioT5_bace_train.csv"
RAW_TOY="dataset/Raw/raw_v1_toy100"
PROCESSED="dataset/Processed/toy100"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: dataset_gen venv not found at $PY"
    echo "       setup venv first (see CLAUDE.md)"
    exit 1
fi

echo "============================================================"
echo "  toy100 fixture build (NUM_WORKERS=$NUM_WORKERS)"
echo "============================================================"

echo ""
echo "[1/3] BACE CSV"
if [[ -f "$RAW_BACE_TRAIN" ]]; then
    echo "      skip: $RAW_BACE_TRAIN already exists"
else
    "$PY" scripts/generate_bace_csv.py
fi

echo ""
echo "[2/3] dataset_generation.run --toy 100 --num_workers $NUM_WORKERS"
(
    cd src
    "../$PY" -m dataset_generation.run \
        --config both \
        --toy 100 \
        --num_workers "$NUM_WORKERS"
)

if [[ ! -d "$RAW_TOY/Train" ]]; then
    echo "ERROR: expected $RAW_TOY/Train after step 2, not found"
    exit 1
fi

echo ""
echo "[3/3] Copy to $PROCESSED (overwrite if exists)"
rm -rf "$PROCESSED"
mkdir -p "$PROCESSED"
for split in Train Val Test; do
    cp -r "$RAW_TOY/$split" "$PROCESSED/$split"
done

echo ""
echo "============================================================"
echo "  DONE"
echo "============================================================"
du -sh "$PROCESSED"
ls -la "$PROCESSED"
echo ""
echo "Next: run the variant smoke tests:"
echo "  source venvs/MolDA/bin/activate"
echo "  export PYTHONPATH=\"\$PWD:\$PWD/src\""
echo "  pytest test/test_variant_smoke.py -v -s"
