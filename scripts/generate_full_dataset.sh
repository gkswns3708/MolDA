#!/usr/bin/env bash
# Generate the full MolDA dataset (dual-column SMILES + SELFIES).
#
# Runs dedup=on and dedup=off as two separate trees so Mol-LLM paper
# reproduction can be compared both ways (see plans/dataset-floating-lollipop.md).
#
# Outputs:
#   dataset/Raw/raw_v1/step1/                 (dedup on:  per-source Arrow)
#   dataset/Raw/raw_v1/step2/                 (dedup on:  after cross-source dedup)
#   dataset/Raw/raw_v1/.step2_done            (dedup on:  marker)
#   dataset/Processed/raw_v1/{Train,Val,Test} (dedup on:  final)
#   dataset/Raw/raw_v1_nodedup/step1/         (dedup off: per-source Arrow)
#   dataset/Processed/raw_v1_nodedup/{Train,Val,Test} (dedup off: final)
#
# Usage:
#   bash scripts/generate_full_dataset.sh              # both dedup on & off
#   DEDUP=on  bash scripts/generate_full_dataset.sh    # only dedup on
#   DEDUP=off bash scripts/generate_full_dataset.sh    # only dedup off
#   NUM_WORKERS=16 bash scripts/generate_full_dataset.sh
#
# Environment:
#   DEDUP         = on | off | both   (default: both)
#   NUM_WORKERS   = int               (default: 8)
#   CONFIG        = both | smiles | selfies (default: both — produces dual-column)
#   LOG_DIR       = path              (default: logs/dataset_gen)

set -euo pipefail

PROJECT_ROOT="/opt/EMNLP_MolDA/New_MolDA"
VENV="${PROJECT_ROOT}/venvs/MolDA"

DEDUP="${DEDUP:-both}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CONFIG="${CONFIG:-both}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/dataset_gen}"

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

run_one() {
  local mode="$1"
  local log="${LOG_DIR}/full_${CONFIG}_dedup-${mode}_${TIMESTAMP}.log"
  echo "=========================================================="
  echo " Full dataset generation — config=${CONFIG} dedup=${mode}"
  echo " workers=${NUM_WORKERS}  log=${log}"
  echo "=========================================================="
  cd "${PROJECT_ROOT}/src"
  python -m dataset_generation.run \
    --config "${CONFIG}" \
    --num_workers "${NUM_WORKERS}" \
    --dedup "${mode}" 2>&1 | tee "${log}"
  echo "[done] dedup=${mode}  → log: ${log}"
}

case "${DEDUP}" in
  on)   run_one on ;;
  off)  run_one off ;;
  both) run_one on; run_one off ;;
  *) echo "ERROR: DEDUP must be one of: on | off | both (got '${DEDUP}')"; exit 1 ;;
esac

echo
echo "All requested generations finished."
echo "Verify with:"
echo "  python -m pytest test/test_dual_column_schema.py test/test_mol_token_type_selection.py \\"
echo "    test/test_validate_generated_dataset.py -v"
