#!/bin/bash
# Build atomwise rejected graphs for the FULL raw_v1_10x_rephrase dataset
# (Train + Val + Test). Default = 0.55 mode (batch, strong negative).
#
# Per-task dispatch (inside `map_by_substructure_replacement_atomwise`):
#   - Reaction tasks (selfies contains `|>>|`, e.g. reagent_prediction):
#     both halves get atomwise variants for rejected + additional_rejected
#   - Text2mol-like tasks (TEXT2MOL_LIKE_TASKS = chebi-20-text2mol,
#     smol-molecule_generation): dummy [C][C][C] is used (Mol-LLM
#     TEXT2MOL_BENCHMARKS convention)
#   - Default: selfies-based atomwise
#
# Schema consistency: non-reaction rows ALSO emit additional_rejected_*
# (filled with CCC dummy) so HF Arrow writer's schema stays uniform —
# fixes the prior 99% KeyError failure.
#
# Excluded tasks: smol-name_conversion-{i2s,i2f,s2i} (Stage-1 only).
#
# Estimated time (48 procs, 0.55 mode, ~3.5M rows after exclude):
#   Train: ~2 hours
#   Val:   ~2 min
#   Test:  ~2 min
#
# Usage:
#   bash scripts/build_full_atomwise.sh
# Env overrides:
#   NUM_PROC, TARGET_FTS, NUM_REJECTED_GRAPHS, REPLACE_RATIO, MAX_ATTEMPTS,
#   SPLITS, EXCLUDE_TASKS, OUTPUT_ROOT.
# Mode shortcuts:
#   - default (TARGET_FTS=0, REPLACE_RATIO=0.3) → 0.55 mode, strong negative
#   - TARGET_FTS=0.7 REPLACE_RATIO=0.05         → 0.7 mode, Mol-LLM §B.4 intent

set -e
cd "$(dirname "$0")/.."

source venv/MolDA/bin/activate

NUM_PROC="${NUM_PROC:-48}"
TARGET_FTS="${TARGET_FTS:-0}"               # 0 = batch (0.55 mode)
NUM_REJECTED_GRAPHS="${NUM_REJECTED_GRAPHS:-6}"
REPLACE_RATIO="${REPLACE_RATIO:-0.3}"       # 0.3 for batch mode, 0.05 for fts-targeted
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"
SPLITS="${SPLITS:-Train Val Test}"
EXCLUDE_TASKS="${EXCLUDE_TASKS:-smol-name_conversion-i2s,smol-name_conversion-i2f,smol-name_conversion-s2i}"

INPUT_ROOT="dataset/Processed/raw_v1_10x_rephrase"
OUTPUT_ROOT="${OUTPUT_ROOT:-dataset/Processed/raw_v1_10x_rephrase_atomwise}"

echo "============================================================"
echo "Full atomwise build — raw_v1_10x_rephrase (all tasks)"
echo "============================================================"
echo "  input  = $INPUT_ROOT/{$SPLITS}"
echo "  output = $OUTPUT_ROOT/{$SPLITS}"
echo "  num_proc            = $NUM_PROC"
echo "  target_fts          = $TARGET_FTS  (0 = batch/0.55 mode)"
echo "  replace_ratio       = $REPLACE_RATIO"
echo "  num_rejected_graphs = $NUM_REJECTED_GRAPHS"
echo "  max_attempts        = $MAX_ATTEMPTS  (fts-targeted only)"
echo "  exclude_tasks       = $EXCLUDE_TASKS"
echo "============================================================"

for split in $SPLITS; do
    src="$INPUT_ROOT/$split"
    dst="$OUTPUT_ROOT/$split"

    if [ ! -d "$src" ]; then
        echo "[skip] $src not found"
        continue
    fi
    if [ -d "$dst" ]; then
        echo "[skip] $dst already exists (delete to rebuild)"
        continue
    fi

    echo
    echo "──────────────────────────────────────────────"
    echo "[$split] starting build at $(date)"
    echo "──────────────────────────────────────────────"

    PYTHONPATH=src python scripts/build_chebi_mol2text_atomwise.py \
        --data-dir "$src" \
        --output "$dst" \
        --task-filter "" \
        --exclude-tasks "$EXCLUDE_TASKS" \
        --num-proc "$NUM_PROC" \
        --target-fts "$TARGET_FTS" \
        --num-rejected-graphs "$NUM_REJECTED_GRAPHS" \
        --replace-ratio "$REPLACE_RATIO" \
        --max-attempts "$MAX_ATTEMPTS" \
        --skip-additional

    echo "[$split] done at $(date). output: $dst"
done

echo
echo "============================================================"
echo "All splits built. Final output: $OUTPUT_ROOT/"
echo "============================================================"
du -sh "$OUTPUT_ROOT"/* 2>/dev/null
