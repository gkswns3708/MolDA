#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/opt/11-MolDA/New_MolDA"
VENV="${PROJECT_ROOT}/venvs/MolDA"
TEST_DIR="${PROJECT_ROOT}/test"

# Activate venv
source "${VENV}/bin/activate"

# Ensure project root in PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

echo "=============================================="
echo "MolDA Test Suite Runner"
echo "=============================================="
echo "Python : $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPUs   : $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=============================================="

RUN_MODE="${1:-all}"

case "$RUN_MODE" in
    cpu)
        echo "[Mode] CPU-only tests (skipping GPU tests)"
        python -m pytest "${TEST_DIR}" -m "not gpu" -v --tb=short \
            --junitxml="${TEST_DIR}/report_cpu.xml" \
            2>&1 | tee "${TEST_DIR}/test_output_cpu.log"
        ;;
    gpu)
        echo "[Mode] GPU tests only"
        python -m pytest "${TEST_DIR}" -m "gpu" -v --tb=short \
            --junitxml="${TEST_DIR}/report_gpu.xml" \
            2>&1 | tee "${TEST_DIR}/test_output_gpu.log"
        ;;
    fast)
        echo "[Mode] Fast tests only (no slow)"
        python -m pytest "${TEST_DIR}" -m "not slow" -v --tb=short \
            --junitxml="${TEST_DIR}/report_fast.xml" \
            2>&1 | tee "${TEST_DIR}/test_output_fast.log"
        ;;
    all)
        echo "[Mode] All tests (CPU → GPU → Diagnostic Report)"
        echo ""
        echo "── Phase 1: CPU-only tests ──"
        python -m pytest "${TEST_DIR}" -m "not gpu" -v --tb=short \
            --junitxml="${TEST_DIR}/report_cpu.xml" \
            2>&1 | tee "${TEST_DIR}/test_output_cpu.log"
        CPU_EXIT=${PIPESTATUS[0]}

        echo ""
        echo "── Phase 2: GPU tests ──"
        python -m pytest "${TEST_DIR}" -m "gpu" -v --tb=short \
            --junitxml="${TEST_DIR}/report_gpu.xml" \
            2>&1 | tee "${TEST_DIR}/test_output_gpu.log"
        GPU_EXIT=${PIPESTATUS[0]}

        echo ""
        echo "── Phase 3: Diagnostic Report ──"
        python "${TEST_DIR}/test_diagnostic_report.py" \
            2>&1 | tee "${TEST_DIR}/test_output_diagnostic.log"
        DIAG_EXIT=${PIPESTATUS[0]}

        echo ""
        echo "=============================================="
        echo "Summary:"
        echo "  CPU tests exit code:        ${CPU_EXIT}"
        echo "  GPU tests exit code:        ${GPU_EXIT}"
        echo "  Diagnostic report exit code: ${DIAG_EXIT}"
        echo "=============================================="

        [ ${CPU_EXIT} -eq 0 ] && [ ${GPU_EXIT} -eq 0 ] && [ ${DIAG_EXIT} -eq 0 ]
        ;;
    single)
        # Run a single test file: ./run_tests.sh single test_loss.py
        TEST_FILE="${2:-}"
        if [ -z "$TEST_FILE" ]; then
            echo "Usage: $0 single <test_file.py>"
            exit 1
        fi
        echo "[Mode] Single file: ${TEST_FILE}"
        python -m pytest "${TEST_DIR}/${TEST_FILE}" -v --tb=long \
            2>&1 | tee "${TEST_DIR}/test_output_single.log"
        ;;
    *)
        echo "Usage: $0 [cpu|gpu|fast|all|single <file>]"
        echo ""
        echo "Modes:"
        echo "  cpu    - CPU-only tests (no GPU required)"
        echo "  gpu    - GPU tests only (model loading)"
        echo "  fast   - All non-slow tests"
        echo "  all    - Full suite: CPU first, then GPU"
        echo "  single - Run a single test file"
        exit 1
        ;;
esac

echo ""
echo "Reports: ${TEST_DIR}/report_*.xml"
echo "Logs:    ${TEST_DIR}/test_output_*.log"
