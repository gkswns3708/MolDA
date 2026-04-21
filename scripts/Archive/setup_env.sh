#!/bin/bash
# =============================================================
# setup_env.sh — 새 pod에서 실행
# CUDA 버전 자동 감지 → PyTorch 설치 → venv 재구성
# =============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENVS_DIR="$PROJECT_ROOT/venvs"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

echo "============================================"
echo " New_MolDA 환경 설치 스크립트"
echo "============================================"
echo "프로젝트 루트: $PROJECT_ROOT"
echo ""

# --- Step 1: CUDA 버전 감지 ---
echo "[1/5] CUDA 버전 감지 중..."

if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found. GPU driver가 설치되어 있는지 확인하세요."
    exit 1
fi

nvidia-smi | head -4
echo ""

# nvidia-smi 출력에서 CUDA 버전 파싱
CUDA_FULL=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
CUDA_MAJOR=$(echo "$CUDA_FULL" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_FULL" | cut -d. -f2)

echo "감지된 CUDA 버전: $CUDA_FULL"

# CUDA 버전 → PyTorch index URL 매핑
if [ "$CUDA_MAJOR" -ge 13 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 6 ]); then
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
    CUDA_TAG="cu126"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    CUDA_TAG="cu124"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    CUDA_TAG="cu121"
else
    echo "[WARNING] CUDA $CUDA_FULL 은 오래된 버전입니다. cu121로 시도합니다."
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    CUDA_TAG="cu121"
fi

echo "PyTorch CUDA 빌드: $CUDA_TAG ($TORCH_INDEX)"
echo ""

# --- Step 2: Python 확인 ---
echo "[2/5] Python 확인 중..."

if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "[ERROR] $PYTHON_BIN 을 찾을 수 없습니다."
    echo "  다른 Python 경로를 사용하려면: PYTHON_BIN=/path/to/python bash setup_env.sh"
    exit 1
fi

PYTHON_VER=$("$PYTHON_BIN" --version 2>&1)
echo "Python: $PYTHON_VER ($PYTHON_BIN)"
echo ""

# --- Step 3: MolDA venv 설치 ---
echo "[3/5] MolDA venv 설치 중..."

MOLDA_REQ="$PROJECT_ROOT/requirements_molda.txt"
if [ ! -f "$MOLDA_REQ" ]; then
    echo "[ERROR] $MOLDA_REQ 을 찾을 수 없습니다. export_env.sh를 먼저 실행하세요."
    exit 1
fi

mkdir -p "$VENVS_DIR"
"$PYTHON_BIN" -m venv "$VENVS_DIR/MolDA"
source "$VENVS_DIR/MolDA/bin/activate"

echo "  PyTorch 설치 중 ($CUDA_TAG)..."
pip install --upgrade pip
pip install torch torchvision --index-url "$TORCH_INDEX"

echo "  나머지 패키지 설치 중..."
# 주석 처리된 torch 라인 제외하고 설치
grep -v '^#' "$MOLDA_REQ" | grep -v '^$' | pip install -r /dev/stdin 2>&1 || {
    echo ""
    echo "  [WARNING] 일부 패키지 설치 실패. 개별 설치를 시도합니다..."
    grep -v '^#' "$MOLDA_REQ" | grep -v '^$' | while read -r pkg; do
        pip install "$pkg" 2>/dev/null || echo "  [SKIP] $pkg"
    done
}

deactivate
echo "  MolDA venv 설치 완료."
echo ""

# --- Step 4: dataset_gen venv 설치 ---
echo "[4/5] dataset_gen venv 설치 중..."

DGEN_REQ="$PROJECT_ROOT/requirements_dataset_gen.txt"
if [ ! -f "$DGEN_REQ" ]; then
    echo "[ERROR] $DGEN_REQ 을 찾을 수 없습니다. export_env.sh를 먼저 실행하세요."
    exit 1
fi

"$PYTHON_BIN" -m venv "$VENVS_DIR/dataset_gen"
source "$VENVS_DIR/dataset_gen/bin/activate"

echo "  PyTorch 설치 중 ($CUDA_TAG)..."
pip install --upgrade pip
pip install torch --index-url "$TORCH_INDEX"

echo "  나머지 패키지 설치 중..."
grep -v '^#' "$DGEN_REQ" | grep -v '^$' | pip install -r /dev/stdin 2>&1 || {
    echo ""
    echo "  [WARNING] 일부 패키지 설치 실패. 개별 설치를 시도합니다..."
    grep -v '^#' "$DGEN_REQ" | grep -v '^$' | while read -r pkg; do
        pip install "$pkg" 2>/dev/null || echo "  [SKIP] $pkg"
    done
}

deactivate
echo "  dataset_gen venv 설치 완료."
echo ""

# --- Step 5: 검증 ---
echo "[5/5] 환경 검증 중..."
echo ""

echo "=== MolDA venv ==="
"$VENVS_DIR/MolDA/bin/python" -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
import datasets
print(f'  datasets: {datasets.__version__}')
" 2>&1 || echo "  [ERROR] MolDA 환경 검증 실패"

echo ""
echo "=== dataset_gen venv ==="
"$VENVS_DIR/dataset_gen/bin/python" -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
import datasets
print(f'  datasets: {datasets.__version__}')
" 2>&1 || echo "  [ERROR] dataset_gen 환경 검증 실패"

echo ""
echo "============================================"
echo " 설치 완료!"
echo ""
echo " 사용법:"
echo "   MolDA:       source $VENVS_DIR/MolDA/bin/activate"
echo "   dataset_gen: source $VENVS_DIR/dataset_gen/bin/activate"
echo "============================================"
