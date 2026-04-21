#!/bin/bash
# =============================================================
# export_env.sh — 현재 pod에서 실행
# New_MolDA 프로젝트를 새 pod으로 이전하기 위한 패킹 스크립트
# =============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${1:-$(dirname "$PROJECT_ROOT")}"
ARCHIVE_NAME="New_MolDA_project.tar.gz"

echo "============================================"
echo " New_MolDA 환경 추출 스크립트"
echo "============================================"
echo "프로젝트 루트: $PROJECT_ROOT"
echo "출력 디렉토리: $OUTPUT_DIR"
echo ""

# --- Step 1: requirements 추출 ---
echo "[1/3] requirements 추출 중..."

MOLDA_VENV="$PROJECT_ROOT/venvs/MolDA"
DGEN_VENV="$PROJECT_ROOT/venvs/dataset_gen"

GPU_FILTER="^nvidia|^cuda-bindings|^cuda-pathfinder|^cuda-toolkit"

if [ -x "$MOLDA_VENV/bin/pip" ]; then
    "$MOLDA_VENV/bin/pip" freeze | grep -ivE "$GPU_FILTER" > "$PROJECT_ROOT/requirements_molda.txt"
    echo "  MolDA: $(wc -l < "$PROJECT_ROOT/requirements_molda.txt") packages → requirements_molda.txt"
else
    echo "  [WARNING] MolDA venv not found at $MOLDA_VENV"
fi

if [ -x "$DGEN_VENV/bin/pip" ]; then
    "$DGEN_VENV/bin/pip" freeze | grep -ivE "$GPU_FILTER" > "$PROJECT_ROOT/requirements_dataset_gen.txt"
    echo "  dataset_gen: $(wc -l < "$PROJECT_ROOT/requirements_dataset_gen.txt") packages → requirements_dataset_gen.txt"
else
    echo "  [WARNING] dataset_gen venv not found at $DGEN_VENV"
fi

# --- Step 2: torch 버전 라인에 주석 추가 ---
echo ""
echo "[2/3] torch CUDA 빌드 태그 처리 중..."

for req_file in "$PROJECT_ROOT/requirements_molda.txt" "$PROJECT_ROOT/requirements_dataset_gen.txt"; do
    if [ -f "$req_file" ]; then
        # torch==X.Y.Z+cuXXX → 주석 처리 (새 pod에서 CUDA 맞춰 재설치)
        sed -i 's/^\(torch==.*+cu[0-9]*\)$/# \1  # CUDA-specific build — reinstalled by setup_env.sh/' "$req_file"
        sed -i 's/^\(torchvision==.*\)$/# \1  # reinstalled with torch/' "$req_file"
        echo "  처리 완료: $(basename "$req_file")"
    fi
done

# --- Step 3: tar.gz 패킹 (src + configs + scripts + requirements만) ---
echo ""
echo "[3/3] 프로젝트 패킹 중 (src/scripts/docs/requirements만)..."

PARENT_DIR="$(dirname "$PROJECT_ROOT")"
PROJECT_DIRNAME="$(basename "$PROJECT_ROOT")"

tar czf "$OUTPUT_DIR/$ARCHIVE_NAME" \
    -C "$PARENT_DIR" \
    "$PROJECT_DIRNAME/src/" \
    "$PROJECT_DIRNAME/scripts/" \
    "$PROJECT_DIRNAME/docs/" \
    "$PROJECT_DIRNAME/test/" \
    "$PROJECT_DIRNAME/CLAUDE.md" \
    "$PROJECT_DIRNAME/requirements_molda.txt" \
    "$PROJECT_DIRNAME/requirements_dataset_gen.txt"

ARCHIVE_SIZE=$(du -h "$OUTPUT_DIR/$ARCHIVE_NAME" | cut -f1)
echo "  생성 완료: $OUTPUT_DIR/$ARCHIVE_NAME ($ARCHIVE_SIZE)"

echo ""
echo "============================================"
echo " 완료! 다음 단계:"
echo "  1. $ARCHIVE_NAME 을 새 pod으로 복사"
echo "  2. 새 pod에서 tar xzf $ARCHIVE_NAME"
echo "  3. cd $PROJECT_DIRNAME && bash scripts/setup_env.sh"
echo ""
echo " [참고] dataset, checkpoint, hf-cache는 제외됨"
echo "  → 새 pod에서 직접 다운로드/생성 필요"
echo "============================================"
