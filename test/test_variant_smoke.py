"""3 variant full-epoch smoke test (GPU 필요).

각 Hydra config가 toy100 fixture로 **1 epoch 학습 + 전체 validation**까지 돌고
metric 계산이 실제로 수행되는지 검증.

PASS 기준:
- train.py가 exit 0으로 종료
- 로그에 training backward / full-epoch 진행 흔적
- on_validation_epoch_end 진입 + metric 계산 완료 + predictions JSON 저장

Variants:
- toy_SELFIES          : mol_token_type=selfies, add_mol_dict=true  (SELFIES vocab 등록)
- toy_SELFIES_no_dict  : mol_token_type=selfies, add_mol_dict=false (BPE 분해)
- toy_SMILES           : mol_token_type=smiles,  add_mol_dict=false (SMILES는 dict N/A)

로그는 `logs/smoke/{variant}.log`에 영구 저장되어 사후 감사 가능.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.dataset, pytest.mark.slow]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
FIXTURE_TRAIN = PROJECT_ROOT / "dataset" / "Processed" / "toy100" / "Train"
LOG_DIR = PROJECT_ROOT / "logs" / "smoke"

VARIANTS = [
    "toy_SELFIES",
    "toy_SELFIES_no_dict",
    "toy_SMILES",
]

# 로그에 반드시 존재해야 할 문자열 → 사용자 PASS 기준 증거.
# 하나라도 빠지면 exit 0이라도 실패 처리.
REQUIRED_MARKERS = [
    ("train/loss",                "training_step 미도달 — backward 경로 안 돎"),
    ("Epoch 0",                   "1 epoch 시작 흔적 없음"),
    ("epoch_end: START",          "on_validation_epoch_end 미진입"),
    ("computing metrics done",    "metric 계산 async 완료 로그 없음"),
    ("ALL DONE (predictions saved", "predictions JSON 저장 완료 로그 없음"),
]


def _require_fixture():
    """toy100 fixture 없으면 친절한 skip 메시지로 종료."""
    if not FIXTURE_TRAIN.exists():
        pytest.skip(
            f"toy100 fixture missing at {FIXTURE_TRAIN}\n"
            f"Run: bash scripts/Archive/make_toy100_fixture.sh"
        )


@pytest.fixture(scope="module")
def venv_python():
    """Prefer MolDA venv Python; fall back to current interpreter."""
    candidate = PROJECT_ROOT / "venvs" / "MolDA" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_smoke(variant, venv_python):
    """1 epoch 학습 + 전체 validation + metric 계산까지 정상 종료 검증."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    _require_fixture()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{variant}.log"

    # 6 GPU DDP: batch_size=16 × 6 = global 96, accum=1
    cmd = [
        venv_python,
        str(TRAIN_SCRIPT),
        "--config-name", variant,
        # 학습: toy100 1 epoch 전체
        "training.max_epochs=1",
        "training.max_steps=-1",
        "training.batch_size=16",
        "training.global_batch_size=96",
        # Validation: 전체 task 순회
        "validation.inference_batch_size=22",
        "validation.num_sanity_val_steps=0",
        "validation.limit_val_batches=1.0",
        "validation.check_val_every_n_epoch=1",
        # Diffusion generation: low step
        "generation.sampling_steps=8",
        'generation.val_strategies=["random"]',
        "wandb.enabled=false",
        'hardware.devices="0,1,2,3,4,5"',
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:" + env.get("PYTHONPATH", "")

    with open(log_file, "w") as f:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), env=env,
            stdout=f, stderr=subprocess.STDOUT,
            timeout=3600,  # 1시간
        )

    log_text = log_file.read_text(errors="replace")

    # 1) exit code 체크 — 실패 시 tail 60줄 노출
    if proc.returncode != 0:
        tail = "\n".join(log_text.splitlines()[-60:])
        pytest.fail(
            f"[{variant}] train.py exited with {proc.returncode}.\n"
            f"Full log: {log_file}\n"
            f"--- last 60 log lines ---\n{tail}"
        )

    # 2) metric 산출 흔적 체크 — 사용자 PASS 기준
    missing = [msg for marker, msg in REQUIRED_MARKERS if marker not in log_text]
    if missing:
        tail = "\n".join(log_text.splitlines()[-60:])
        pytest.fail(
            f"[{variant}] exit 0이지만 PASS 기준 미충족:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + f"\nFull log: {log_file}\n"
            f"--- last 60 log lines ---\n{tail}"
        )
