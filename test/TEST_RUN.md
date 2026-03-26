# MolDA 테스트 실행 가이드

## 사전 준비

```bash
# 가상환경 활성화
source /opt/11-MolDA/New_MolDA/venvs/MolDA/bin/activate

# pytest 설치 (최초 1회)
pip install pytest

# 프로젝트 루트로 이동
cd /opt/11-MolDA/New_MolDA
```

## 실행 방법

### run_tests.sh 사용 (권장)

```bash
# 실행 권한 부여 (최초 1회)
chmod +x test/run_tests.sh

# 전체 테스트 (CPU → GPU 순차 실행)
bash test/run_tests.sh all

# CPU 테스트만 (빠른 피드백, GPU 불필요)
bash test/run_tests.sh cpu

# GPU 테스트만 (모델 로딩 필요, 느림)
bash test/run_tests.sh gpu

# Slow 제외 빠른 테스트
bash test/run_tests.sh fast

# 특정 파일만 실행
bash test/run_tests.sh single test_loss.py
```

### pytest 직접 사용

```bash
export PYTHONPATH="/opt/11-MolDA/New_MolDA:$PYTHONPATH"
cd /opt/11-MolDA/New_MolDA

# 전체 실행
python -m pytest test/ -v --tb=short

# CPU 테스트만
python -m pytest test/ -m "not gpu" -v --tb=short

# GPU 테스트만
python -m pytest test/ -m "gpu" -v --tb=short

# 특정 파일
python -m pytest test/test_loss.py -v --tb=long

# 특정 테스트 클래스
python -m pytest test/test_loss.py::TestMakeNoisy -v

# 특정 테스트 함수
python -m pytest test/test_loss.py::TestMakeNoisy::test_at_least_one_masked_per_sample -v

# 실패 시 즉시 중단 (-x)
python -m pytest test/ -m "not gpu" -v --tb=short -x
```

## 테스트 구조

### CPU 테스트 (GPU 불필요, 빠름)

| 파일 | 검증 대상 | 소요 시간 |
|------|-----------|-----------|
| `test_added_tokens.py` | 특수 토큰 정의 (BOOL, FLOAT, SELFIES 등) | < 1초 |
| `test_scheduler.py` | WarmupStableDecayLRScheduler 3단계 동작 | < 1초 |
| `test_loss.py` | MaskedDiffusionLoss (make_noisy, forward, NaN 로깅) | < 2초 |
| `test_metrics.py` | 태그 파싱 + classification/regression/molecule/caption 평가 | < 3초 |
| `test_collator.py` | TrainCollator(right-pad) / EvalCollator(left-pad) | ~10초* |
| `test_dataset.py` | MoleculeDataset 로딩 + toy100 데이터 검증 | < 5초 |
| `test_datamodule.py` | MolDADataModule (DataLoader 생성 + 배치 검증) | ~15초* |
| `test_logging.py` | ValidationSampleLogger / StepwiseLogger 파일 출력 | < 2초 |

\* tokenizer 다운로드 포함 (첫 실행 시)

### GPU 테스트 (모델 로딩 필요)

| 파일 | 검증 대상 | 소요 시간 |
|------|-----------|-----------|
| `test_model.py` | LLaDAWrapper(LoRA, vocab확장) + MolDA(forward, likelihood) | ~2-3분 |
| `test_trainer.py` | MolDATrainer(optimizer 설정, checkpoint 필터링) | ~2-3분 |
| `test_e2e.py` | 전체 파이프라인: data → model → loss → backward → optimizer | ~3-5분 |

## 출력물

### 터미널 출력
```
test/test_added_tokens.py::TestAddedTokens::test_all_tokens_are_strings PASSED
test/test_added_tokens.py::TestAddedTokens::test_no_duplicate_tokens PASSED
...
============ 45 passed, 2 skipped in 120.5s ============
```

### 리포트 파일
- `test/report_cpu.xml` — CPU 테스트 JUnit XML
- `test/report_gpu.xml` — GPU 테스트 JUnit XML
- `test/test_output_cpu.log` — CPU 테스트 전체 로그
- `test/test_output_gpu.log` — GPU 테스트 전체 로그

## 문제 해결

### CUDA 없을 때
GPU 테스트는 자동으로 스킵됩니다. CPU 테스트만 실행하세요:
```bash
bash test/run_tests.sh cpu
```

### Import 에러 발생 시
PYTHONPATH 설정 확인:
```bash
export PYTHONPATH="/opt/11-MolDA/New_MolDA:$PYTHONPATH"
```

### 특정 패키지 미설치
```bash
# selfies/rdkit 관련 테스트 실패 → molecule_evaluate 테스트만 영향
pip install selfies rdkit-pypi

# nltk/rouge_score 관련 → caption_evaluate 테스트만 영향
pip install nltk rouge-score

# sklearn 관련 → classification_evaluate 테스트만 영향
pip install scikit-learn
```

### HuggingFace 캐시
토크나이저/모델은 `hf-cache/` 디렉토리에 캐시됩니다:
```bash
export HF_HOME="/opt/11-MolDA/New_MolDA/hf-cache"
```

## 테스트 마커 설명

| 마커 | 의미 | 사용법 |
|------|------|--------|
| `gpu` | CUDA GPU 필요 | `-m gpu` / `-m "not gpu"` |
| `slow` | 10초 이상 소요 (모델 로딩) | `-m "not slow"` |
| `integration` | 여러 컴포넌트 통합 테스트 | `-m integration` |
