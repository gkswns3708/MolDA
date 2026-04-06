# MolDA — Claude Code Context

## 프로젝트 개요
MolDA는 **LLaDA**(Masked Diffusion LM) 백본 위에 GNN(GINE + TokenGT) + Q-Former를 결합하여
분자 성질 예측 / 분자 생성 등 다양한 분자 task를 수행하는 multi-modal 모델이다.
Mol-LLM의 아키텍처 + 학습 방법론을 LLaDA backbone으로 대체한 구조.

- 백본 LLM: `GSAI-ML/LLaDA-8B-Instruct`
- 분자 표현: SMILES + Graph (GINE & TokenGT concat → Q-Former → LLM)
- 프레임워크: PyTorch Lightning + Hydra + HuggingFace datasets
- 참고 논문: `/opt/11-MolDA/Reference/` 아래 LLaDA, Mol-LLM 논문

## 프로젝트 루트
`/opt/11-MolDA/New_MolDA/`

## 기존 코드 (문제가 있어 리팩토링 중)
`/opt/11-MolDA/Old_MolDA/` — 구현 참고용으로 사용. 직접 수정하지 않음.

## 목표 디렉터리 구조
```
New_MolDA/
├── CLAUDE.md
├── dataset/
│   ├── Train_toy100/
│   ├── Val_toy100/
│   └── Test_toy100/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASET_SPEC.md
│   └── STAGE_TRAINING.md
├── src/
│   ├── configs/
│   │   ├── config.yaml           # Hydra top-level (defaults: data, gnn, trainer)
│   │   ├── data/                 # tag 기반 버전 관리
│   │   │   └── toy100.yaml
│   │   ├── gnn/
│   │   │   └── gine_tokengt.yaml
│   │   ├── trainer/
│   │   │   ├── default.yaml      # 공통 hyperparameter
│   │   │   ├── stage1.yaml
│   │   │   ├── stage2.yaml
│   │   │   └── stage3.yaml
│   │   └── download/             # 데이터 생성 config
│   ├── official_LLaDA/       # 공식 LLaDA repo 클론 (수정 금지)
│   ├── data/
│   │   ├── dataset.py
│   │   ├── datamodule.py
│   │   └── collator.py
│   ├── model/
│   │   ├── molda.py          # 통합 모델 (stage별 freeze 정책)
│   │   ├── llada_wrapper.py  # LLaDA + LoRA
│   │   ├── qformer.py        # Blip2QFormer (5 layers, 32 queries)
│   │   ├── gnn.py            # GINE + TokenGT concat
│   │   └── loss.py           # masked diffusion loss + NaN 추적
│   ├── training/
│   │   ├── trainer.py        # LightningModule
│   │   ├── metrics.py        # 기존 help_funcs.py 대체
│   │   └── scheduler.py      # WarmupStableDecayLR
│   └── generation/
│       └── generate.py       # LLaDA generate 래핑 (공식 코드 수정 금지)
├── scripts/
│   ├── train.py              # Hydra entry point
│   └── run_infer_llada_official.py
└── tests/
    ├── test_dataset.py
    ├── test_model.py
    └── test_ddp.py
```

## 현재 구현 상태 (2026-03-27 기준, v0.2.0 완료)
- [x] `src/official_LLaDA/` — 수정 없이 유지
- [x] `scripts/run_infer_llada_official.py`
- [x] Toy dataset 3-split (`dataset/*_toy100/`)
- [x] `src/configs/` — Hydra config (default, stage1/2/3, toy100, gnn, test)
- [x] `src/data/` — Dataset, DataModule, TrainCollator, EvalCollator
- [x] `src/model/` — MolDA, LLaDAWrapper (LoRA + vocab expansion), GNN/QFormer stubs
- [x] `src/training/` — MolDATrainer, MaskedDiffusionLoss, Metrics, WSD Scheduler
- [x] `src/generation/` — generate wrapper + generate_with_logging
- [x] `src/loggers/` — ValidationSampleLogger, StepwiseLogger
- [x] `scripts/train.py` — Hydra entry point
- [x] `test/` — 13 test files (unit + integration + e2e)

## 가상환경 (venv) 사용 규칙
- **Dataset 생성 및 preprocessing**: `/opt/11-MolDA/New_MolDA/venvs/dataset_gen` 사용
  - 예: `dataset_generator.py` 실행, 데이터 전처리 스크립트 등
  - `datasets==2.16.1` 환경 (HF datasets Arrow 포맷 호환성)
- **그 외 모든 작업 (학습, 추론, 테스트 등)**: `/opt/11-MolDA/New_MolDA/venvs/MolDA` 사용

## 핵심 설계 원칙
1. **`src/official_LLaDA/generate.py` 수정 금지.** 래핑만 허용.
2. **Dataset 버전은 Hydra `data` config의 `tag`로 관리.**
3. **GNN: GINE와 TokenGT를 각각 forward → sequence dim concat → Q-Former 입력.**
4. **DDP: DataLoader `drop_last=True` 필수. `find_unused_parameters=False` (각 stage 별도 실행 + freeze 정책으로 불필요).**
5. **Train collator = right padding(EOS), Eval collator = left padding(PAD).**
6. **Loss NaN 발생 시 sample index, input_ids, task를 log에 기록.**
7. **Optimizer 5 param groups: LoRA / embed_orig / embed_new / head_orig+new / Q-Former+GNN.**
8. **LLaDA 레이어 이름: embedding=`wte`, lm_head=`ff_out` (표준 LLaMA와 다름). weight_tying=True로 override하여 wte가 output에도 사용됨.**
9. **파이프라인 로직 변경 시 반드시 관련 테스트 코드를 `tests/`에 작성/업데이트한다.** (상세: `.claude/rules/testing.md`)

## 참고 구현 (Old_MolDA)
필요 시 아래 파일을 참고:
- `Old_MolDA/model/gine_tokengt.py` — GINE+TokenGT concat 구현
- `Old_MolDA/model/blip2_llada.py` — forward/generate 구현
- `Old_MolDA/model/blip2_stage3.py` — LightningModule
- `Old_MolDA/data_utils.py` — DataCollator
- `Old_MolDA/configs/` — Hydra config 구조

## TODO
- **Per-task loss logging 병목 모니터링**: `trainer.py`의 `training_step`에서 per-task loss/loss_no_eos를 wandb에 로깅 중. 현재는 batch 내 unique task 수만큼 `self.log()` 호출 + `torch.tensor()` boolean mask 생성. Task 종류가 많아지거나 batch_size가 커지면 병목이 될 수 있음. 병목 확인 시 → (1) `self.log()` 호출을 줄이기 위해 dict 기반 batch logging, (2) boolean mask를 미리 collator에서 계산, (3) N step마다만 per-task 로깅 등의 최적화 적용.

## 세부 문서
- 아키텍처 상세 → `docs/ARCHITECTURE.md`
- Dataset 스키마 → `docs/DATASET_SPEC.md`
- Stage별 학습 설정 → `docs/STAGE_TRAINING.md`
- 테스트 파일 가이드 → `docs/TESTING.md`
- **소스 코드 점검 사항 → `docs/KNOWN_ISSUES.md`** — 테스트에서 발견된 불일치/의심 항목. 전체 수정·리팩토링 시 반드시 참고하여 해당 소스 코드를 확인할 것.
