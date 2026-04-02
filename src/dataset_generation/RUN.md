# Dataset Generation Commands

모든 명령은 **`src/` 디렉터리**에서 실행한다.

```bash
cd /opt/11-MolDA/New_MolDA/src
source ../venvs/dataset_gen/bin/activate
```

---

## 1. Full Dataset 생성

### SMILES 표현

```bash
python -m dataset_generation.run --config smiles --num_workers 16
```

- Config: `configs/download/smiles.yaml`
- 출력: `dataset/Raw/SMILES/SMILES_raw_v1/{Train,Val,Test}/`

### SELFIES 표현

```bash
python -m dataset_generation.run --config selfies --num_workers 16
```

- Config: `configs/download/selfies.yaml`
- 출력: `dataset/Raw/SELFIES/SELFIES_raw_v1/{Train,Val,Test}/`

### SMILES + SELFIES 동시 생성 (both)

```bash
python -m dataset_generation.run --config both --num_workers 16
```

- Config: `configs/download/both.yaml`
- `mol_representation: ['selfies', 'smiles']` → 각 representation에 대해 순차 실행
- 출력: `SELFIES/SELFIES_raw_v1/` + `SMILES/SMILES_raw_v1/` 양쪽 모두 생성

---

## 2. Toy Mode (빠른 검증)

각 task별 N개만 sampling하여 축소 데이터셋으로 전체 파이프라인을 실행한다.

```bash
# SMILES, task별 100개
python -m dataset_generation.run --config smiles --toy 100

# SELFIES, task별 5개
python -m dataset_generation.run --config selfies --toy 5

# Both, task별 100개
python -m dataset_generation.run --config both --toy 100
```

- 출력: `dataset/Raw/SMILES/SMILES_raw_v1_toy100/{Train,Val,Test}/` (data_tag에 `_toyN` 접미사 자동 추가)

---

## 3. 옵션 정리

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--config` | `smiles` | `configs/download/` 내 YAML 파일명 (확장자 제외). 공백 구분으로 여러 개 가능 |
| `--config_dir` | `configs/download/` | config 디렉터리 경로 |
| `--num_workers` | `0` (auto) | 병렬 처리 worker 수. 0이면 `os.cpu_count()` |
| `--toy` | `None` (OFF) | task별 sampling 수. 지정 시 toy mode 활성화 |

---

## 4. 테스트 실행

```bash
cd /opt/11-MolDA/New_MolDA
pytest test/test_dedup.py -v
```

---

## 5. 파이프라인 단계 설명

```
Step 1 : Download & Process Individual Tasks
         각 소스(SMolInstruct, Mol-Instructions, ChEBI-20, DeepChem, BACE)의
         source-native split을 그대로 사용하여 개별 Arrow 파일로 저장

Step 2 : Cross-Source Decontamination
         1) build_eval_blacklist  — eval boundary(test+val)에서 entity key 수집
         2) remove_eval_leakage  — train에서 eval blacklist entity 제거
         3) dedup_within_family  — 같은 family 내 cross-source 중복 제거
         4) contamination audit report 출력

Step 3 : Concatenate & Map
         decontaminated Arrow 파일 → split별 concat → 프롬프트 포맷팅 → 최종 저장
```

---

## 6. 출력 파일 구조

```
dataset/Raw/{REPR}/{data_tag}/
├── {task}_subtask-{idx}_train/   # Step 1 개별 Arrow (decontaminated after Step 2)
├── {task}_subtask-{idx}_val/
├── {task}_subtask-{idx}_test/
├── Train/                        # Step 3 최종 (전체 task concat + prompt 포맷팅)
├── Val/
└── Test/
```
