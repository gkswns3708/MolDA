# Dataset Generation Commands

모든 명령은 **`src/` 디렉터리**에서 실행한다.

```bash
cd /opt/EMNLP_MolDA/New_MolDA/src
source ../venvs/dataset_gen/bin/activate
```

---

## 1. Full Dataset 생성

Step 3의 `prepare_data_instance`가 한 번의 실행에서 SMILES/SELFIES 컬럼을 함께 생성하므로
단일 command로 충분하다. (현재 `configs/download/{smiles,selfies,both}.yaml` 세 파일은 내용이 동일하다.)

```bash
python -m dataset_generation.run --config both --num_workers 16
```

- Config: `configs/download/both.yaml`
- 출력: `dataset/Raw/raw_v1/{Train,Val,Test}/` (representation별 하위 폴더 없음)

---

## 2. Toy Mode (빠른 검증)

각 task별 N개만 sampling하여 축소 데이터셋으로 전체 파이프라인을 실행한다.

```bash
# task별 100개
python -m dataset_generation.run --config both --toy 100

# task별 5개
python -m dataset_generation.run --config both --toy 5
```

- 출력: `dataset/Raw/raw_v1_toy100/{Train,Val,Test}/` (data_tag에 `_toyN` 접미사 자동 추가)

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
cd /opt/EMNLP_MolDA/New_MolDA
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
         decontaminated Arrow 파일 → split별 concat → 프롬프트 포맷팅
         (SMILES/SELFIES 양쪽 컬럼 동시 생성) → SELFIES 변환 실패 샘플 필터링
         → 최종 저장
```

---

## 6. 출력 파일 구조

```
dataset/Raw/{data_tag}/
├── {task}_subtask-{idx}_train/   # Step 1 개별 Arrow (decontaminated after Step 2)
├── {task}_subtask-{idx}_val/
├── {task}_subtask-{idx}_test/
├── Train/                        # Step 3 최종 (전체 task concat + prompt 포맷팅)
├── Val/
└── Test/
```
