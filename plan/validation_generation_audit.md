# validation.py + generation 감사 결과

> 작성일: 2026-05-12
> 대상: `src/training/validation.py`, `src/training/metrics.py`, `src/loggers/*`, `src/generation/generate.py`
> 감사 패턴: Stage 3 audit과 동일 (Category 1 값 이상 / Category 2 중복·dead code)

총 30개 finding (CRITICAL 2 / WARN 12 / MINOR 16). 본 audit 결과 중 **CRITICAL 1개는 즉시 fix 적용 완료**, 나머지는 user 결정 대기.

---

## 즉시 fix 적용된 항목

### #1 — config default mismatch (validation.py TrainPredictionLogger)
**문제**: 
- yaml default ([trainer/default.yaml](../src/configs/trainer/default.yaml)): `train_prediction_log_interval: 1000`, `train_prediction_max_positions: 512`
- 코드 fallback ([validation.py:327-328](../src/training/validation.py#L327-L328)): `.get("...", 100)`, `.get("...", 50)`
- 영향: 정상 cfg에선 yaml 값이 우선되므로 무영향. 그러나 experiment override가 logging 섹션을 일부만 채우면 silent하게 10× 더 자주 + 1/10 짧게 로그.

**Fix**: 코드 fallback 을 yaml과 일치시킴 (1000 / 512). [validation.py:327-330](../src/training/validation.py#L327)에 주석 + 값 변경.

**검증**: pytest 신규 case 1개 (Task C에 포함) — TrainPredictionLogger init 시 default 가 yaml과 일치하는지 cfg.logging 비워서 instantiate.

---

## CRITICAL 이지만 즉시 fix 보류 (사용자 결정 대기)

### #18 — Thread-unsafe `_val_custom_chart_history` dict access
**문제**: [validation.py:161-187, 616](../src/training/validation.py)에서 async thread가 `self._val_custom_chart_history` dict 수정. 동시에 main thread가 다른 strategy 결과 추가 중이면 dict race condition 가능.

**고려사항**:
- 학습 hot path (validation phase) 직접 수정 — Stage 2 학습 중인 지금은 위험
- 실제 trigger 시나리오: multi-epoch 학습 + async 비교 chart 활성 → 현재 Stage 2는 max_epochs=1 이라 immediate risk 낮음
- 단, Stage 3 V-MolPO는 max_epochs=3~20 이라 위험 노출

**제안 fix** (다음 PR):
```python
import threading
class ValidationMixin:
    def __init__(self, ...):
        self._chart_history_lock = threading.Lock()
        self._val_custom_chart_history = {}
    
    def _update_val_chart_history(self, ...):
        with self._chart_history_lock:
            # existing logic
```

→ Stage 2 학습이 validation phase 진입하기 전 (epoch end 직전, ~15h 후) 적용 또는 학습 종료 후 처리.

---

## WARN 항목 (12개)

| # | 파일:라인 | 발견 | 제안 fix |
|---|---|---|---|
| #2 | validation.py:338-340 | `self.cfg.generation` / `self.cfg.data.gen_max_len` 직접 attribute 접근, `.get()` 없음 → cfg에 generation 없으면 AttributeError | `self.cfg.get("generation", {})` 패턴 통일 |
| #3 | generate.py:43,107 + stepwise_logger.py:47 | `mask_id=126336` 3개 파일에 hardcode | `src/training/loss.py:MASK_TOKEN_ID` 이미 존재 → 거기로 import 통일 |
| #7,8,9 | validation.py:37/108, 62/113, 95/155 | path builder / load / cleanup 의 instance vs static 메서드 중복 (3 pairs) | static 으로 통일 (async-safe), instance method 제거 |
| #13 | validation.py:147 | static method 내부 `print()` | logger 사용 |
| #14 | validation.py:381-420,500-505 | `[COV]` prefix 15+ debug print | `logger.debug()` 로 교체 또는 `MOLDA_COV_TRACE` env 보호 |
| #15 | validation.py:508-648 | async thread 안에서 print() 10+ | `logging.getLogger(__name__)` |
| #19 | validation.py:200-206,643 | async thread 가 wandb logger.experiment.log() 호출 — wandb API thread-safety 미보장 | enqueue pattern 도입 검토 |
| #20 | validation.py:338-340 | (= #2 와 동일 issue) | (= #2) |
| #21 | generate.py:70-76 | semi_ar 일 때 gen_length 가 block_size 배수가 아니면 silent하게 올림 (e.g. 128 → 256) | adjusted value 로깅 |
| #22 | validation.py:209-216 | wandb media tmpdir 사라졌을 때 silent recreate | recovery 정보 info log |

---

## MINOR 항목 (16개, 요약)

- 하드코드 magic numbers (token width 18 vs 20, separator length 64 vs 80, BLEU weights 0.25 4-tuple)
- dead helper functions (`_parse_float_tag`, `evaluate_by_task`, `_get_wandb_logger`)
- 중복 import alias (`import os as _os`), 함수 내부 import (`from collections import Counter as _C`)
- 진단 메시지 부재 (NAME_CONVERSION_TASKS skip silent, `get_task_type` 기본값 fallback, 미지정 task 무 경고)
- `hasattr` 가드 누락 (validation.py:314 `self._stepwise_logger` 가 setup 전이면 AttributeError)
- semi_ar 검증 분기 누락 (`gen_cfg.val_strategies` 와 `gen_cfg.semi_ar.enabled` 일관성 없음)
- `import torch` 가 metrics.py 함수 안쪽 (module top 으로 옮기는 게 표준)

---

## 권장 처리 순서

1. **현재 Stage 2 학습 안전 영역 (학습이 validation 진입 전)**:
   - ✅ #1 fix 완료 (이 commit)
   - #3 mask_id 상수화 — 안전 (constants 추가, import 변경)
   - #7-9 static method 통일 — 안전 (mixin 내부 helper)
   - MINOR magic number / dead code 정리

2. **Stage 2 validation phase 진입 직전 또는 학습 종료 후**:
   - #18 thread safety lock 도입
   - #2, #19, #21 generation 관련 robustness 강화

3. **별도 PR로 분리**:
   - #14, #15 print → logger refactor (큰 patch)
   - #22 wandb recovery log
   - MINOR 16개 일괄 정리

---

## Top 5 Priority Fix (audit 결과 종합)

1. **(✅ 적용)** #1 default mismatch (validation.py:327-328)
2. **(대기)** #18 thread safety (validation.py async dict)
3. **(대기)** #2 cfg.generation 직접 접근 → .get() (validation.py:338,340)
4. **(대기)** #7-9 instance/static 메서드 중복 통합
5. **(대기)** #14,15 print → logger 일괄 교체

---

## 부록: 감사 통계

- 총 finding: 30
- 카테고리: CRITICAL 2 / WARN 12 / MINOR 16
- 적용됨: 1
- 대기: 29
- 파일별 분포: validation.py 19, metrics.py 4, loggers/ 4, generate.py 3
