"""_log_strategy_comparison_charts가 wandb 예외에 견고한지 검증.

회귀 배경:
- wandb의 custom chart 로깅은 Table을 `/tmp/tmp*wandb-media/`에 직렬화한다.
- 해당 tempdir이 외부 cleanup / stale run 등으로 사라지면 FileNotFoundError 발생.
- 이 실패가 동일 async 경로의 prediction 저장/ JSONL cleanup을 막으면 안 된다.

검증:
1. 개별 chart `log()` 호출이 예외를 던져도 루프가 끝까지 돈다(전 chart 시도).
2. media tempdir이 없으면 호출 전에 복구된다.
"""

import os
from unittest import mock

from src.training.validation import ValidationMixin


class _Stub(ValidationMixin):
    def __init__(self):
        self._val_custom_chart_history = {
            ("bace", "accuracy"): {
                "epochs": [0, 1],
                "strategies": {
                    "random_random": [0.5, 0.6],
                    "low_confidence_semi_ar": [0.4, 0.7],
                },
            },
            ("tox21", "accuracy"): {
                "epochs": [0, 1],
                "strategies": {
                    "random_random": [0.3, 0.35],
                    "low_confidence_semi_ar": [0.2, 0.25],
                },
            },
        }


class _FakeExperiment:
    def __init__(self, tmp_dir, fail_first=False):
        self._settings = mock.Mock(_tmp_dir=tmp_dir)
        self._fail_first = fail_first
        self.calls = []

    def log(self, payload, step=None):
        self.calls.append((payload, step))
        if self._fail_first and len(self.calls) == 1:
            raise FileNotFoundError(
                "[Errno 2] No such file or directory: '/tmp/tmpXXXXwandb-media/abc.table.json'"
            )


class _FakeWandbLogger:
    def __init__(self, experiment):
        self.experiment = experiment


# 타입 이름이 "WandbLogger"여야 코드가 인식함
_FakeWandbLogger.__name__ = "WandbLogger"


def test_per_chart_exception_does_not_abort_remaining_charts():
    """첫 chart가 터져도 나머지 chart도 시도되어야 함."""
    s = _Stub()
    exp = _FakeExperiment(tmp_dir="/tmp/nonexistent_fake_wandb_dir", fail_first=True)
    lg = _FakeWandbLogger(exp)

    # 예외가 밖으로 새지 않아야 한다
    s._log_strategy_comparison_charts([lg], step=100)

    # 2개 chart 모두 log()가 호출되었어야 한다
    assert len(exp.calls) == 2


def test_missing_media_tempdir_is_recreated(tmp_path):
    """tmp_dir이 없으면 호출 전에 재생성되어야 한다."""
    missing = tmp_path / "wandb-media-gone"
    assert not missing.exists()

    s = _Stub()
    exp = _FakeExperiment(tmp_dir=str(missing), fail_first=False)
    lg = _FakeWandbLogger(exp)

    s._log_strategy_comparison_charts([lg], step=1)

    assert missing.exists() and missing.is_dir()
    assert len(exp.calls) == 2


def test_no_wandb_logger_is_noop():
    """WandbLogger가 없으면 조용히 no-op."""
    s = _Stub()

    class _Other:
        pass

    # 예외 없이 통과
    s._log_strategy_comparison_charts([_Other()], step=0)
