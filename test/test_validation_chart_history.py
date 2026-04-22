"""ValidationMixin._update_val_chart_history 동작 검증.

- (task, metric) 키에 epoch×strategy 값을 누적
- 서로 다른 길이의 strategy 시계열을 NaN padding으로 정렬
- 길이가 일관되게 유지되어 line_series 입력으로 쓰기 적합한지 확인
"""

import math

from src.training.validation import ValidationMixin


class _Stub(ValidationMixin):
    """_update_val_chart_history를 단독으로 호출하기 위한 초소형 stub."""

    def __init__(self):
        self._val_custom_chart_history = {}


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


def test_single_strategy_single_epoch():
    s = _Stub()
    s._update_val_chart_history("bace", "accuracy", "random_random", epoch=0, value=0.5)
    hist = s._val_custom_chart_history[("bace", "accuracy")]
    assert hist["epochs"] == [0]
    assert hist["strategies"] == {"random_random": [0.5]}


def test_two_strategies_same_epoch():
    s = _Stub()
    s._update_val_chart_history("bace", "accuracy", "random_random", epoch=0, value=0.5)
    s._update_val_chart_history("bace", "accuracy", "low_confidence_semi_ar", epoch=0, value=0.7)
    hist = s._val_custom_chart_history[("bace", "accuracy")]
    assert hist["epochs"] == [0]
    assert hist["strategies"]["random_random"] == [0.5]
    assert hist["strategies"]["low_confidence_semi_ar"] == [0.7]


def test_late_registering_strategy_backfills_nan():
    s = _Stub()
    s._update_val_chart_history("bace", "accuracy", "random_random", epoch=0, value=0.5)
    s._update_val_chart_history("bace", "accuracy", "random_random", epoch=1, value=0.6)
    # 새 strategy가 epoch 1에서 처음 등장 — epoch 0 자리는 NaN으로 채워져야 함
    s._update_val_chart_history("bace", "accuracy", "low_confidence_semi_ar", epoch=1, value=0.8)
    hist = s._val_custom_chart_history[("bace", "accuracy")]
    assert hist["epochs"] == [0, 1]
    assert hist["strategies"]["random_random"] == [0.5, 0.6]
    lc_vals = hist["strategies"]["low_confidence_semi_ar"]
    assert len(lc_vals) == 2
    assert _isnan(lc_vals[0])
    assert lc_vals[1] == 0.8


def test_different_tasks_keep_separate_history():
    s = _Stub()
    s._update_val_chart_history("bace", "accuracy", "random_random", epoch=0, value=0.5)
    s._update_val_chart_history("tox21", "accuracy", "random_random", epoch=0, value=0.3)
    assert ("bace", "accuracy") in s._val_custom_chart_history
    assert ("tox21", "accuracy") in s._val_custom_chart_history
    assert s._val_custom_chart_history[("tox21", "accuracy")]["strategies"]["random_random"] == [0.3]


def test_accepts_torch_tensor_value():
    import torch
    s = _Stub()
    s._update_val_chart_history("bace", "accuracy", "random_random",
                                epoch=0, value=torch.tensor(0.42))
    vals = s._val_custom_chart_history[("bace", "accuracy")]["strategies"]["random_random"]
    assert vals == [0.42] or abs(vals[0] - 0.42) < 1e-6
