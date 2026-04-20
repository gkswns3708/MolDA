"""Pytest wrapper around `validate_generated_dataset.validate_split`.

toy100 fixture에 대해 dual-column validator를 돌려 errors가 0인지 검증.
CLI로 자세한 리포트를 보려면:
    python test/validate_generated_dataset.py --data_root dataset/toy100
"""

import os

import pytest
from datasets import load_from_disk

from validate_generated_dataset import validate_split

pytestmark = pytest.mark.dataset


def _split_path(cfg, split_key: str) -> str:
    return os.path.join(cfg.data.root, cfg.data.splits[split_key])


@pytest.mark.parametrize("split_key", ["train", "val", "test"])
def test_validator_passes_on_toy100(cfg, split_key):
    path = _split_path(cfg, split_key)
    ds = load_from_disk(path)
    result = validate_split(ds, split_key, sample_limit=len(ds))
    assert not result["errors"], (
        f"[{split_key}] validator found errors:\n"
        + "\n".join(
            f"  {cat}: {msgs[:3]}"
            for cat, msgs in result["errors"].items()
        )
    )
