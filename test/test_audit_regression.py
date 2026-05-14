"""Regression tests catching the two CRITICAL findings from the audit on
   /opt/MolDA/plan/{stage3_audit_summary, validation_generation_audit}.md.

   - test_train_prediction_default_matches_yaml: ValidationMixin's TrainPredictionLogger
     fallback defaults must match trainer/default.yaml so partial cfg overrides
     don't silently log 10× more often (audit #1).
   - test_v_molpo_ema_persists_through_checkpoint_hooks: trainer-level integration
     of TaskAnchorEMA serialization across on_save_checkpoint / on_load_checkpoint.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.training.v_molpo_loss import TaskAnchorEMA


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────
# Audit finding #1: config default mismatch
# ─────────────────────────────────────────────

def _parse_yaml_int(path: Path, key: str) -> int:
    """Lightweight scalar-int extractor — avoids YAML lib for a plain repo file."""
    text = path.read_text()
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(-?\d+)\s*(#.*)?$", text, re.MULTILINE)
    assert m, f"Key '{key}' not found as scalar int in {path}"
    return int(m.group(1))


def _parse_code_default(path: Path, key: str) -> int:
    """Extract the integer literal default inside cfg.logging.get('<key>', <int>)."""
    text = path.read_text()
    pattern = rf'\.get\(\s*"{re.escape(key)}"\s*,\s*(-?\d+)\s*\)'
    m = re.search(pattern, text)
    assert m, f".get('{key}', <int>) not found in {path}"
    return int(m.group(1))


class TestConfigDefaultsMatchYaml:
    """The .get() fallback defaults inside ValidationMixin.setup must match
    trainer/default.yaml so a partial logging override doesn't silently
    revert to wrong defaults."""

    YAML_PATH = REPO_ROOT / "src/configs/trainer/default.yaml"
    CODE_PATH = REPO_ROOT / "src/training/validation.py"

    @pytest.mark.parametrize("key", [
        "train_prediction_log_interval",
        "train_prediction_max_positions",
    ])
    def test_default_matches_yaml(self, key):
        yaml_v = _parse_yaml_int(self.YAML_PATH, key)
        code_v = _parse_code_default(self.CODE_PATH, key)
        assert yaml_v == code_v, (
            f"Defensive default drift: cfg.logging.{key} yaml={yaml_v} "
            f"but validation.py .get fallback={code_v}. "
            f"Partial cfg overrides will silently use {code_v}."
        )


# ─────────────────────────────────────────────
# Trainer-integration EMA round-trip (CPU)
# ─────────────────────────────────────────────

class _StubTrainerForCheckpoint:
    """Minimal stand-in for MolDATrainer to exercise CheckpointMixin EMA hooks
    without loading LLaDA-8B. The mixin only calls getattr(self.model,
    'task_anchor_ema'), reads .state_dict() / .load_state_dict()."""

    def __init__(self, ema: TaskAnchorEMA):
        self.model = SimpleNamespace(task_anchor_ema=ema)


def test_v_molpo_ema_persists_through_checkpoint_hooks():
    import torch
    from src.training.checkpoint import CheckpointMixin

    # 1. Setup: EMA with task entries
    ema = TaskAnchorEMA(alpha=0.99)
    ema.update(torch.tensor([0.5, -0.3, 1.0]), tasks=["t1", "t2", "t1"])
    snap_before = dict(ema.state_dict())

    # 2. on_save_checkpoint puts EMA state into the checkpoint dict
    saver = _StubTrainerForCheckpoint(ema)
    checkpoint = {"state_dict": {}}  # ref_model.* filter path is no-op here
    CheckpointMixin.on_save_checkpoint(saver, checkpoint)
    assert "v_molpo_task_anchor_ema" in checkpoint, (
        "on_save_checkpoint failed to persist task_anchor_ema state"
    )

    # 3. Fresh EMA + on_load_checkpoint restores it (epoch-cross persistence)
    fresh_ema = TaskAnchorEMA(alpha=0.99)
    assert len(fresh_ema) == 0
    loader = _StubTrainerForCheckpoint(fresh_ema)
    CheckpointMixin.on_load_checkpoint(loader, checkpoint)

    snap_after = dict(fresh_ema.state_dict())
    assert snap_after == snap_before, (
        f"EMA state mismatch after round-trip:\n"
        f"  before: {snap_before}\n  after:  {snap_after}"
    )


def test_checkpoint_ref_model_filter_drops_ref_keys():
    """on_save_checkpoint must drop model.ref_model.* keys — they are reloadable
    from ref_ckpt_path and shipping them inflates ckpts and risks divergence."""
    from src.training.checkpoint import CheckpointMixin

    fake_sd = {
        "model.llada.lora_A.weight": "kept",
        "model.gnn.encoder.x": "kept",
        "model.qformer.Qformer.weight": "kept",
        "model.ref_model.llada.lora_A.weight": "dropped",
        "model.ref_model.gnn.encoder.x": "dropped",
        "model.unrelated_frozen.weight": "dropped",  # base LLM
    }
    checkpoint = {"state_dict": dict(fake_sd)}
    saver = _StubTrainerForCheckpoint(TaskAnchorEMA(alpha=0.99))
    CheckpointMixin.on_save_checkpoint(saver, checkpoint)

    remaining = set(checkpoint["state_dict"].keys())
    assert "model.ref_model.llada.lora_A.weight" not in remaining
    assert "model.ref_model.gnn.encoder.x" not in remaining
    # trainable keepers stay
    assert "model.llada.lora_A.weight" in remaining
    assert "model.gnn.encoder.x" in remaining
    assert "model.qformer.Qformer.weight" in remaining
