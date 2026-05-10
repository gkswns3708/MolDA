"""Tests for src/model/ref_llada_wrapper.py.

Lightweight tests (no model load):
  - _extract_state_dict prefix stripping
  - ckpt with vs without "state_dict" wrapping

Heavy tests (GPU + 8B model load) are deferred to Phase 3 integration.
"""
import torch
import pytest

from src.model.ref_llada_wrapper import _extract_state_dict


class TestExtractStateDict:

    def test_lightning_format_strips_prefix(self):
        """Lightning ckpt: {state_dict: {"model.foo": ..., "model.bar": ...}}."""
        ckpt = {
            "state_dict": {
                "model.llada.embed.weight": torch.zeros(10, 4),
                "model.qformer.layer.0.weight": torch.zeros(8, 4),
            },
            "epoch": 5,
        }
        out = _extract_state_dict(ckpt, strip_prefix="model.")
        assert set(out.keys()) == {"llada.embed.weight", "qformer.layer.0.weight"}
        assert out["llada.embed.weight"].shape == (10, 4)

    def test_raw_state_dict_no_prefix(self):
        """If ckpt is a raw dict of weights (no 'state_dict' key), return as-is."""
        raw = {
            "llada.embed.weight": torch.zeros(10, 4),
            "qformer.layer.0.weight": torch.zeros(8, 4),
        }
        out = _extract_state_dict(raw, strip_prefix="model.")
        assert set(out.keys()) == {"llada.embed.weight", "qformer.layer.0.weight"}

    def test_mixed_keys(self):
        """Some keys with prefix, some without — only strip those with prefix."""
        ckpt = {
            "state_dict": {
                "model.foo": torch.zeros(2),
                "bar": torch.zeros(3),
            },
        }
        out = _extract_state_dict(ckpt, strip_prefix="model.")
        assert "foo" in out and "bar" in out
        assert "model.foo" not in out

    def test_custom_prefix(self):
        ckpt = {"state_dict": {"trainer.model.x": torch.zeros(1)}}
        out = _extract_state_dict(ckpt, strip_prefix="trainer.model.")
        assert list(out.keys()) == ["x"]


@pytest.mark.gpu
@pytest.mark.slow
def test_ref_molda_freeze(trainer_module):
    """Integration test: actually instantiate RefMolDA from a real ckpt.

    Requires GPU + 8B model load. Marked slow/gpu — skipped in default test runs.
    Saves trainer_module's current state to a temp file and uses it as ref ckpt.
    """
    import tempfile
    import os

    from src.model.ref_llada_wrapper import RefMolDA

    cfg = trainer_module.cfg

    # Save current MolDA state as a fake "stage 2 ckpt"
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "fake_stage2.ckpt")
        sd = {f"model.{k}": v for k, v in trainer_module.model.state_dict().items()}
        torch.save({"state_dict": sd}, ckpt_path)

        ref = RefMolDA(cfg, ref_ckpt_path=ckpt_path)
        # All params frozen
        n_grad = sum(1 for p in ref.parameters() if p.requires_grad)
        assert n_grad == 0, f"RefMolDA must have 0 trainable params, got {n_grad}"
        # Module in eval mode
        assert not ref.training
