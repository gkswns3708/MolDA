"""Tests for MolDATrainer Lightning module (GPU required).

Uses session-scoped trainer_module fixture from conftest.py.
"""

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.slow
class TestConfigureOptimizers:

    def test_param_groups_count(self, trainer_module):
        optimizer = trainer_module.configure_optimizers()
        groups = optimizer.param_groups
        # weight_tying=True: lora, embed (2 groups — head 비어있음, other=0.0)
        assert len(groups) >= 2, f"Expected >= 2 param groups, got {len(groups)}"

    def test_param_group_names(self, trainer_module):
        optimizer = trainer_module.configure_optimizers()
        names = [g["name"] for g in optimizer.param_groups]
        assert "lora" in names
        assert "embed" in names
        # weight_tying=True: head group은 없을 수 있음

    def test_param_group_lr_values(self, trainer_module, cfg):
        optimizer = trainer_module.configure_optimizers()
        for group in optimizer.param_groups:
            if group["name"] == "lora":
                assert group["lr"] == pytest.approx(cfg.lr.lora)
            elif group["name"] == "embed":
                assert group["lr"] == pytest.approx(cfg.lr.embed_orig)
            elif group["name"] == "head":
                assert group["lr"] == pytest.approx(cfg.lr.head_orig)

    def test_all_param_groups_have_params(self, trainer_module):
        optimizer = trainer_module.configure_optimizers()
        for group in optimizer.param_groups:
            assert len(group["params"]) > 0, f"Group {group['name']} has no params"


@pytest.mark.gpu
@pytest.mark.slow
class TestCheckpointFiltering:

    def test_on_save_checkpoint_filters_frozen(self, trainer_module):
        checkpoint = {"state_dict": dict(trainer_module.state_dict())}
        trainer_module.on_save_checkpoint(checkpoint)
        # After filtering, no base transformer keys should remain
        for key in checkpoint["state_dict"]:
            # Base transformer weights (not lora, not embed/head) should be removed
            is_base = ("q_proj.weight" in key or "k_proj.weight" in key) and "lora" not in key and "modules_to_save" not in key
            assert not is_base, f"Frozen param not removed: {key}"

    def test_checkpoint_has_lora(self, trainer_module):
        checkpoint = {"state_dict": dict(trainer_module.state_dict())}
        trainer_module.on_save_checkpoint(checkpoint)
        lora_keys = [k for k in checkpoint["state_dict"] if "lora_" in k]
        assert len(lora_keys) > 0, "No LoRA keys in checkpoint"

    def test_checkpoint_has_embed(self, trainer_module):
        checkpoint = {"state_dict": dict(trainer_module.state_dict())}
        trainer_module.on_save_checkpoint(checkpoint)
        embed_keys = [k for k in checkpoint["state_dict"] if "wte" in k or "embed" in k]
        assert len(embed_keys) > 0, "No embedding keys in checkpoint"


# ─────────────────────────────────────────────
# Masking-ratio bucket logging (CPU — no GPU required)
# ─────────────────────────────────────────────

class TestMaskRatioBucketLogging:
    """training_step의 bucket 집계 루프가 bucket edges와 일관되는지 검증."""

    BUCKET_EDGES = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    BUCKET_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    def _bucketize(self, p_mask_vec):
        inner_edges = self.BUCKET_EDGES[1:-1]
        return torch.bucketize(p_mask_vec, inner_edges)

    def test_boundary_values_land_in_expected_bucket(self):
        # torch.bucketize(right=False) with inner_edges=[0.2, 0.4, 0.6, 0.8]:
        # bucket 0 = (-inf, 0.2], bucket 1 = (0.2, 0.4], ..., bucket 4 = (0.8, +inf)
        # i.e. value == edge → 왼쪽 bucket에 포함된다 (ties go left).
        p = torch.tensor([0.0, 0.19, 0.2, 0.39, 0.4, 0.99, 1.0])
        ids = self._bucketize(p)
        assert ids.tolist() == [0, 0, 0, 1, 1, 4, 4]

    def test_every_sample_mapped_to_exactly_one_bucket(self):
        p = torch.rand(128)
        ids = self._bucketize(p)
        assert ids.shape == (128,)
        assert (ids >= 0).all() and (ids < 5).all()

    def test_bucket_label_count_matches_edges(self):
        assert len(self.BUCKET_LABELS) == len(self.BUCKET_EDGES) - 1


# ─────────────────────────────────────────────
# Config reflection — train_prediction_log_interval 기본값
# ─────────────────────────────────────────────

class TestConfigTrainPredictionLogInterval:
    """default.yaml에서 train_prediction_log_interval이 1000으로 반영되어 있는지."""

    def test_default_is_1000(self):
        from pathlib import Path
        import yaml
        cfg_path = Path(__file__).resolve().parent.parent / "src" / "configs" / "trainer" / "default.yaml"
        text = cfg_path.read_text(encoding="utf-8")
        # @package _global_ 헤더 때문에 yaml.safe_load로 전체 파싱
        data = yaml.safe_load(text)
        assert data["logging"]["train_prediction_log_interval"] == 1000
