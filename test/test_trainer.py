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
        # Stage 1: lora, embed, head (3 groups, other=0.0 so excluded)
        assert len(groups) == 3, f"Expected 3 param groups, got {len(groups)}"

    def test_param_group_names(self, trainer_module):
        optimizer = trainer_module.configure_optimizers()
        names = [g["name"] for g in optimizer.param_groups]
        assert "lora" in names
        assert "embed" in names
        assert "head" in names

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
