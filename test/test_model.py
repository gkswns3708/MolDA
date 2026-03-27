"""Tests for MolDA and LLaDAWrapper model (GPU required).

Uses session-scoped fixtures from conftest.py (모델 1개만 GPU에 로드).
"""

import pytest
import torch

from src.training.loss import MASK_TOKEN_ID


# ─────────────────────────────────────────────
# LLaDAWrapper Tests
# ─────────────────────────────────────────────

@pytest.mark.gpu
@pytest.mark.slow
class TestLLaDAWrapper:

    def test_tokenizer_loaded(self, llada_wrapper):
        assert llada_wrapper.tokenizer is not None

    def test_vocab_expanded(self, llada_wrapper, cfg):
        vocab_size = len(llada_wrapper.tokenizer)
        assert vocab_size > cfg.model.original_vocab_size, (
            f"Vocab not expanded: {vocab_size} <= {cfg.model.original_vocab_size}"
        )

    def test_special_tokens_in_vocab(self, llada_wrapper):
        tokenizer = llada_wrapper.tokenizer
        for token in ["<BOOLEAN>", "</BOOLEAN>", "<SELFIES>", "</SELFIES>",
                       "<FLOAT>", "</FLOAT>", "<mol>", "<INSTRUCTION>"]:
            token_id = tokenizer.convert_tokens_to_ids(token)
            assert token_id != tokenizer.unk_token_id, f"Token {token} not found in vocab"

    def test_lora_applied(self, llada_wrapper):
        lora_params = [n for n, _ in llada_wrapper.model.named_parameters() if "lora_" in n]
        assert len(lora_params) > 0, "No LoRA parameters found"

    def test_trainable_params_exist(self, llada_wrapper):
        trainable = [(n, p) for n, p in llada_wrapper.model.named_parameters() if p.requires_grad]
        assert len(trainable) > 0, "No trainable parameters"
        # Should include lora, wte (weight_tying=True: wte = output head)
        names = [n for n, _ in trainable]
        has_lora = any("lora_" in n for n in names)
        has_embed = any("wte" in n or "embed" in n for n in names)
        assert has_lora, "No LoRA parameters are trainable"
        assert has_embed, "No embedding parameters are trainable"

    def test_frozen_base_weights(self, llada_wrapper):
        """Base attention/MLP weights (non-LoRA) should be frozen."""
        for name, param in llada_wrapper.model.named_parameters():
            if "lora_" not in name and "modules_to_save" not in name:
                if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                    assert not param.requires_grad, f"Base param should be frozen: {name}"


# ─────────────────────────────────────────────
# MolDA Tests
# ─────────────────────────────────────────────

@pytest.mark.gpu
@pytest.mark.slow
class TestMolDA:

    def _make_gpu_batch(self, molda_model, batch_size=2, seq_len=64, prompt_len=32):
        """Create a synthetic batch on GPU."""
        vocab_size = len(molda_model.tokenizer)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device="cuda")
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "tasks": ["smol-property_prediction-bbbp"] * batch_size,
            "global_step": 0,
        }

    def test_forward_produces_loss(self, molda_model):
        batch = self._make_gpu_batch(molda_model)
        molda_model.train()
        result = molda_model(batch)
        assert "loss" in result
        assert "answer_length_mean" in result

    def test_forward_loss_finite(self, molda_model):
        batch = self._make_gpu_batch(molda_model)
        molda_model.train()
        result = molda_model(batch)
        assert torch.isfinite(result["loss"]), f"Loss is not finite: {result['loss']}"

    def test_binary_prob_shape(self, molda_model):
        tokenizer = molda_model.tokenizer
        # Create prompt
        prompt = "<INSTRUCTION>Predict property.</INSTRUCTION> Input: molecule"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        prompt_ids = prompt_ids.cuda()
        prompt_mask = torch.ones_like(prompt_ids)

        molda_model.eval()
        probs = molda_model.compute_binary_prob_likelihood(prompt_ids, prompt_mask)
        assert probs.shape == (1, 2), f"Expected [1, 2], got {probs.shape}"

    def test_binary_prob_sums_to_one(self, molda_model):
        tokenizer = molda_model.tokenizer
        prompt = "<INSTRUCTION>Predict.</INSTRUCTION> mol"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").cuda()
        prompt_mask = torch.ones_like(prompt_ids)

        molda_model.eval()
        probs = molda_model.compute_binary_prob_likelihood(prompt_ids, prompt_mask)
        assert probs.sum(dim=1).item() == pytest.approx(1.0, abs=1e-4)
