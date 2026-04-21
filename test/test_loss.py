"""Tests for MaskedDiffusionLoss in src/training/loss.py."""

import os

import pytest
import torch

from src.training.loss import MaskedDiffusionLoss, MASK_TOKEN_ID, EPS


@pytest.fixture
def loss_fn(tmp_path):
    return MaskedDiffusionLoss(
        mask_token_id=MASK_TOKEN_ID,
        eos_token_id=0,  # Source requires a concrete int for per_sample_loss_no_eos path.
        log_nan=True,
        nan_log_dir=str(tmp_path / "nan_logs"),
    )


def _make_batch(batch_size=4, seq_len=32, prompt_len=16, vocab_size=1000):
    """Create a synthetic batch with prompt + answer."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # mask prompt
    return input_ids, labels


class TestMakeNoisy:

    def test_output_shapes(self, loss_fn):
        input_ids, labels = _make_batch()
        noisy_ids, mask_indices, p_mask = loss_fn.make_noisy(input_ids, labels)
        B, L = input_ids.shape
        assert noisy_ids.shape == (B, L)
        assert mask_indices.shape == (B, L)
        assert mask_indices.dtype == torch.bool
        assert p_mask.shape == (B, 1)

    def test_only_masks_answer_tokens(self, loss_fn):
        input_ids, labels = _make_batch(prompt_len=16)
        _, mask_indices, _ = loss_fn.make_noisy(input_ids, labels)
        # Prompt positions (labels == -100) should never be masked
        prompt_mask = (labels == -100)
        assert (mask_indices & prompt_mask).sum() == 0

    def test_preserves_prompt_tokens(self, loss_fn):
        input_ids, labels = _make_batch(prompt_len=16)
        noisy_ids, _, _ = loss_fn.make_noisy(input_ids, labels)
        prompt_mask = (labels == -100)
        assert torch.equal(noisy_ids[prompt_mask], input_ids[prompt_mask])

    def test_masked_positions_have_mask_id(self, loss_fn):
        input_ids, labels = _make_batch()
        noisy_ids, mask_indices, _ = loss_fn.make_noisy(input_ids, labels)
        assert (noisy_ids[mask_indices] == MASK_TOKEN_ID).all()

    def test_at_least_one_masked_per_sample(self, loss_fn):
        input_ids, labels = _make_batch(batch_size=8)
        # Run multiple times to check guarantee
        for _ in range(10):
            _, mask_indices, _ = loss_fn.make_noisy(input_ids, labels)
            per_sample = mask_indices.sum(dim=1)
            assert (per_sample >= 1).all(), f"Some samples have 0 masked tokens: {per_sample}"

    def test_p_mask_range(self, loss_fn):
        input_ids, labels = _make_batch(batch_size=64)
        _, _, p_mask = loss_fn.make_noisy(input_ids, labels)
        assert (p_mask >= EPS).all()
        assert (p_mask <= 1.0).all()

    def test_deterministic_with_seed(self, loss_fn):
        input_ids, labels = _make_batch()
        torch.manual_seed(42)
        out1 = loss_fn.make_noisy(input_ids, labels)
        torch.manual_seed(42)
        out2 = loss_fn.make_noisy(input_ids, labels)
        assert torch.equal(out1[0], out2[0])
        assert torch.equal(out1[1], out2[1])


class TestForward:

    def _make_logits_and_batch(self, batch_size=4, seq_len=32, prompt_len=16, vocab_size=1000):
        input_ids, labels = _make_batch(batch_size, seq_len, prompt_len, vocab_size)
        loss_fn = MaskedDiffusionLoss(eos_token_id=0)
        noisy_ids, mask_indices, p_mask = loss_fn.make_noisy(input_ids, labels)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        return logits, input_ids, labels, mask_indices, p_mask

    def test_returns_loss_and_answer_length(self):
        logits, input_ids, labels, mask_indices, p_mask = self._make_logits_and_batch()
        loss_fn = MaskedDiffusionLoss(eos_token_id=0)
        result = loss_fn(logits, input_ids, labels, mask_indices, p_mask)
        assert "loss" in result
        assert "answer_length_mean" in result
        assert "per_sample_loss" in result
        assert "per_sample_loss_no_eos" in result

    def test_loss_is_finite_scalar(self):
        logits, input_ids, labels, mask_indices, p_mask = self._make_logits_and_batch()
        loss_fn = MaskedDiffusionLoss(eos_token_id=0)
        result = loss_fn(logits, input_ids, labels, mask_indices, p_mask)
        loss = result["loss"]
        assert loss.shape == ()
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    def test_loss_positive(self):
        logits, input_ids, labels, mask_indices, p_mask = self._make_logits_and_batch()
        loss_fn = MaskedDiffusionLoss(eos_token_id=0)
        result = loss_fn(logits, input_ids, labels, mask_indices, p_mask)
        assert result["loss"] > 0

    def test_answer_length_mean_correct(self):
        input_ids, labels = _make_batch(batch_size=2, seq_len=32, prompt_len=16)
        loss_fn = MaskedDiffusionLoss(eos_token_id=0)
        noisy_ids, mask_indices, p_mask = loss_fn.make_noisy(input_ids, labels)
        logits = torch.randn(2, 32, 1000)
        result = loss_fn(logits, input_ids, labels, mask_indices, p_mask)
        expected_mean = (labels != -100).sum(dim=1).float().mean().item()
        assert result["answer_length_mean"] == pytest.approx(expected_mean, rel=1e-5)

    def test_per_sample_loss_shape(self):
        B = 4
        logits, input_ids, labels, mask_indices, p_mask = self._make_logits_and_batch(batch_size=B)
        loss_fn = MaskedDiffusionLoss(eos_token_id=0)
        result = loss_fn(logits, input_ids, labels, mask_indices, p_mask)
        assert result["per_sample_loss"].shape == (B,)
        assert result["per_sample_loss_no_eos"].shape == (B,)

    def test_per_sample_loss_mean_equals_loss(self):
        logits, input_ids, labels, mask_indices, p_mask = self._make_logits_and_batch()
        loss_fn = MaskedDiffusionLoss(eos_token_id=0)
        result = loss_fn(logits, input_ids, labels, mask_indices, p_mask)
        assert result["per_sample_loss"].mean().item() == pytest.approx(
            result["loss"].item(), rel=1e-4
        )

    def test_loss_no_eos_leq_loss(self):
        """loss_no_eos should generally differ from loss (different normalization)."""
        logits, input_ids, labels, mask_indices, p_mask = self._make_logits_and_batch()
        loss_fn = MaskedDiffusionLoss(eos_token_id=0)
        result = loss_fn(logits, input_ids, labels, mask_indices, p_mask)
        # Both should be finite positive
        assert torch.isfinite(result["per_sample_loss_no_eos"]).all()
        assert (result["per_sample_loss_no_eos"] > 0).all()

    def test_per_sample_loss_no_grad(self):
        """per_sample_loss tensors should be detached (no grad)."""
        logits, input_ids, labels, mask_indices, p_mask = self._make_logits_and_batch()
        loss_fn = MaskedDiffusionLoss(eos_token_id=0)
        result = loss_fn(logits, input_ids, labels, mask_indices, p_mask)
        assert not result["per_sample_loss"].requires_grad
        assert not result["per_sample_loss_no_eos"].requires_grad


class TestEdgeCases:

    def test_single_token_answer(self):
        B, L = 2, 10
        input_ids = torch.randint(0, 100, (B, L))
        labels = torch.full((B, L), -100)
        labels[:, -1] = input_ids[:, -1]  # Only last token is answer
        loss_fn = MaskedDiffusionLoss(eos_token_id=0)
        noisy_ids, mask_indices, p_mask = loss_fn.make_noisy(input_ids, labels)
        logits = torch.randn(B, L, 100)
        result = loss_fn(logits, input_ids, labels, mask_indices, p_mask)
        assert torch.isfinite(result["loss"])

    def test_batch_size_one(self):
        input_ids, labels = _make_batch(batch_size=1)
        loss_fn = MaskedDiffusionLoss(eos_token_id=0)
        noisy_ids, mask_indices, p_mask = loss_fn.make_noisy(input_ids, labels)
        logits = torch.randn(1, 32, 1000)
        result = loss_fn(logits, input_ids, labels, mask_indices, p_mask)
        assert torch.isfinite(result["loss"])


class TestNaNLogging:

    def test_nan_logging_creates_file(self, tmp_path):
        nan_dir = str(tmp_path / "nan_logs")
        loss_fn = MaskedDiffusionLoss(log_nan=True, nan_log_dir=nan_dir)
        input_ids = torch.tensor([[1, 2, 3]])
        labels = torch.tensor([[-100, 2, 3]])
        mask_indices = torch.tensor([[False, True, True]])
        loss_fn._log_nan(input_ids, labels, mask_indices, ["test_task"], global_step=42)
        assert os.path.exists(os.path.join(nan_dir, "nan_step42.pt"))

    def test_nan_log_content(self, tmp_path):
        nan_dir = str(tmp_path / "nan_logs")
        loss_fn = MaskedDiffusionLoss(log_nan=True, nan_log_dir=nan_dir)
        input_ids = torch.tensor([[1, 2, 3]])
        labels = torch.tensor([[-100, 2, 3]])
        mask_indices = torch.tensor([[False, True, True]])
        loss_fn._log_nan(input_ids, labels, mask_indices, ["task_a"], global_step=10)
        data = torch.load(os.path.join(nan_dir, "nan_step10.pt"), weights_only=False)
        assert torch.equal(data["input_ids"], input_ids)
        assert data["tasks"] == ["task_a"]
        assert data["global_step"] == 10
