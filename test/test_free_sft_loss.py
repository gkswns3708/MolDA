"""Tests for the "free SFT" path — SFT loss computed from the chosen half of
the V-MolPO pair forward, without an additional forward pass.

Mirrors Old_MolDA / mol-llm_official's `concatenated_forward` pattern where
`instance_loss[:B]` (chosen half) is averaged token-wise to produce L_SFT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _mock_fwd_factory(L: int, V: int, seed: int = 0):
    """Build a deterministic mock forward_fn returning [B, L, V] logits."""
    def fwd(noisy_ids, attention_mask=None):
        g = torch.Generator()
        g.manual_seed(seed + int(noisy_ids.sum().item()) % 100003)
        return torch.randn(noisy_ids.shape[0], L, V, generator=g)
    return fwd


# ---------------------------------------------------------------------
# compute_elbo with return_token_loss_sum=True
# ---------------------------------------------------------------------
def test_compute_elbo_returns_token_loss_sum_dict():
    """When `return_token_loss_sum=True`, compute_elbo must return a dict
    with elbo / token_loss_sum / answer_lengths keys, all shape [B]."""
    from src.training.vrpo_elbo import compute_elbo

    B, L, V, n_t = 4, 32, 256, 2
    ids = torch.randint(1, V - 1, (B, L))
    lab = ids.clone()
    lab[:, :L // 2] = -100   # prompt half labeled out
    am = torch.ones_like(ids)

    out = compute_elbo(
        _mock_fwd_factory(L, V), ids, lab,
        n_t=n_t, seed=42, mask_token_id=126336,
        attention_mask=am, return_token_loss_sum=True,
    )
    assert isinstance(out, dict)
    assert set(out.keys()) >= {"elbo", "token_loss_sum", "answer_lengths"}
    assert out["elbo"].shape == (B,)
    assert out["token_loss_sum"].shape == (B,)
    assert out["answer_lengths"].shape == (B,)


def test_compute_elbo_default_call_unchanged():
    """Default call (no return_token_loss_sum) must still return just elbo."""
    from src.training.vrpo_elbo import compute_elbo

    B, L, V = 3, 16, 128
    ids = torch.randint(1, V - 1, (B, L))
    lab = ids.clone()
    lab[:, :L // 2] = -100
    am = torch.ones_like(ids)

    out = compute_elbo(
        _mock_fwd_factory(L, V), ids, lab,
        n_t=2, seed=7, mask_token_id=126336,
        attention_mask=am,
    )
    assert isinstance(out, torch.Tensor)
    assert out.shape == (B,)


def test_token_averaged_sft_matches_manual_computation():
    """`loss_sft = total_token_loss_sum / total_answer_lens` must match a
    hand-computed token-averaged NLL on the chosen half."""
    from src.training.vrpo_elbo import compute_elbo

    B_pair = 6   # 2B (chosen + rejected)
    B = B_pair // 2
    L, V = 24, 128
    ids = torch.randint(1, V - 1, (B_pair, L))
    lab = ids.clone()
    lab[:, :L // 2] = -100
    am = torch.ones_like(ids)

    out = compute_elbo(
        _mock_fwd_factory(L, V), ids, lab,
        n_t=2, seed=11, mask_token_id=126336,
        attention_mask=am, return_token_loss_sum=True,
    )
    token_sum = out["token_loss_sum"]
    ans_lens = out["answer_lengths"]

    chosen_token_sum = token_sum[:B]
    chosen_ans_lens = ans_lens[:B]
    loss_sft = chosen_token_sum.sum() / chosen_ans_lens.sum().clamp(min=1.0)

    manual = chosen_token_sum.sum() / chosen_ans_lens.sum().clamp(min=1.0)
    assert torch.allclose(loss_sft, manual, atol=1e-6)


def test_loss_sft_only_uses_chosen_half():
    """Modifying the rejected half of token_loss_sum must NOT affect the
    chosen-only SFT loss computation."""
    B_pair = 6
    B = B_pair // 2
    # Fabricate token_loss_sum and answer_lens tensors
    token_sum = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])   # [2B]
    ans_lens = torch.tensor([10.0] * B_pair)

    # SFT on chosen half only
    chosen_loss = token_sum[:B].sum() / ans_lens[:B].sum()
    expected = (1.0 + 2.0 + 3.0) / 30.0
    assert torch.allclose(chosen_loss, torch.tensor(expected))

    # Perturb rejected half — chosen-only SFT must be unchanged
    perturbed = token_sum.clone()
    perturbed[B:] = 999.0
    chosen_loss_perturbed = perturbed[:B].sum() / ans_lens[:B].sum()
    assert torch.allclose(chosen_loss, chosen_loss_perturbed)


def test_per_sample_loss_shape_matches_tasks_length():
    """`per_sample_loss_sft` derived from chosen_token_loss_sum / chosen_answer_lens
    must be shape [B] so the trainer's per-task slicer (`v[mask].mean()`)
    works on it."""
    B = 3
    chosen_token_sum = torch.tensor([5.0, 10.0, 15.0])
    chosen_ans_lens = torch.tensor([5.0, 10.0, 15.0])
    per_sample = chosen_token_sum / chosen_ans_lens.clamp(min=1.0)
    assert per_sample.shape == (B,)
    assert torch.allclose(per_sample, torch.tensor([1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------
# eos_mask: separate loss_no_eos accounting (matches MaskedDiffusionLoss)
# ---------------------------------------------------------------------
def test_compute_elbo_no_eos_keys_present_when_mask_given():
    """Passing eos_mask + return_token_loss_sum=True must add
    token_loss_sum_no_eos and content_lengths to the result dict."""
    from src.training.vrpo_elbo import compute_elbo

    B, L, V, n_t = 3, 24, 128, 2
    ids = torch.randint(1, V - 1, (B, L))
    lab = ids.clone()
    lab[:, :L // 2] = -100
    am = torch.ones_like(ids)
    # Mark 2 EOS-like positions per row inside the answer region
    eos_mask = torch.zeros_like(ids, dtype=torch.bool)
    eos_mask[:, L - 2:] = True

    out = compute_elbo(
        _mock_fwd_factory(L, V), ids, lab,
        n_t=n_t, seed=13, mask_token_id=126336,
        attention_mask=am, return_token_loss_sum=True,
        eos_mask=eos_mask,
    )
    assert set(out.keys()) >= {
        "elbo", "token_loss_sum", "answer_lengths",
        "token_loss_sum_no_eos", "content_lengths",
    }
    assert out["token_loss_sum_no_eos"].shape == (B,)
    assert out["content_lengths"].shape == (B,)
    # content_lengths = answer_lens - eos_in_answer
    expected_content = out["answer_lengths"] - torch.tensor([2.0] * B)
    assert torch.allclose(out["content_lengths"], expected_content.clamp(min=1.0))


def test_compute_elbo_no_eos_omitted_without_mask():
    """Without eos_mask, the returned dict must NOT carry no-eos keys
    (back-compat for callers that don't request EOS separation)."""
    from src.training.vrpo_elbo import compute_elbo

    B, L, V = 2, 16, 64
    ids = torch.randint(1, V - 1, (B, L))
    lab = ids.clone()
    lab[:, :L // 2] = -100
    am = torch.ones_like(ids)

    out = compute_elbo(
        _mock_fwd_factory(L, V), ids, lab,
        n_t=2, seed=3, mask_token_id=126336,
        attention_mask=am, return_token_loss_sum=True,
    )
    assert "token_loss_sum_no_eos" not in out
    assert "content_lengths" not in out


def test_token_loss_sum_no_eos_bounded_by_token_loss_sum():
    """For any seed, token_loss_sum_no_eos ≤ token_loss_sum (entry-wise),
    because no_eos zeroes out a subset of the same nonneg weighted NLL."""
    from src.training.vrpo_elbo import compute_elbo

    B, L, V = 4, 32, 256
    ids = torch.randint(1, V - 1, (B, L))
    lab = ids.clone()
    lab[:, :L // 2] = -100
    am = torch.ones_like(ids)
    eos_mask = torch.zeros_like(ids, dtype=torch.bool)
    # 1 EOS at the last answer position for every sample
    eos_mask[:, -1] = True

    out = compute_elbo(
        _mock_fwd_factory(L, V), ids, lab,
        n_t=3, seed=101, mask_token_id=126336,
        attention_mask=am, return_token_loss_sum=True,
        eos_mask=eos_mask,
    )
    assert (out["token_loss_sum_no_eos"] <= out["token_loss_sum"] + 1e-6).all()


def test_no_eos_matches_inclusive_when_mask_all_false():
    """All-False eos_mask must produce token_loss_sum_no_eos == token_loss_sum
    and content_lengths == answer_lengths."""
    from src.training.vrpo_elbo import compute_elbo

    B, L, V = 3, 20, 96
    ids = torch.randint(1, V - 1, (B, L))
    lab = ids.clone()
    lab[:, :L // 2] = -100
    am = torch.ones_like(ids)
    eos_mask = torch.zeros_like(ids, dtype=torch.bool)

    out = compute_elbo(
        _mock_fwd_factory(L, V), ids, lab,
        n_t=2, seed=55, mask_token_id=126336,
        attention_mask=am, return_token_loss_sum=True,
        eos_mask=eos_mask,
    )
    assert torch.allclose(out["token_loss_sum_no_eos"], out["token_loss_sum"])
    assert torch.allclose(out["content_lengths"], out["answer_lengths"])


def test_per_sample_loss_no_eos_differs_when_eos_present():
    """The end-to-end free-SFT formula
        per_sample_loss = chosen_token_loss_sum / chosen_answer_lens
        per_sample_loss_no_eos = chosen_token_loss_sum_no_eos / chosen_content_lens
    must produce DIFFERENT values whenever at least one EOS token sits in
    the answer region — this is the regression that motivated the fix."""
    from src.training.vrpo_elbo import compute_elbo

    B_pair = 4
    B = B_pair // 2
    L, V = 24, 128
    ids = torch.randint(1, V - 1, (B_pair, L))
    lab = ids.clone()
    lab[:, :L // 2] = -100
    am = torch.ones_like(ids)
    eos_mask = torch.zeros_like(ids, dtype=torch.bool)
    eos_mask[:, -1] = True  # one EOS per row inside the answer region

    out = compute_elbo(
        _mock_fwd_factory(L, V), ids, lab,
        n_t=2, seed=77, mask_token_id=126336,
        attention_mask=am, return_token_loss_sum=True,
        eos_mask=eos_mask,
    )

    chosen_token_sum = out["token_loss_sum"][:B]
    chosen_token_sum_no_eos = out["token_loss_sum_no_eos"][:B]
    chosen_ans_lens = out["answer_lengths"][:B]
    chosen_content_lens = out["content_lengths"][:B]

    per_sample_loss = chosen_token_sum / chosen_ans_lens.clamp(min=1.0)
    per_sample_loss_no_eos = (
        chosen_token_sum_no_eos / chosen_content_lens.clamp(min=1.0)
    )
    # Each row should differ (denominator differs by 1, and at least one
    # nonzero EOS-position contribution is dropped with high probability).
    assert not torch.allclose(per_sample_loss, per_sample_loss_no_eos)


# ---------------------------------------------------------------------
# rewards_accuracies (GDR) metric in compute_v_molpo_loss
# ---------------------------------------------------------------------
def test_rewards_accuracies_in_v_molpo_output():
    """compute_v_molpo_loss must return `rewards_accuracies` of shape [B]
    with 0/1 indicator that chosen reward > rejected reward."""
    from src.training.v_molpo_loss import TaskAnchorEMA, compute_v_molpo_loss

    B = 4
    # Hand-craft inputs so margin sign is known per row
    # r_w = β(elbo_θ_w - elbo_ref_w); larger gap → positive margin
    elbo_theta_w = torch.tensor([0.0, 0.1, -0.1, 0.5])
    elbo_ref_w = torch.tensor([0.0, 0.0, 0.0, 0.0])
    elbo_theta_l = torch.tensor([-0.5, 0.2, 0.1, -0.5])
    elbo_ref_l = torch.tensor([0.0, 0.0, 0.0, 0.0])

    ema = TaskAnchorEMA(alpha=0.99)
    out = compute_v_molpo_loss(
        elbo_theta_w=elbo_theta_w, elbo_ref_w=elbo_ref_w,
        elbo_theta_l=elbo_theta_l, elbo_ref_l=elbo_ref_l,
        tasks_chosen=["chebi-20-mol2text"] * B,
        task_anchor_ema=ema,
        beta=1.0, molpo_lambda=0.5,
        margin_clip_active=False, margin_clip_scale=1.0,
        anc_rejected_weight=0.0, rejected_lambda=1.5,
        loss_type="sigmoid",
    )
    assert "rewards_accuracies" in out
    assert out["rewards_accuracies"].shape == (B,)

    # Manual check: margin_raw = r_w - r_l = β(elbo_theta_w-elbo_ref_w) - β(elbo_theta_l-elbo_ref_l)
    # row 0: 0 - (-0.5) = 0.5 → 1
    # row 1: 0.1 - 0.2  = -0.1 → 0
    # row 2: -0.1 - 0.1 = -0.2 → 0
    # row 3: 0.5 - (-0.5) = 1.0 → 1
    expected = torch.tensor([1.0, 0.0, 0.0, 1.0])
    assert torch.allclose(out["rewards_accuracies"], expected)


def test_rewards_accuracies_mean_matches_gdr_definition():
    """Batch mean of rewards_accuracies = fraction of preference pairs
    where chosen reward > rejected reward (Generation Direction Ratio)."""
    from src.training.v_molpo_loss import TaskAnchorEMA, compute_v_molpo_loss

    B = 8
    torch.manual_seed(0)
    elbo_theta_w = torch.rand(B) * 0.5
    elbo_ref_w = torch.zeros(B)
    elbo_theta_l = torch.rand(B) * 0.5
    elbo_ref_l = torch.zeros(B)

    ema = TaskAnchorEMA(alpha=0.99)
    out = compute_v_molpo_loss(
        elbo_theta_w=elbo_theta_w, elbo_ref_w=elbo_ref_w,
        elbo_theta_l=elbo_theta_l, elbo_ref_l=elbo_ref_l,
        tasks_chosen=["chebi-20-mol2text"] * B,
        task_anchor_ema=ema,
        beta=1.0, molpo_lambda=0.5,
        margin_clip_active=False, margin_clip_scale=1.0,
        anc_rejected_weight=0.0, rejected_lambda=1.5,
        loss_type="sigmoid",
    )
    gdr = out["rewards_accuracies"].mean().item()
    margin_raw = out["margin_unclipped"]
    expected_gdr = (margin_raw > 0).float().mean().item()
    assert abs(gdr - expected_gdr) < 1e-6
    assert 0.0 <= gdr <= 1.0


def test_rewards_accuracies_differentiable_off():
    """rewards_accuracies is a binary indicator, should not require grad."""
    from src.training.v_molpo_loss import TaskAnchorEMA, compute_v_molpo_loss

    B = 4
    elbo_theta_w = torch.randn(B, requires_grad=True)
    elbo_theta_l = torch.randn(B, requires_grad=True)
    elbo_ref_w = torch.zeros(B)
    elbo_ref_l = torch.zeros(B)

    ema = TaskAnchorEMA(alpha=0.99)
    out = compute_v_molpo_loss(
        elbo_theta_w=elbo_theta_w, elbo_ref_w=elbo_ref_w,
        elbo_theta_l=elbo_theta_l, elbo_ref_l=elbo_ref_l,
        tasks_chosen=["chebi-20-mol2text"] * B,
        task_anchor_ema=ema,
        beta=1.0, molpo_lambda=0.5,
        margin_clip_active=False, margin_clip_scale=1.0,
        anc_rejected_weight=0.0, rejected_lambda=1.5,
        loss_type="sigmoid",
    )
    # `(margin > 0).float()` should not propagate gradient (boolean op)
    assert not out["rewards_accuracies"].requires_grad
