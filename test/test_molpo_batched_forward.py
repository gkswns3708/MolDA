"""Unit tests for batched chosen+rejected forward in `_molpo_forward`.

After the refactor (matching Old_MolDA / mol-llm_official `concatenated_forward`):
  - PyG `_slice_batch` is called once for the combined [chosen; rejected] pair
    instead of twice (chosen, rejected).
  - `compute_elbo` runs once for πθ and once for πref on the 2B batch.
  - ELBO results are then split into [:B] (chosen) and [B:] (rejected).

These tests don't run the full model (LLaDA-8B + GNN is too heavy for unit tests)
— they verify:
  1. `_slice_batch` handles a 2B slice correctly (PyG re-batch, tensor slicing).
  2. `compute_elbo` returns a [2B] tensor given 2B input_ids/labels.
  3. Splitting the [2B] ELBO into chosen/rejected halves preserves shape/values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_compute_elbo_returns_2B_tensor():
    """compute_elbo must work on any batch size, including 2B for the
    combined chosen+rejected pair."""
    from src.training.vrpo_elbo import compute_elbo

    B = 3
    L = 32
    V = 256
    n_t = 2

    # Mock forward_fn: returns [batch, L, V] random logits
    def mock_fwd(noisy_ids, attention_mask=None):
        return torch.randn(noisy_ids.shape[0], L, V)

    # Single B-row call — keep ids within [1, V-1] so gather() indices are safe
    ids_B = torch.randint(1, V - 1, (B, L))
    lab_B = ids_B.clone()
    lab_B[:, :L // 2] = -100  # prompt half labeled out
    am_B = torch.ones_like(ids_B)
    elbo_B = compute_elbo(
        mock_fwd, ids_B, lab_B, n_t=n_t, seed=42,
        mask_token_id=126336, attention_mask=am_B,
    )
    assert elbo_B.shape == (B,), f"Expected [B={B}], got {elbo_B.shape}"

    # Doubled 2B-row call (simulates batched chosen+rejected). Second half
    # uses fresh ids (still within vocab) — represents "rejected" rows.
    ids_2B = torch.cat([ids_B, torch.randint(1, V - 1, (B, L))], dim=0)
    lab_2B = torch.cat([lab_B, lab_B], dim=0)
    am_2B = torch.cat([am_B, am_B], dim=0)
    elbo_2B = compute_elbo(
        mock_fwd, ids_2B, lab_2B, n_t=n_t, seed=42,
        mask_token_id=126336, attention_mask=am_2B,
    )
    assert elbo_2B.shape == (2 * B,), f"Expected [2B={2 * B}], got {elbo_2B.shape}"


def test_split_elbo_halves_preserve_dtype():
    """After batched forward, splitting [2B] into [:B] and [B:] must give
    tensors that can be passed to compute_v_molpo_loss without type issues."""
    from src.training.v_molpo_loss import TaskAnchorEMA, compute_v_molpo_loss

    B = 4
    elbo_theta_pair = torch.randn(2 * B) * 0.1 - 0.5
    elbo_ref_pair = torch.randn(2 * B) * 0.1 - 0.6

    elbo_theta_w = elbo_theta_pair[:B]
    elbo_theta_l = elbo_theta_pair[B:]
    elbo_ref_w = elbo_ref_pair[:B]
    elbo_ref_l = elbo_ref_pair[B:]

    assert elbo_theta_w.shape == (B,)
    assert elbo_theta_l.shape == (B,)
    assert elbo_ref_w.shape == (B,)
    assert elbo_ref_l.shape == (B,)

    # Verify compute_v_molpo_loss can consume the split halves
    ema = TaskAnchorEMA(alpha=0.99)
    out = compute_v_molpo_loss(
        elbo_theta_w=elbo_theta_w,
        elbo_theta_l=elbo_theta_l,
        elbo_ref_w=elbo_ref_w,
        elbo_ref_l=elbo_ref_l,
        tasks_chosen=["chebi-20-mol2text"] * B,
        task_anchor_ema=ema,
        beta=2.0,
        molpo_lambda=0.5,
        margin_clip_active=False,
        margin_clip_scale=1.0,
        anc_rejected_weight=0.0,
        rejected_lambda=1.5,
        loss_type="sigmoid",
    )
    assert out["rewards_chosen"].shape == (B,)
    assert out["rewards_rejected"].shape == (B,)
    assert out["margin"].shape == (B,)
    assert out["loss_pref_per_sample"].shape == (B,)


def test_slice_batch_handles_2B_slice():
    """`_slice_batch` must correctly slice a 2B sub-range out of a larger
    batch, including PyG `Batch` re-batching."""
    try:
        from torch_geometric.data import Batch, Data
    except ImportError:
        pytest.skip("torch_geometric not available")

    from src.model.molda import MolDA  # noqa

    # Build a fake batch with 3B rows (mimicking molpo_batch_division=3 layout)
    B = 2
    total = 3 * B  # sft + chosen + rejected
    L = 16

    # Tensors of shape [total, L]
    input_ids = torch.arange(total * L).reshape(total, L)
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    tasks = [f"task_{i}" for i in range(total)]

    # PyG Batch with `total` Data objects (different node counts to verify split)
    data_list = [
        Data(
            x=torch.zeros(3 + i, 4, dtype=torch.long),  # different node count per row
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.zeros(1, 3, dtype=torch.long),
        )
        for i in range(total)
    ]
    graphs = Batch.from_data_list(data_list)

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "tasks": tasks,
        "graphs": graphs,
    }

    # Slice the chosen+rejected pair (div=3 → slice(B, 3B))
    pair_slice = slice(B, 3 * B)

    # Reproduce _slice_batch logic (avoid instantiating MolDA which needs a config)
    sub = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            sub[k] = v[pair_slice]
        elif k == "tasks" and isinstance(v, list):
            sub[k] = list(v[pair_slice])
        elif k == "graphs":
            dl = v.to_data_list()
            sub[k] = Batch.from_data_list(dl[pair_slice])

    assert sub["input_ids"].shape == (2 * B, L)
    assert sub["labels"].shape == (2 * B, L)
    assert sub["attention_mask"].shape == (2 * B, L)
    assert len(sub["tasks"]) == 2 * B
    assert sub["tasks"][0] == f"task_{B}"
    assert sub["tasks"][-1] == f"task_{total - 1}"
    # PyG Batch should have 2B graphs
    assert len(sub["graphs"].to_data_list()) == 2 * B


def test_pair_slice_covers_chosen_and_rejected_div2():
    """Verify pair_slice arithmetic for batch_division=2."""
    B = 4
    chosen_slice = slice(0, B)
    rejected_slice = slice(B, 2 * B)
    pair_slice = slice(chosen_slice.start, rejected_slice.stop)
    assert pair_slice == slice(0, 2 * B)


def test_pair_slice_covers_chosen_and_rejected_div3():
    """Verify pair_slice arithmetic for batch_division=3 (SFT slot skipped)."""
    B = 4
    sft_slice = slice(0, B)         # noqa: F841 (illustrative)
    chosen_slice = slice(B, 2 * B)
    rejected_slice = slice(2 * B, 3 * B)
    pair_slice = slice(chosen_slice.start, rejected_slice.stop)
    assert pair_slice == slice(B, 3 * B)
    # length is still 2B
    assert pair_slice.stop - pair_slice.start == 2 * B
