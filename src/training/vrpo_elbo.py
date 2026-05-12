"""
VRPO (Variance-Reduced Preference Optimization) ELBO estimator.

Reference: Zhu et al., "LLaDA 1.5: Variance-Reduced Preference Optimization
for Large Language Diffusion Models", arXiv:2505.19223.

Computes per-sample ELBO log-likelihood estimates for masked diffusion LMs using:
  - n_t-MC averaging across diffusion timesteps (Theorem 2: V[B̂] ∝ 1/n_t)
  - Optimal allocation: n_t timesteps, 1 mask realization per timestep
  - Antithetic sampling: πθ and πref share the same seed → same (T, M)
    (Theorem 3: V[diff] reduced when Corr(B̂_θ, B̂_ref) > 0)

Returned `B̂_π(y)` is a log-likelihood approximation per sample. Higher = better.
For DPO-E:  r_θ(y) = β · (B̂_θ(y) − B̂_ref(y))
"""
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

DEFAULT_MASK_TOKEN_ID = 126336  # LLaDA <|mdm_mask|>
DEFAULT_EPS = 1e-3   # LLaDA paper standard timestep floor (plan/stage3.md §4)


def sample_shared_TM(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    n_t: int,
    seed: int,
    eps: float = DEFAULT_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample (T, M) for n_t timesteps with deterministic seed.

    Both πθ and πref must call this with the same `seed` to share noise.

    Args:
        input_ids: [B, L]
        labels:    [B, L] -100 = prompt position
        n_t:       number of timestep samples
        seed:      RNG seed (shared between policy and reference)
        eps:       p_mask floor (1e-3)

    Returns:
        T:            [n_t, B] sampled timesteps in (eps, 1]
        mask_indices: [n_t, B, L] bool — masked positions per (j, b)
                      Already AND-ed with answer_mask. ≥1 mask per (j,b)
                      whenever answer_mask has ≥1 entry for that b.
    """
    B, L = input_ids.shape
    device = input_ids.device
    answer_mask = labels != -100

    # CPU generator → reproducible across CUDA/CPU; cheap for sampling
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    T_cpu = torch.rand((n_t, B), generator=g)
    p_mask = (1.0 - eps) * T_cpu + eps  # [n_t, B]

    rand = torch.rand((n_t, B, L), generator=g)
    answer_mask_cpu = answer_mask.cpu()
    mask_indices = (rand < p_mask.unsqueeze(-1)) & answer_mask_cpu.unsqueeze(0)

    # ≥1 mask guarantee — fall back to a random answer position when empty
    n_masked = mask_indices.sum(dim=-1)  # [n_t, B]
    answer_lens = answer_mask_cpu.sum(dim=-1)  # [B]
    for j in range(n_t):
        for b in range(B):
            if n_masked[j, b] == 0 and answer_lens[b] > 0:
                positions = answer_mask_cpu[b].nonzero(as_tuple=False).squeeze(-1)
                idx = torch.randint(len(positions), (1,), generator=g).item()
                mask_indices[j, b, positions[idx]] = True

    return T_cpu.to(device), mask_indices.to(device)


def compute_elbo(
    forward_fn: Callable,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    n_t: int,
    seed: int,
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
    attention_mask: Optional[torch.Tensor] = None,
    eps: float = DEFAULT_EPS,
    return_per_t: bool = False,
):
    """n_t-MC ELBO log-likelihood estimate per sample.

    B̂_π(y) = (1/n_t) Σ_j  −[(1/p_mask_j) · Σ_i∈M_j CE(x_i, logits_i)] / answer_len

    Args:
        forward_fn:    callable(noisy_ids: [B,L], attention_mask: [B,L]?) -> logits [B,L,V]
        input_ids:     [B, L] clean ids
        labels:        [B, L] -100 for prompt
        n_t:           number of timestep MC samples
        seed:          RNG seed (shared across πθ and πref for antithetic)
        mask_token_id: <|mdm_mask|> id
        attention_mask: [B, L] optional
        eps:           p_mask floor
        return_per_t:  if True, return per-timestep ELBO too

    Returns:
        elbo: [B] per-sample log-likelihood approximation (≤ 0)
        (per_t_elbo: [n_t, B] if return_per_t)
    """
    B, L = input_ids.shape
    device = input_ids.device

    answer_mask = labels != -100
    answer_lens = answer_mask.sum(dim=1).float().clamp(min=1.0)  # [B]

    T, mask_indices_all = sample_shared_TM(
        input_ids=input_ids, labels=labels, n_t=n_t, seed=seed, eps=eps
    )
    p_mask = (1.0 - eps) * T + eps  # [n_t, B], on device

    per_t_elbo = torch.zeros(n_t, B, device=device)

    for j in range(n_t):
        mask_j = mask_indices_all[j]  # [B, L] bool
        noisy_ids = torch.where(mask_j, mask_token_id, input_ids)

        logits = forward_fn(noisy_ids, attention_mask)  # [B, L, V]

        # Per-token NLL: -log p(x_i | y_t, prompt)
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_nll = -log_probs.gather(
            dim=-1, index=input_ids.unsqueeze(-1)
        ).squeeze(-1)  # [B, L]

        # Mask out unmasked positions, importance-weight by 1/p_mask
        per_token_nll = per_token_nll * mask_j.float()
        weighted = per_token_nll / p_mask[j].unsqueeze(-1)  # [B, L]

        nll_per_sample = weighted.sum(dim=1) / answer_lens  # [B]
        per_t_elbo[j] = -nll_per_sample  # log p ≈ -NLL

    elbo = per_t_elbo.mean(dim=0)  # [B]

    if return_per_t:
        return elbo, per_t_elbo
    return elbo


def compute_dpo_e_score(
    elbo_theta_w: torch.Tensor,
    elbo_ref_w: torch.Tensor,
    elbo_theta_l: torch.Tensor,
    elbo_ref_l: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """DPO-E preference score from four ELBO estimates.

    ŝ = β · [(B̂_θ(y_w) − B̂_ref(y_w)) − (B̂_θ(y_l) − B̂_ref(y_l))]

    All inputs are [B] tensors.
    """
    r_w = beta * (elbo_theta_w - elbo_ref_w)
    r_l = beta * (elbo_theta_l - elbo_ref_l)
    return r_w - r_l  # margin
