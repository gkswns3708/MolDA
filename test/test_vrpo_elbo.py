"""Unit tests for src/training/vrpo_elbo.py.

Validates VRPO claims (LLaDA 1.5, arXiv:2505.19223):
  - Theorem 2: V[B̂_π] ∝ 1/n_t  (more timestep MC samples → less variance)
  - Theorem 3: shared (T, M) noise → V[B̂_θ − B̂_ref] reduced when Corr > 0
  - Reproducibility: same seed → identical (T, M) and identical ELBO output
  - NaN safety: empty answer → no NaN
"""
import torch
import pytest

from src.training.vrpo_elbo import (
    compute_elbo,
    sample_shared_TM,
    compute_dpo_e_score,
    DEFAULT_MASK_TOKEN_ID,
)


# ─────────────────────────────────────────────────────────────────────
# Mock model factory — deterministic forward without loading LLaDA-8B
# ─────────────────────────────────────────────────────────────────────

def make_mock_forward(vocab_size: int, hidden: int = 16, weight_seed: int = 0):
    """Create a deterministic mock forward fn: noisy_ids → logits.

    Output is a function of (noisy_ids, weights) only. Different `weight_seed`
    gives a different model; the SAME seed always produces identical output.
    """
    g = torch.Generator()
    g.manual_seed(weight_seed)
    embed = torch.randn(vocab_size, hidden, generator=g)
    proj = torch.randn(hidden, vocab_size, generator=g) * 0.5

    def fwd(noisy_ids, attention_mask=None):
        h = embed[noisy_ids]  # [B, L, H]
        return h @ proj       # [B, L, V]

    return fwd


def make_correlated_mock(vocab_size: int, hidden: int = 16,
                        weight_seed: int = 0, perturb: float = 0.05):
    """Make a model = base model + small perturbation. Mimics SFT relationship."""
    g_base = torch.Generator()
    g_base.manual_seed(weight_seed)
    embed_base = torch.randn(vocab_size, hidden, generator=g_base)
    proj_base = torch.randn(hidden, vocab_size, generator=g_base) * 0.5

    g_p = torch.Generator()
    g_p.manual_seed(weight_seed + 1000)
    embed = embed_base + perturb * torch.randn(vocab_size, hidden, generator=g_p)
    proj = proj_base + perturb * torch.randn(hidden, vocab_size, generator=g_p) * 0.5

    def fwd(noisy_ids, attention_mask=None):
        h = embed[noisy_ids]
        return h @ proj

    return fwd


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

VOCAB = 200
B = 4
L = 32
PROMPT_LEN = 8
MASK_ID = DEFAULT_MASK_TOKEN_ID  # outside [0, VOCAB) so we use a smaller stand-in below


@pytest.fixture
def small_mask_id():
    """Override MASK_TOKEN_ID for mock vocab (real LLaDA mask=126336, our mock vocab=200)."""
    return VOCAB - 1  # use last token as mask


@pytest.fixture
def fake_batch(small_mask_id):
    """Construct a small batch:
       - input_ids in [0, VOCAB-1) so mask_id (VOCAB-1) is not a real token
       - labels: -100 for first PROMPT_LEN positions, real ids elsewhere
    """
    g = torch.Generator()
    g.manual_seed(42)
    input_ids = torch.randint(0, VOCAB - 1, (B, L), generator=g)
    labels = input_ids.clone()
    labels[:, :PROMPT_LEN] = -100
    return input_ids, labels


# ─────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────

def test_sample_shared_TM_deterministic(fake_batch):
    """Same seed → identical (T, M)."""
    input_ids, labels = fake_batch

    T1, M1 = sample_shared_TM(input_ids, labels, n_t=4, seed=123)
    T2, M2 = sample_shared_TM(input_ids, labels, n_t=4, seed=123)

    assert torch.equal(T1, T2)
    assert torch.equal(M1, M2)


def test_sample_shared_TM_different_seeds(fake_batch):
    """Different seeds → different (T, M)."""
    input_ids, labels = fake_batch

    T1, _ = sample_shared_TM(input_ids, labels, n_t=4, seed=1)
    T2, _ = sample_shared_TM(input_ids, labels, n_t=4, seed=2)

    assert not torch.equal(T1, T2)


def test_compute_elbo_deterministic(fake_batch, small_mask_id):
    """Same seed + same model → identical ELBO output."""
    input_ids, labels = fake_batch
    fwd = make_mock_forward(VOCAB, weight_seed=0)

    e1 = compute_elbo(fwd, input_ids, labels, n_t=4, seed=7,
                     mask_token_id=small_mask_id)
    e2 = compute_elbo(fwd, input_ids, labels, n_t=4, seed=7,
                     mask_token_id=small_mask_id)

    assert torch.equal(e1, e2)


# ─────────────────────────────────────────────────────────────────────
# Theorem 2: V[B̂] ∝ 1/n_t
# ─────────────────────────────────────────────────────────────────────

def test_variance_decrease_with_n_t(fake_batch, small_mask_id):
    """V[B̂] should approximately scale as 1/n_t (Theorem 2).

    Use eps=0.1 so 1/p_mask ∈ [1, 10] (bounded amplification).
    With default eps=1e-3 the per-t loss is heavy-tailed (1/p_mask up to 1000)
    and 50 trials are insufficient to estimate empirical variance reliably.
    For full Phase 0 measurement on real LLaDA, see scripts/measure_vrpo_variance.py.
    """
    input_ids, labels = fake_batch
    fwd = make_mock_forward(VOCAB, weight_seed=0)
    eps = 0.1

    n_trials = 200
    variances = {}
    for n_t in [1, 2, 4, 8]:
        elbos = []
        for trial in range(n_trials):
            e = compute_elbo(fwd, input_ids, labels, n_t=n_t,
                             seed=trial * 1000 + n_t * 17,
                             mask_token_id=small_mask_id, eps=eps)
            elbos.append(e)
        elbos_stack = torch.stack(elbos, dim=0)  # [n_trials, B]
        variances[n_t] = elbos_stack.var(dim=0).mean().item()

    # Theorem 2: V(n_t=8)/V(n_t=1) ≈ 1/8 = 0.125.
    # 200 trials → SE of variance estimate ≈ 10% per measurement, accept [0.04, 0.30].
    ratio = variances[8] / variances[1]
    assert 0.04 <= ratio <= 0.30, (
        f"V(n_t=8)/V(n_t=1) = {ratio:.4f}, expected ≈ 0.125. variances={variances}"
    )
    # Monotonic (with 30% slack for noise)
    assert variances[2] < variances[1] * 1.3, f"variances={variances}"
    assert variances[4] < variances[2] * 1.3, f"variances={variances}"
    assert variances[8] < variances[4] * 1.3, f"variances={variances}"


# ─────────────────────────────────────────────────────────────────────
# Theorem 3: antithetic noise sharing reduces V[B̂_θ − B̂_ref]
# ─────────────────────────────────────────────────────────────────────

def test_antithetic_variance_reduction(fake_batch, small_mask_id):
    """V[B̂_θ - B̂_ref] with shared seed should be lower than with independent seeds
    when the two models are correlated (Corr > 0).
    """
    input_ids, labels = fake_batch

    # Two correlated mock models (small perturbation)
    fwd_theta = make_correlated_mock(VOCAB, weight_seed=0, perturb=0.05)
    fwd_ref = make_correlated_mock(VOCAB, weight_seed=0, perturb=0.05)
    # NOTE: same weight_seed=0 → embed_base/proj_base identical → only perturbations differ
    # So fwd_theta and fwd_ref are very similar. This mimics πθ ≈ πref (SFT init).

    # Sanity: re-build with same args gives different perturbation each call?
    # No — make_correlated_mock uses weight_seed+1000 deterministic. Both calls produce
    # IDENTICAL outputs. We need them slightly different. Build θ from base, ref from base+δ.

    # Re-do: θ = base (perturb=0), ref = base + small perturb
    fwd_theta = make_mock_forward(VOCAB, weight_seed=0)
    fwd_ref = make_correlated_mock(VOCAB, weight_seed=0, perturb=0.1)

    n_trials = 50
    n_t = 4

    # (a) Shared seed (antithetic ON)
    diffs_shared = []
    for trial in range(n_trials):
        seed = trial * 1000
        e_theta = compute_elbo(fwd_theta, input_ids, labels, n_t=n_t, seed=seed,
                               mask_token_id=small_mask_id)
        e_ref = compute_elbo(fwd_ref, input_ids, labels, n_t=n_t, seed=seed,
                             mask_token_id=small_mask_id)
        diffs_shared.append(e_theta - e_ref)
    diffs_shared = torch.stack(diffs_shared, dim=0)
    V_shared = diffs_shared.var(dim=0).mean().item()

    # (b) Independent seeds (antithetic OFF)
    diffs_indep = []
    for trial in range(n_trials):
        e_theta = compute_elbo(fwd_theta, input_ids, labels, n_t=n_t, seed=trial * 7,
                               mask_token_id=small_mask_id)
        e_ref = compute_elbo(fwd_ref, input_ids, labels, n_t=n_t, seed=trial * 11 + 5,
                             mask_token_id=small_mask_id)
        diffs_indep.append(e_theta - e_ref)
    diffs_indep = torch.stack(diffs_indep, dim=0)
    V_indep = diffs_indep.var(dim=0).mean().item()

    # Theorem 3: V_shared < V_indep when Corr > 0
    assert V_shared < V_indep, (
        f"Antithetic should reduce variance. V_shared={V_shared:.6f}, "
        f"V_indep={V_indep:.6f}"
    )

    # Stronger: ratio should be at most ~0.7 (depends on Corr strength)
    ratio = V_shared / V_indep
    assert ratio < 0.7, (
        f"Antithetic ratio V_shared/V_indep={ratio:.4f}, expected < 0.7"
    )


# ─────────────────────────────────────────────────────────────────────
# NaN safety
# ─────────────────────────────────────────────────────────────────────

def test_zero_answer_no_nan(small_mask_id):
    """Sample with all-prompt labels (no answer) should not produce NaN."""
    input_ids = torch.randint(0, VOCAB - 1, (2, L))
    labels = torch.full((2, L), -100)  # all prompt — no answer

    fwd = make_mock_forward(VOCAB, weight_seed=0)
    e = compute_elbo(fwd, input_ids, labels, n_t=4, seed=42,
                     mask_token_id=small_mask_id)

    # No NaN, finite values
    assert torch.isfinite(e).all(), f"ELBO has non-finite values: {e}"
    # When no answer, ELBO should be 0 (no contribution)
    assert torch.allclose(e, torch.zeros_like(e), atol=1e-5)


def test_partial_answer_no_nan(small_mask_id):
    """Mixed batch: some samples with answer, some without. No NaN."""
    input_ids = torch.randint(0, VOCAB - 1, (3, L))
    labels = input_ids.clone()
    labels[0, :] = -100   # sample 0 has no answer
    labels[1, :PROMPT_LEN] = -100  # sample 1 has answer
    labels[2, :PROMPT_LEN] = -100  # sample 2 has answer

    fwd = make_mock_forward(VOCAB, weight_seed=0)
    e = compute_elbo(fwd, input_ids, labels, n_t=4, seed=42,
                     mask_token_id=small_mask_id)

    assert torch.isfinite(e).all()
    assert torch.allclose(e[0], torch.tensor(0.0), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
# Shape / API checks
# ─────────────────────────────────────────────────────────────────────

def test_elbo_shape(fake_batch, small_mask_id):
    """Output shape [B] regardless of n_t."""
    input_ids, labels = fake_batch
    fwd = make_mock_forward(VOCAB, weight_seed=0)

    for n_t in [1, 4, 8]:
        e = compute_elbo(fwd, input_ids, labels, n_t=n_t, seed=0,
                         mask_token_id=small_mask_id)
        assert e.shape == (B,)


def test_return_per_t(fake_batch, small_mask_id):
    """return_per_t=True returns [n_t, B] alongside [B]."""
    input_ids, labels = fake_batch
    fwd = make_mock_forward(VOCAB, weight_seed=0)

    n_t = 4
    elbo, per_t = compute_elbo(fwd, input_ids, labels, n_t=n_t, seed=0,
                                mask_token_id=small_mask_id, return_per_t=True)
    assert elbo.shape == (B,)
    assert per_t.shape == (n_t, B)
    assert torch.allclose(elbo, per_t.mean(dim=0))


def test_elbo_negative_or_zero(fake_batch, small_mask_id):
    """ELBO ≈ log p(y|x), so ELBO ≤ 0 (negative NLL)."""
    input_ids, labels = fake_batch
    fwd = make_mock_forward(VOCAB, weight_seed=0)

    e = compute_elbo(fwd, input_ids, labels, n_t=4, seed=0,
                     mask_token_id=small_mask_id)
    assert (e <= 1e-5).all(), f"ELBO must be ≤ 0, got {e}"


# ─────────────────────────────────────────────────────────────────────
# DPO-E margin helper
# ─────────────────────────────────────────────────────────────────────

def test_dpo_e_score_zero_when_theta_equals_ref():
    """If πθ == πref, ŝ = 0 for any (y_w, y_l)."""
    B = 4
    e_theta_w = torch.tensor([-1.5, -2.0, -1.8, -1.2])
    e_ref_w = e_theta_w.clone()
    e_theta_l = torch.tensor([-2.5, -2.8, -2.4, -2.0])
    e_ref_l = e_theta_l.clone()

    margin = compute_dpo_e_score(e_theta_w, e_ref_w, e_theta_l, e_ref_l, beta=0.1)
    assert torch.allclose(margin, torch.zeros(B))


def test_dpo_e_score_sign_flip():
    """Swapping (y_w, y_l) flips margin sign."""
    e_theta_w = torch.tensor([-1.0])
    e_ref_w = torch.tensor([-2.0])
    e_theta_l = torch.tensor([-3.0])
    e_ref_l = torch.tensor([-2.5])

    m_wl = compute_dpo_e_score(e_theta_w, e_ref_w, e_theta_l, e_ref_l, beta=1.0)
    m_lw = compute_dpo_e_score(e_theta_l, e_ref_l, e_theta_w, e_ref_w, beta=1.0)
    assert torch.allclose(m_wl, -m_lw)
