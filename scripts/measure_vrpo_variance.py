"""
Phase 0 variance measurement for V-MolPO design.

Purpose: empirically validate VRPO Theorem 2 (V[B̂] ∝ 1/n_t) and the antithetic
variance reduction effect on REAL LLaDA + ChEBI captioning data, before
implementing the rest of V-MolPO. Outputs a table that informs the choice of
n_t for Phase 1+.

Usage:
    cd /opt/EMNLP_MolDA/New_MolDA
    python scripts/measure_vrpo_variance.py \
        +experiment=selfies_dict \
        trainer=stage1 \
        pretrained_ckpt_path=./checkpoint/selfies_dict/stage1/last.ckpt \
        +phase0.n_pairs=8 \
        +phase0.n_trials=100 \
        +phase0.weight_perturb=0.001

Interpretation:
    V[B̂(n_t)] / V[B̂(n_t=1)]  →  expected ≈ 1/n_t (Theorem 2)
    V_shared[ŝ] / V_indep[ŝ]   →  < 1 when Corr(B̂_θ, B̂_ref) > 0 (Theorem 3)
"""
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.set_float32_matmul_precision("medium")

# PyTorch 2.6+ ckpt compatibility (matches scripts/train.py)
_orig_torch_load = torch.load
def _trusted_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _trusted_torch_load

import copy
import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.trainer import MolDATrainer
from src.training.vrpo_elbo import compute_elbo
from src.data.datamodule import MolDADataModule


def make_forward_fn(molda):
    """Create a forward callable that wraps MolDA's LLaDA model.

    For Phase 0, we use string-only forward (no graph injection) to keep the
    measurement focused on the diffusion ELBO variance, not multimodal effects.
    """
    @torch.no_grad()
    def fwd(noisy_ids, attention_mask=None):
        # molda.llada is the LLaDAWrapper; .model is the PEFT-wrapped HF model
        out = molda.llada.model(
            input_ids=noisy_ids,
            attention_mask=attention_mask,
        )
        return out.logits
    return fwd


def perturb_model(molda, scale: float):
    """Make a deep copy of molda.llada.model with weights perturbed by scale*N(0,1).

    Used as a stand-in for π_ref ≠ π_θ when both come from same ckpt.
    """
    model_copy = copy.deepcopy(molda.llada.model)
    g = torch.Generator(device="cpu").manual_seed(0)
    for p in model_copy.parameters():
        if p.dtype.is_floating_point:
            noise = torch.randn(p.shape, generator=g, dtype=torch.float32) * scale
            p.data.add_(noise.to(p.device, dtype=p.dtype))
    return model_copy


def make_perturbed_forward_fn(model_copy):
    @torch.no_grad()
    def fwd(noisy_ids, attention_mask=None):
        return model_copy(input_ids=noisy_ids, attention_mask=attention_mask).logits
    return fwd


def measure_theorem2(fwd_theta, batch, mask_id, n_t_list, n_trials, eps_list, device):
    """V[B̂_θ(y)] vs n_t for several eps."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attn = batch.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)

    rows = []
    for eps in eps_list:
        for n_t in n_t_list:
            elbos = []
            for trial in range(n_trials):
                e = compute_elbo(
                    fwd_theta, input_ids, labels,
                    n_t=n_t, seed=trial * 7919 + n_t * 113,
                    mask_token_id=mask_id, attention_mask=attn, eps=eps,
                )
                elbos.append(e.cpu())
            stack = torch.stack(elbos, dim=0)  # [trials, B]
            v = stack.var(dim=0).mean().item()
            rows.append((eps, n_t, v))
    return rows


def measure_theorem3(fwd_theta, fwd_ref, batch_w, batch_l, mask_id,
                     n_t, n_trials, eps, device, beta=0.1):
    """V[ŝ] with shared seed (antithetic) vs independent seeds."""
    def to_dev(b):
        ids = b["input_ids"].to(device)
        lab = b["labels"].to(device)
        am = b.get("attention_mask")
        if am is not None:
            am = am.to(device)
        return ids, lab, am

    ids_w, lab_w, attn_w = to_dev(batch_w)
    ids_l, lab_l, attn_l = to_dev(batch_l)

    def trial(seed_w_theta, seed_w_ref, seed_l_theta, seed_l_ref):
        e_tw = compute_elbo(fwd_theta, ids_w, lab_w, n_t=n_t, seed=seed_w_theta,
                            mask_token_id=mask_id, attention_mask=attn_w, eps=eps)
        e_rw = compute_elbo(fwd_ref, ids_w, lab_w, n_t=n_t, seed=seed_w_ref,
                            mask_token_id=mask_id, attention_mask=attn_w, eps=eps)
        e_tl = compute_elbo(fwd_theta, ids_l, lab_l, n_t=n_t, seed=seed_l_theta,
                            mask_token_id=mask_id, attention_mask=attn_l, eps=eps)
        e_rl = compute_elbo(fwd_ref, ids_l, lab_l, n_t=n_t, seed=seed_l_ref,
                            mask_token_id=mask_id, attention_mask=attn_l, eps=eps)
        return beta * ((e_tw - e_rw) - (e_tl - e_rl))  # [B]

    # Shared seeds: same seed for θ/ref on each y
    shared = []
    for k in range(n_trials):
        s_w = k * 1000
        s_l = k * 1000 + 7
        shared.append(trial(s_w, s_w, s_l, s_l).cpu())
    shared_stack = torch.stack(shared, dim=0)
    v_shared = shared_stack.var(dim=0).mean().item()

    # Independent seeds
    indep = []
    for k in range(n_trials):
        indep.append(
            trial(k * 1000, k * 1000 + 333, k * 1000 + 7, k * 1000 + 7777).cpu()
        )
    indep_stack = torch.stack(indep, dim=0)
    v_indep = indep_stack.var(dim=0).mean().item()

    return v_shared, v_indep


def get_chebi_batches(dm, n_pairs):
    """Pull two batches (chosen-side, rejected-side) from val dataloader.

    For Phase 0 variance measurement, we don't need real preferences — any two
    sample groups suffice. Use first n_pairs samples as 'y_w' and next as 'y_l'.
    """
    dm.setup("fit")
    val_loader = dm.val_dataloader()
    if isinstance(val_loader, list):
        val_loader = val_loader[0]
    batches = []
    for b in val_loader:
        batches.append(b)
        if len(batches) >= 2:
            break

    if len(batches) < 2:
        # fall back to splitting one batch in half
        b = batches[0]
        B = b["input_ids"].shape[0]
        assert B >= 2
        half = B // 2
        b_w = {k: (v[:half] if torch.is_tensor(v) else v) for k, v in b.items()}
        b_l = {k: (v[half:half * 2] if torch.is_tensor(v) else v) for k, v in b.items()}
        return b_w, b_l

    # Trim to n_pairs
    b_w, b_l = batches[0], batches[1]
    if b_w["input_ids"].shape[0] > n_pairs:
        b_w = {k: (v[:n_pairs] if torch.is_tensor(v) else v) for k, v in b_w.items()}
    if b_l["input_ids"].shape[0] > n_pairs:
        b_l = {k: (v[:n_pairs] if torch.is_tensor(v) else v) for k, v in b_l.items()}
    return b_w, b_l


def print_theorem2_table(rows, baseline_n_t=1):
    print("\n" + "=" * 78)
    print("Theorem 2 — V[B̂(y_w; n_t)] vs n_t (single-model ELBO variance)")
    print("=" * 78)
    print(f"{'eps':>10}  {'n_t':>4}  {'V[B̂]':>14}  {'ratio':>8}  {'ideal 1/n_t':>12}")
    print("-" * 78)
    by_eps = {}
    for eps, n_t, v in rows:
        by_eps.setdefault(eps, {})[n_t] = v
    for eps, d in by_eps.items():
        v_base = d.get(baseline_n_t)
        for n_t in sorted(d.keys()):
            v = d[n_t]
            ratio = v / v_base if v_base else float("nan")
            ideal = baseline_n_t / n_t
            print(f"{eps:>10.3g}  {n_t:>4d}  {v:>14.6f}  {ratio:>8.4f}  {ideal:>12.4f}")
        print("-" * 78)


def print_theorem3(v_shared, v_indep):
    print("\n" + "=" * 78)
    print("Theorem 3 — V[ŝ] with shared (T,M) (antithetic) vs independent seeds")
    print("=" * 78)
    print(f"  V[ŝ]   shared seeds (antithetic on)   = {v_shared:.6f}")
    print(f"  V[ŝ]   independent seeds (off)        = {v_indep:.6f}")
    print(f"  ratio  shared / indep                 = {v_shared / v_indep:.4f}")
    print(f"  expected: < 1 when Corr(B̂_θ, B̂_ref) > 0")
    print("=" * 78)


@hydra.main(config_path="../src/configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    # Phase 0 hyperparameters with sensible defaults
    p0 = cfg.get("phase0", {})
    n_pairs = int(p0.get("n_pairs", 8))
    n_trials = int(p0.get("n_trials", 100))
    perturb = float(p0.get("weight_perturb", 0.001))
    n_t_list = list(p0.get("n_t_list", [1, 2, 4, 8]))
    eps_list = list(p0.get("eps_list", [1e-3, 0.05, 0.1]))
    beta = float(p0.get("beta", 0.1))

    print(f"Phase 0 config: n_pairs={n_pairs}, n_trials={n_trials}, "
          f"perturb={perturb}, n_t_list={n_t_list}, eps_list={eps_list}, beta={beta}")

    # Build model (loads pretrained_ckpt if specified)
    print("\n[1/4] Building MolDATrainer (loads ckpt if pretrained_ckpt_path set) ...")
    trainer = MolDATrainer(cfg)
    if torch.cuda.is_available():
        trainer = trainer.cuda()
    trainer.eval()
    molda = trainer.model
    mask_id = trainer.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
    if mask_id is None or mask_id == trainer.tokenizer.unk_token_id:
        # Fallback to default LLaDA mask token id
        from src.training.vrpo_elbo import DEFAULT_MASK_TOKEN_ID
        mask_id = DEFAULT_MASK_TOKEN_ID
    print(f"  mask_token_id = {mask_id}")

    # Build perturbed model as a stand-in π_ref ≠ π_θ
    print(f"\n[2/4] Creating perturbed model (perturb={perturb}) as π_ref ...")
    ref_model = perturb_model(molda, scale=perturb)
    ref_model.eval()
    fwd_theta = make_forward_fn(molda)
    fwd_ref = make_perturbed_forward_fn(ref_model)

    # Get data
    print("\n[3/4] Loading val batches from datamodule ...")
    dm = MolDADataModule(tokenizer=trainer.tokenizer, cfg=cfg)
    batch_w, batch_l = get_chebi_batches(dm, n_pairs)
    device = next(molda.parameters()).device
    print(f"  batch_w: input_ids {tuple(batch_w['input_ids'].shape)}")
    print(f"  batch_l: input_ids {tuple(batch_l['input_ids'].shape)}")

    # Theorem 2 measurement
    print(f"\n[4/4] Measuring V[B̂] vs n_t (n_trials={n_trials}) ...")
    rows = measure_theorem2(
        fwd_theta=fwd_theta, batch=batch_w, mask_id=mask_id,
        n_t_list=n_t_list, n_trials=n_trials, eps_list=eps_list, device=device,
    )
    print_theorem2_table(rows)

    # Theorem 3 measurement (use middle n_t and middle eps)
    n_t_mid = n_t_list[len(n_t_list) // 2]
    eps_mid = eps_list[len(eps_list) // 2]
    print(f"\n[Theorem 3 grid: n_t={n_t_mid}, eps={eps_mid}]")
    v_shared, v_indep = measure_theorem3(
        fwd_theta=fwd_theta, fwd_ref=fwd_ref,
        batch_w=batch_w, batch_l=batch_l, mask_id=mask_id,
        n_t=n_t_mid, n_trials=n_trials, eps=eps_mid, device=device, beta=beta,
    )
    print_theorem3(v_shared, v_indep)

    print("\nDone.")


if __name__ == "__main__":
    main()
