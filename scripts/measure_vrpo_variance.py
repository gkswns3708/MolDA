"""
Phase 0 variance measurement for V-MolPO design.

Purpose: empirically validate VRPO Theorem 2 (V[B̂] ∝ 1/n_t) and Theorem 3
(antithetic variance reduction) on REAL LLaDA + Stage 2 ckpt + ChEBI data,
before implementing the rest of stage 3 V-MolPO.

DDP: trials are sharded across ranks (each rank holds its own model copy).
Use torchrun for multi-GPU acceleration.

Usage (single GPU):
    cd /opt/EMNLP_MolDA/New_MolDA
    python scripts/measure_vrpo_variance.py \\
        +experiment=selfies_dict_rephrase_stage2 trainer=stage2 \\
        pretrained_ckpt_path=./checkpoint/.../stage2/last.ckpt \\
        hardware.devices="'0'" \\
        +phase0.n_trials=80

Usage (DDP, 6 GPUs):
    torchrun --standalone --nproc_per_node=6 scripts/measure_vrpo_variance.py \\
        +experiment=selfies_dict_rephrase_stage2 trainer=stage2 \\
        pretrained_ckpt_path=./checkpoint/.../stage2/last.ckpt \\
        hardware.devices="'0,1,2,3,4,5'" \\
        +phase0.n_trials=80

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
import torch.distributed as dist
torch.set_float32_matmul_precision("medium")

# PyTorch 2.6+ ckpt compat (matches scripts/train.py)
_orig_torch_load = torch.load
def _trusted_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _trusted_torch_load

import copy
import hydra
from omegaconf import DictConfig

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.trainer import MolDATrainer
from src.training.vrpo_elbo import compute_elbo, DEFAULT_MASK_TOKEN_ID
from src.data.datamodule import MolDADataModule


# ─────────────────────────────────────────────────────────────────
# Distributed setup
# ─────────────────────────────────────────────────────────────────

def init_dist():
    """Initialize torch.distributed if launched via torchrun, else single-rank."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def rank0_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────
# Forward functions (string-only LLaDA forward; graphs are constant noise
# for Phase 0 — including them would only add a fixed bias, not variance)
# ─────────────────────────────────────────────────────────────────

def make_forward_fn(molda):
    @torch.no_grad()
    def fwd(noisy_ids, attention_mask=None):
        return molda.llada.model(
            input_ids=noisy_ids, attention_mask=attention_mask
        ).logits
    return fwd


def perturb_model(molda, scale: float):
    """Deep copy of molda.llada.model with weights += scale·N(0,1).

    Stand-in for π_ref ≠ π_θ when both come from the same ckpt (mimics SFT step).
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


# ─────────────────────────────────────────────────────────────────
# Distributed trial sharding + gather
# ─────────────────────────────────────────────────────────────────

def my_trial_indices(n_trials, rank, world_size):
    """Interleaved sharding: rank r gets trials [r, r+W, r+2W, ...]."""
    return list(range(rank, n_trials, world_size))


def gather_elbos(my_elbos, my_trial_idx, n_trials, B, rank, world_size, device):
    """Gather per-trial ELBOs from all ranks → tensor [n_trials, B] on rank 0."""
    if world_size == 1:
        # my_trial_idx == [0..n_trials-1], my_elbos shape [n_trials, B]
        return my_elbos
    # Each rank sends (trial_idx_list, elbos_list) as a Python object
    payload = (my_trial_idx, my_elbos.cpu().tolist())
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(payload, gathered, dst=0)
    if rank != 0:
        return None
    full = torch.zeros(n_trials, B)
    for r_idx, (idxs, vals) in enumerate(gathered):
        for k, t_idx in enumerate(idxs):
            full[t_idx] = torch.tensor(vals[k])
    return full


# ─────────────────────────────────────────────────────────────────
# Theorem 2: V[B̂_θ(y)] vs n_t
# ─────────────────────────────────────────────────────────────────

def measure_theorem2(fwd_theta, batch, mask_id, n_t_list, n_trials, eps_list,
                     device, rank, world_size):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attn = batch.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    B = input_ids.shape[0]

    rows = []
    my_idx = my_trial_indices(n_trials, rank, world_size)

    for eps in eps_list:
        for n_t in n_t_list:
            my_elbos = []
            for trial in my_idx:
                e = compute_elbo(
                    fwd_theta, input_ids, labels,
                    n_t=n_t, seed=trial * 7919 + n_t * 113,
                    mask_token_id=mask_id, attention_mask=attn, eps=eps,
                )
                my_elbos.append(e.cpu())
            my_elbos = torch.stack(my_elbos, dim=0) if my_elbos else torch.zeros(0, B)
            full = gather_elbos(my_elbos, my_idx, n_trials, B, rank, world_size, device)
            if rank == 0:
                v = full.var(dim=0).mean().item()
                rows.append((eps, n_t, v))
                rank0_print(rank,
                            f"  [Theorem2] eps={eps:>6.3g}  n_t={n_t:>2d}  V[B̂]={v:.6f}")
    return rows


# ─────────────────────────────────────────────────────────────────
# Theorem 3: V[ŝ] shared (T,M) vs independent
# ─────────────────────────────────────────────────────────────────

def measure_theorem3(fwd_theta, fwd_ref, batch_w, batch_l, mask_id,
                     n_t, n_trials, eps, device, rank, world_size, beta=0.1):
    def to_dev(b):
        ids = b["input_ids"].to(device)
        lab = b["labels"].to(device)
        am = b.get("attention_mask")
        if am is not None:
            am = am.to(device)
        return ids, lab, am

    ids_w, lab_w, attn_w = to_dev(batch_w)
    ids_l, lab_l, attn_l = to_dev(batch_l)
    Bw = ids_w.shape[0]

    def trial_fn(seed_w_t, seed_w_r, seed_l_t, seed_l_r):
        e_tw = compute_elbo(fwd_theta, ids_w, lab_w, n_t=n_t, seed=seed_w_t,
                            mask_token_id=mask_id, attention_mask=attn_w, eps=eps)
        e_rw = compute_elbo(fwd_ref,   ids_w, lab_w, n_t=n_t, seed=seed_w_r,
                            mask_token_id=mask_id, attention_mask=attn_w, eps=eps)
        e_tl = compute_elbo(fwd_theta, ids_l, lab_l, n_t=n_t, seed=seed_l_t,
                            mask_token_id=mask_id, attention_mask=attn_l, eps=eps)
        e_rl = compute_elbo(fwd_ref,   ids_l, lab_l, n_t=n_t, seed=seed_l_r,
                            mask_token_id=mask_id, attention_mask=attn_l, eps=eps)
        return beta * ((e_tw - e_rw) - (e_tl - e_rl))

    my_idx = my_trial_indices(n_trials, rank, world_size)

    # Shared seeds (antithetic ON)
    my_shared = []
    for k in my_idx:
        s_w, s_l = k * 1000, k * 1000 + 7
        my_shared.append(trial_fn(s_w, s_w, s_l, s_l).cpu())
    my_shared = torch.stack(my_shared, dim=0) if my_shared else torch.zeros(0, Bw)
    full_shared = gather_elbos(my_shared, my_idx, n_trials, Bw, rank, world_size, device)

    # Independent seeds (antithetic OFF)
    my_indep = []
    for k in my_idx:
        my_indep.append(
            trial_fn(k * 1000, k * 1000 + 333, k * 1000 + 7, k * 1000 + 7777).cpu()
        )
    my_indep = torch.stack(my_indep, dim=0) if my_indep else torch.zeros(0, Bw)
    full_indep = gather_elbos(my_indep, my_idx, n_trials, Bw, rank, world_size, device)

    if rank == 0:
        v_shared = full_shared.var(dim=0).mean().item()
        v_indep = full_indep.var(dim=0).mean().item()
        return v_shared, v_indep
    return None, None


# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────

def get_chebi_batches(dm, n_pairs):
    """Pull two TRAIN batches (TrainCollator → input_ids+labels).

    val loader uses EvalCollator (prompt-only, no labels) which is unsuitable
    for ELBO measurement. Use train loader instead.
    """
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    batches = []
    for b in train_loader:
        if "input_ids" not in b:
            continue
        batches.append(b)
        if len(batches) >= 2:
            break
    if len(batches) < 2:
        if not batches:
            raise RuntimeError("Train dataloader produced no batches with 'input_ids'.")
        b = batches[0]
        B = b["input_ids"].shape[0]
        half = B // 2
        if half < 1:
            raise RuntimeError(f"Train batch size {B} too small to split.")
        return (
            {k: (v[:half] if torch.is_tensor(v) else v) for k, v in b.items()},
            {k: (v[half:half * 2] if torch.is_tensor(v) else v) for k, v in b.items()},
        )
    bw, bl = batches[0], batches[1]
    if bw["input_ids"].shape[0] > n_pairs:
        bw = {k: (v[:n_pairs] if torch.is_tensor(v) else v) for k, v in bw.items()}
    if bl["input_ids"].shape[0] > n_pairs:
        bl = {k: (v[:n_pairs] if torch.is_tensor(v) else v) for k, v in bl.items()}
    return bw, bl


# ─────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────

def print_theorem2_table(rows, baseline_n_t=1):
    print("\n" + "=" * 78)
    print("Theorem 2 — V[B̂(y_w; n_t)] vs n_t  (single-model ELBO variance)")
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


def print_theorem3(v_shared, v_indep, n_t, eps):
    print("\n" + "=" * 78)
    print(f"Theorem 3 — V[ŝ] (n_t={n_t}, eps={eps})")
    print("=" * 78)
    print(f"  shared seeds (antithetic ON)   V[ŝ] = {v_shared:.6f}")
    print(f"  independent seeds (antithetic OFF) V[ŝ] = {v_indep:.6f}")
    print(f"  ratio  shared / indep                = {v_shared / v_indep:.4f}")
    print("  expected: < 1 when Corr(B̂_θ, B̂_ref) > 0")
    print("=" * 78)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

@hydra.main(config_path="../src/configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    rank, local_rank, world_size = init_dist()

    p0 = cfg.get("phase0", {})
    n_pairs = int(p0.get("n_pairs", 8))
    n_trials = int(p0.get("n_trials", 80))
    perturb = float(p0.get("weight_perturb", 0.001))
    n_t_list = list(p0.get("n_t_list", [1, 2, 4, 8]))
    eps_list = list(p0.get("eps_list", [1e-3, 0.05, 0.1]))
    beta = float(p0.get("beta", 0.1))

    rank0_print(rank, "=" * 78)
    rank0_print(rank, "Phase 0 — VRPO variance measurement")
    rank0_print(rank, "=" * 78)
    rank0_print(rank,
                f"  world_size={world_size}, rank={rank}, local_rank={local_rank}")
    rank0_print(rank,
                f"  n_pairs={n_pairs}, n_trials={n_trials}, perturb={perturb}, beta={beta}")
    rank0_print(rank, f"  n_t_list={n_t_list}, eps_list={eps_list}")

    # Build model on this rank's GPU
    rank0_print(rank, "\n[1/4] Building MolDATrainer (loads ckpt if pretrained_ckpt_path set) ...")
    trainer = MolDATrainer(cfg)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    trainer = trainer.to(device)
    trainer.eval()
    molda = trainer.model

    mask_id = trainer.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
    if mask_id is None or mask_id == trainer.tokenizer.unk_token_id:
        mask_id = DEFAULT_MASK_TOKEN_ID
    rank0_print(rank, f"  mask_token_id = {mask_id}")

    # Perturbed copy = π_ref
    rank0_print(rank, f"\n[2/4] Creating perturbed model (perturb={perturb}) as π_ref ...")
    ref_model = perturb_model(molda, scale=perturb)
    ref_model.eval()
    fwd_theta = make_forward_fn(molda)
    fwd_ref = make_perturbed_forward_fn(ref_model)

    # Data (rank 0 only really uses it but every rank needs it for trials)
    rank0_print(rank, "\n[3/4] Loading val batches ...")
    dm = MolDADataModule(tokenizer=trainer.tokenizer, cfg=cfg)
    batch_w, batch_l = get_chebi_batches(dm, n_pairs)
    rank0_print(rank, f"  batch_w: {tuple(batch_w['input_ids'].shape)}")
    rank0_print(rank, f"  batch_l: {tuple(batch_l['input_ids'].shape)}")

    # Theorem 2
    rank0_print(rank, f"\n[4/4] Measuring V[B̂] vs n_t (n_trials={n_trials}, "
                f"sharded across {world_size} ranks) ...")
    rows = measure_theorem2(
        fwd_theta=fwd_theta, batch=batch_w, mask_id=mask_id,
        n_t_list=n_t_list, n_trials=n_trials, eps_list=eps_list,
        device=device, rank=rank, world_size=world_size,
    )
    if rank == 0:
        print_theorem2_table(rows)

    # Theorem 3 — single (n_t, eps) cell
    n_t_mid = n_t_list[len(n_t_list) // 2]
    eps_mid = eps_list[len(eps_list) // 2]
    rank0_print(rank, f"\n[Theorem 3 grid: n_t={n_t_mid}, eps={eps_mid}]")
    v_shared, v_indep = measure_theorem3(
        fwd_theta=fwd_theta, fwd_ref=fwd_ref,
        batch_w=batch_w, batch_l=batch_l, mask_id=mask_id,
        n_t=n_t_mid, n_trials=n_trials, eps=eps_mid,
        device=device, rank=rank, world_size=world_size, beta=beta,
    )
    if rank == 0:
        print_theorem3(v_shared, v_indep, n_t_mid, eps_mid)
        print("\nDone.")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
