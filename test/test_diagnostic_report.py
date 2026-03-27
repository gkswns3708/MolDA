"""
Diagnostic report generator: 학습 1 step의 전체 과정을 텍스트로 시각화.

실행:
    cd /opt/11-MolDA/New_MolDA
    source venvs/MolDA/bin/activate
    PYTHONPATH=/opt/11-MolDA/New_MolDA CUDA_VISIBLE_DEVICES=0 \
        python test/test_diagnostic_report.py

출력:
    test/diagnostic_report.md
"""

import os
import sys
import datetime

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "hf-cache"))

DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
ORIGINAL_VOCAB_SIZE = 126349
MASK_TOKEN_ID = 126336

# ─────────────────────────────────────────
# Config (conftest.py와 동일)
# ─────────────────────────────────────────
cfg = OmegaConf.create({
    "seed": 42, "mode": "ft", "stage": 1, "debug": False, "ckpt_path": None,
    "model": {
        "llm": "GSAI-ML/LLaDA-8B-Instruct", "tune_llm": "lora", "tune_gnn": False,
        "mol_representation": "string_only", "original_vocab_size": ORIGINAL_VOCAB_SIZE,
    },
    "lora": {"r": 64, "alpha": 32, "dropout": 0.05, "config_path": None},
    "tokenizer": {
        "add_selfies_tokens": True,
        "selfies_token_path": os.path.join(PROJECT_ROOT, "src", "model", "selfies_dict.txt"),
    },
    "data": {
        "root": DATASET_ROOT,
        "splits": {"train": "Train_toy100", "val": "Val_toy100", "test": "Test_toy100"},
        "max_length": 512, "gen_max_len": 256, "truncation": True, "padding": "max_length", "min_len": 8,
    },
    "training": {
        "max_steps": -1, "max_epochs": 3, "batch_size": 2,
        "accumulate_grad_batches": 1, "weight_decay": 0.1, "gradient_clip_val": 1.0,
    },
    "scheduler": {"warmup_steps": 50, "decay_ratio": 0.1, "min_lr_ratio": 0.1},
    "lr": {
        "lora": 2.5e-3, "embed_orig": 2.5e-5, "embed_new": 2.5e-5,
        "head_orig": 2.5e-5, "head_new": 2.5e-5, "other": 0.0,
    },
    "hardware": {
        "accelerator": "gpu", "devices": "0", "precision": "bf16-mixed",
        "num_workers": 0, "find_unused_parameters": True,
    },
    "generation": {
        "remasking_strategy": "low_confidence", "sampling_steps": 32,
        "semi_ar": {"enabled": False, "block_size": 32, "steps_per_block": 4},
    },
    "validation": {
        "num_sanity_val_steps": 0, "val_check_interval": 1.0,
        "check_val_every_n_epoch": 1, "limit_val_batches": 1.0, "inference_batch_size": 8,
    },
    "logging": {
        "dir": "/tmp/molda_test_ckpt", "log_every_n_steps": 1, "save_on_n_steps": 500,
        "save_top_k_checkpoints": -1, "save_every_n_epochs": 1, "val_log_samples_per_gpu": 1,
        "log_stepwise_denoising": False, "stepwise_max_samples": 8,
        "log_nan_details": True, "nan_log_dir": "/tmp/molda_test_nan",
    },
    "wandb": {"enabled": False},
    "qformer": {
        "num_query_token": 32, "bert_name": "scibert", "bert_hidden_dim": 768,
        "cross_attention_freq": 2, "num_layers": 5,
    },
})


def fmt_tensor_stats(t, name=""):
    """텐서의 요약 통계."""
    t_f = t.float()
    return (f"{name}: shape={list(t.shape)}, dtype={t.dtype}, "
            f"min={t_f.min().item():.6f}, max={t_f.max().item():.6f}, "
            f"mean={t_f.mean().item():.6f}, std={t_f.std().item():.6f}")


def fmt_float(v, decimals=8):
    return f"{v:.{decimals}f}"


def main():
    lines = []
    L = lines.append  # shorthand

    L(f"# MolDA Training Step Diagnostic Report")
    L(f"")
    L(f"> Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L(f"> Device: CUDA {torch.cuda.get_device_name(0)}")
    L(f"> max_length=512, gen_max_len=256 (프로덕션 동일)")
    L(f"")

    # ─────────────────────────────────────────
    # 1. 모델 로딩
    # ─────────────────────────────────────────
    L("## 1. 모델 로딩")
    L("")

    from src.model.molda import MolDA
    model = MolDA(cfg)
    model = model.cuda()
    model.train()

    tokenizer = model.tokenizer
    total_vocab = len(tokenizer)
    new_tokens_count = total_vocab - ORIGINAL_VOCAB_SIZE

    L(f"- LLM: `{cfg.model.llm}`")
    L(f"- Original vocab size: **{ORIGINAL_VOCAB_SIZE:,}**")
    L(f"- Expanded vocab size: **{total_vocab:,}** (+{new_tokens_count:,} tokens)")
    L(f"- LoRA: r={cfg.lora.r}, alpha={cfg.lora.alpha}")
    L(f"- Precision: bfloat16")
    L("")

    # Trainable params 요약
    total_params = 0
    trainable_params = 0
    lora_count = 0
    embed_count = 0
    head_count = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if "lora_" in name:
                lora_count += param.numel()
            elif "embed" in name.lower() or "wte" in name.lower():
                embed_count += param.numel()
            elif "ff_out" in name.lower() or "lm_head" in name.lower() or "output" in name.lower():
                head_count += param.numel()

    L(f"### Parameter Summary")
    L(f"| 구분 | 파라미터 수 | 비율 |")
    L(f"|------|-----------|------|")
    L(f"| Total | {total_params:,} | 100% |")
    L(f"| Trainable | {trainable_params:,} | {trainable_params/total_params*100:.2f}% |")
    L(f"| LoRA | {lora_count:,} | {lora_count/total_params*100:.2f}% |")
    L(f"| Embedding (wte, tied to output) | {embed_count:,} | {embed_count/total_params*100:.2f}% |")
    L(f"| Head (ff_out, 0 if weight_tied) | {head_count:,} | {head_count/total_params*100:.2f}% |")
    L(f"| Frozen | {total_params - trainable_params:,} | {(total_params-trainable_params)/total_params*100:.2f}% |")
    L("")

    # ─────────────────────────────────────────
    # 2. 실제 데이터 로딩
    # ─────────────────────────────────────────
    L("## 2. 실제 데이터 로딩 (Train_toy100)")
    L("")

    from src.data.datamodule import MolDADataModule
    dm = MolDADataModule(tokenizer=tokenizer, cfg=cfg)
    dm.setup("fit")
    dl = dm.train_dataloader()
    batch = next(iter(dl))

    # GPU로 이동
    input_ids = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    tasks = batch["tasks"]
    prompt_lengths = batch["prompt_lengths"]

    B, L_seq = input_ids.shape
    L(f"- Batch size: **{B}**")
    L(f"- Sequence length: **{L_seq}** (max_length={cfg.data.max_length})")
    L(f"- Tasks in batch: `{tasks}`")
    L("")

    # Sample 0 상세
    s = 0
    plen = prompt_lengths[s].item()
    answer_len = (labels[s] != -100).sum().item()
    real_len = attention_mask[s].sum().item()
    pad_len = L_seq - real_len

    L(f"### Sample 0 상세")
    L(f"- Task: `{tasks[s]}`")
    L(f"- Prompt length: **{plen}** tokens")
    L(f"- Answer length: **{answer_len}** tokens")
    L(f"- Padding length: **{int(pad_len)}** tokens (EOS, id={128001})")
    L(f"- Total: {plen} (prompt) + {answer_len} (answer) + {int(pad_len)} (pad) = {L_seq}")
    L("")

    # 전체 토큰 테이블 (prompt + answer + padding 일부)
    answer_start = plen
    answer_end = plen + answer_len
    W = 20  # decoded 열 너비

    def _decode_one(tid):
        """토큰 1개를 사람이 읽을 수 있게 디코드."""
        txt = tokenizer.decode([tid])
        txt = txt.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
        if len(txt) > W - 2:
            txt = txt[:W - 3] + "…"
        return txt

    L(f"#### Prompt 전체 ({plen} tokens)")
    L(f"")
    L(f"```")
    L(f"{'Pos':>5} | {'Token ID':>8} | {'Label':>8} | {'Decoded':<{W}} | Region")
    L(f"{'─' * (5 + 3 + 8 + 3 + 8 + 3 + W + 3 + 10)}")
    for pos in range(plen):
        tid = input_ids[s, pos].item()
        lab = labels[s, pos].item()
        dec = _decode_one(tid)
        lab_str = "-100" if lab == -100 else str(lab)
        L(f"{pos:5d} | {tid:8d} | {lab_str:>8} | {dec:<{W}} | prompt")
    L(f"```")
    L("")

    L(f"#### Answer 전체 ({answer_len} tokens)")
    L(f"")
    L(f"```")
    L(f"{'Pos':>5} | {'Token ID':>8} | {'Label':>8} | {'Decoded':<{W}} | Region")
    L(f"{'─' * (5 + 3 + 8 + 3 + 8 + 3 + W + 3 + 10)}")
    for pos in range(answer_start, answer_end):
        tid = input_ids[s, pos].item()
        lab = labels[s, pos].item()
        dec = _decode_one(tid)
        match = "OK" if tid == lab else "MISMATCH!"
        L(f"{pos:5d} | {tid:8d} | {lab:8d} | {dec:<{W}} | answer  {match}")
    L(f"```")
    L("")

    # Padding 영역 (첫 5개만 표시)
    pad_start = answer_end
    pad_show = min(5, int(pad_len))
    if pad_show > 0:
        L(f"#### Padding (첫 {pad_show} / {int(pad_len)} tokens)")
        L(f"")
        L(f"```")
        L(f"{'Pos':>5} | {'Token ID':>8} | {'Label':>8} | {'Decoded':<{W}} | Region")
        L(f"{'─' * (5 + 3 + 8 + 3 + 8 + 3 + W + 3 + 10)}")
        for pos in range(pad_start, pad_start + pad_show):
            tid = input_ids[s, pos].item()
            lab = labels[s, pos].item()
            dec = _decode_one(tid)
            lab_str = "-100" if lab == -100 else str(lab)
            L(f"{pos:5d} | {tid:8d} | {lab_str:>8} | {dec:<{W}} | padding (EOS)")
        if pad_len > pad_show:
            L(f"  ... ({int(pad_len) - pad_show} more padding tokens, all id={128001})")
        L(f"```")
        L("")

    # ─────────────────────────────────────────
    # 3. Forward Process (make_noisy)
    # ─────────────────────────────────────────
    L("## 3. Forward Process — `make_noisy()`")
    L("")
    L("LLaDA Masked Diffusion: `t ~ U(0,1)` → `p_mask = (1-eps)*t + eps` → answer 토큰을 확률 p_mask로 MASK 교체")
    L("")

    torch.manual_seed(42)
    loss_fn = model.loss_fn
    noisy_ids, mask_indices, p_mask = loss_fn.make_noisy(input_ids, labels)

    L(f"### Masking 결과")
    L(f"| Sample | p_mask | Answer 길이 | Masked 수 | Mask 비율 |")
    L(f"|--------|--------|------------|----------|-----------|")
    for i in range(B):
        ans_len = (labels[i] != -100).sum().item()
        n_masked = mask_indices[i].sum().item()
        ratio = n_masked / max(ans_len, 1) * 100
        L(f"| {i} | {p_mask[i].item():.4f} | {ans_len} | {n_masked} | {ratio:.1f}% |")
    L("")

    # Sample 0의 마스킹 시각화
    L(f"### Sample 0 마스킹 시각화 (answer 영역, 첫 40 tokens)")
    L(f"")
    L(f"```")
    L(f"Position  : 원본 ID → Noisy ID  [MASK?]  Decoded")
    L(f"{'─' * 70}")
    vis_start = plen
    vis_end = min(plen + 40, plen + answer_len)
    for pos in range(vis_start, vis_end):
        orig = input_ids[s, pos].item()
        noisy = noisy_ids[s, pos].item()
        masked = mask_indices[s, pos].item()
        decoded_orig = tokenizer.decode([orig]).replace('\n', '\\n')
        marker = "██ MASK" if masked else "       "
        L(f"  [{pos:4d}] : {orig:6d} → {noisy:6d}  {marker}  '{decoded_orig}'")
    L(f"```")
    L("")

    # Prompt 영역 보존 확인
    prompt_preserved = torch.equal(noisy_ids[s, :plen], input_ids[s, :plen])
    L(f"- Prompt 영역 보존: **{'OK' if prompt_preserved else 'FAIL'}** (noisy_ids[:prompt_len] == input_ids[:prompt_len])")
    L(f"- MASK token ID: **{MASK_TOKEN_ID}** (`<|mdm_mask|>`)")
    L("")

    # ─────────────────────────────────────────
    # 4. Model Forward Pass
    # ─────────────────────────────────────────
    L("## 4. Model Forward Pass")
    L("")

    outputs = model.llada.model(input_ids=noisy_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, L, V]

    L(f"- Input: `noisy_ids` {list(noisy_ids.shape)}")
    L(f"- Output: `logits` {list(logits.shape)} (B, L, Vocab={logits.shape[-1]})")
    L("")

    # ─────────────────────────────────────────
    # 5. Loss 계산
    # ─────────────────────────────────────────
    L("## 5. Loss 계산 — `MaskedDiffusionLoss.forward()`")
    L("")
    L("공식: `loss = Σ [ CE(logit, target) / p_mask / answer_length ] / batch_size`")
    L("")

    loss_dict = loss_fn(
        logits=logits, input_ids=input_ids, labels=labels,
        mask_indices=mask_indices, p_mask=p_mask, tasks=tasks, global_step=0,
    )
    loss_val = loss_dict["loss"]
    ans_len_mean = loss_dict["answer_length_mean"]

    L(f"### 계산 결과")
    L(f"| 항목 | 값 |")
    L(f"|------|-----|")
    L(f"| **Loss** | **{loss_val.item():.6f}** |")
    L(f"| Answer length mean | {ans_len_mean:.2f} |")
    L(f"| Loss is finite | {'YES' if torch.isfinite(loss_val) else 'NO (NaN/Inf!)'} |")
    L(f"| Loss is positive | {'YES' if loss_val.item() > 0 else 'NO'} |")
    L("")

    # ── 전체 시퀀스 통합 테이블: Prediction + Loss 분해 ──
    s0_p_mask = p_mask[s].item()
    s0_ans_len = (labels[s] != -100).sum().item()
    TW = 16  # token 열 너비

    def _dec(tid):
        t = tokenizer.decode([tid]).replace('\n', '\\n').replace('\t', '\\t')
        return t[:TW-1] + "…" if len(t) > TW else t

    L(f"### Sample 0 — 전체 시퀀스 Prediction & Loss (p_mask={s0_p_mask:.4f}, ans_len={s0_ans_len})")
    L(f"")
    L(f"범례: `Region` = P(prompt), A(answer-보존), **M**(answer-MASKED), pad(패딩)")
    L(f"Masked 위치만 loss에 기여. 나머지는 `—`.")
    L(f"")
    L(f"```")
    hdr = (f"{'Pos':>5} | {'Region':>6} | {'정답ID':>7} | {'정답Token':<{TW}} | "
           f"{'예측ID':>7} | {'예측Token':<{TW}} | {'CE Loss':>11} | {'/p_mask':>11} | {'/ans_len':>11}")
    L(hdr)
    L(f"{'─' * len(hdr)}")

    real_len_s0 = int(attention_mask[s].sum().item())
    total_loss_manual = 0.0

    for pos in range(real_len_s0):
        tid = input_ids[s, pos].item()
        lab = labels[s, pos].item()
        is_masked = mask_indices[s, pos].item()

        # Region 태그
        if lab == -100:
            region = "P"
        elif is_masked:
            region = "**M**"
        else:
            region = "A"

        gt_tok = _dec(tid)

        if is_masked:
            # Prediction
            pred_logit = logits[s, pos]
            pred_id = pred_logit.argmax().item()
            pred_tok = _dec(pred_id)

            # CE Loss
            ce = F.cross_entropy(
                pred_logit.unsqueeze(0).float(),
                input_ids[s, pos].unsqueeze(0),
                reduction='none'
            ).item()
            weighted = ce / s0_p_mask
            normalized = weighted / s0_ans_len
            total_loss_manual += normalized

            L(f"{pos:5d} | {region:>6} | {tid:7d} | {gt_tok:<{TW}} | "
              f"{pred_id:7d} | {pred_tok:<{TW}} | {ce:11.4f} | {weighted:11.4f} | {normalized:11.6f}")
        elif lab != -100:
            # Answer but not masked — no loss, prediction still visible
            pred_logit = logits[s, pos]
            pred_id = pred_logit.argmax().item()
            pred_tok = _dec(pred_id)
            L(f"{pos:5d} | {region:>6} | {tid:7d} | {gt_tok:<{TW}} | "
              f"{pred_id:7d} | {pred_tok:<{TW}} | {'—':>11} | {'—':>11} | {'—':>11}")
        else:
            # Prompt — no loss
            L(f"{pos:5d} | {region:>6} | {tid:7d} | {gt_tok:<{TW}} | "
              f"{'—':>7} | {'—':<{TW}} | {'—':>11} | {'—':>11} | {'—':>11}")

    L(f"{'─' * len(hdr)}")
    L(f"{'':>5} | {'':>6} | {'':>7} | {'TOTAL':<{TW}} | "
      f"{'':>7} | {'':<{TW}} | {'':>11} | {'':>11} | {total_loss_manual:11.6f}")
    L(f"```")
    L("")
    L(f"- Padding ({int(pad_len)} tokens, all EOS id=128001) 생략")
    L(f"- **Sample 0 기여도 합계**: {total_loss_manual:.6f}")
    L(f"- **최종 loss** = (Σ all samples) / batch_size = {loss_val.item():.6f}")
    L(f"")
    L(f"> 학습 전이므로 예측 Token이 정답과 무관한 것이 정상. 학습이 진행되면 정답Token과 예측Token이 일치하기 시작.")
    L("")

    # ─────────────────────────────────────────
    # 6. Backward + Optimizer Step
    # ─────────────────────────────────────────
    L("## 6. Backward + Weight Update")
    L("")

    # Optimizer 구성
    lora_params_list = []
    embed_params_list = []
    head_params_list = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params_list.append((name, param))
        elif "embed" in name.lower() or "wte" in name.lower():
            embed_params_list.append((name, param))
        elif "lm_head" in name.lower() or "ff_out" in name.lower() or "output" in name.lower():
            head_params_list.append((name, param))

    param_groups = [
        {"params": [p for _, p in lora_params_list], "lr": cfg.lr.lora, "name": "lora"},
        {"params": [p for _, p in embed_params_list], "lr": cfg.lr.embed_orig, "name": "embed"},
    ]
    if head_params_list:
        param_groups.append(
            {"params": [p for _, p in head_params_list], "lr": cfg.lr.head_orig, "name": "head"}
        )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.training.weight_decay)

    # ── BEFORE snapshot ──
    # Embedding layer
    input_emb = model.llada.model.get_input_embeddings()
    # PEFT wrapping 때문에 실제 weight에 접근
    if hasattr(input_emb, 'modules_to_save'):
        emb_weight = list(input_emb.modules_to_save.values())[0].weight
    elif hasattr(input_emb, 'original_module'):
        emb_weight = input_emb.original_module.weight
    else:
        emb_weight = input_emb.weight

    emb_before_orig = emb_weight[:ORIGINAL_VOCAB_SIZE].detach().cpu().clone()
    emb_before_new = emb_weight[ORIGINAL_VOCAB_SIZE:].detach().cpu().clone()

    # Output head (weight_tying=True면 wte와 동일 텐서)
    output_emb = model.llada.model.get_output_embeddings()
    weight_tied = (output_emb is input_emb) or (
        hasattr(output_emb, 'original_module') and hasattr(input_emb, 'original_module')
        and output_emb.original_module is input_emb.original_module
    )
    if weight_tied:
        head_weight = emb_weight  # same tensor
    elif hasattr(output_emb, 'modules_to_save'):
        head_weight = list(output_emb.modules_to_save.values())[0].weight
    elif hasattr(output_emb, 'original_module'):
        head_weight = output_emb.original_module.weight
    else:
        head_weight = output_emb.weight

    head_before_orig = head_weight[:ORIGINAL_VOCAB_SIZE].detach().cpu().clone()
    head_before_new = head_weight[ORIGINAL_VOCAB_SIZE:].detach().cpu().clone()

    # LoRA: 첫 번째 lora_A, lora_B
    lora_a_name, lora_a_param = None, None
    lora_b_name, lora_b_param = None, None
    for name, param in lora_params_list:
        if "lora_A" in name and lora_a_param is None:
            lora_a_name, lora_a_param = name, param
        if "lora_B" in name and lora_b_param is None:
            lora_b_name, lora_b_param = name, param
        if lora_a_param is not None and lora_b_param is not None:
            break

    lora_a_before = lora_a_param.detach().cpu().clone() if lora_a_param is not None else None
    lora_b_before = lora_b_param.detach().cpu().clone() if lora_b_param is not None else None

    # ── Backward ──
    optimizer.zero_grad()
    loss_val.backward()

    # Gradient 통계
    L(f"### Gradient 통계 (backward 후)")
    L(f"")
    L(f"| Layer | Grad Norm | Grad Mean | Grad Max |")
    L(f"|-------|-----------|-----------|----------|")

    if emb_weight.grad is not None:
        g = emb_weight.grad.cpu().float()
        g_orig = g[:ORIGINAL_VOCAB_SIZE]
        g_new = g[ORIGINAL_VOCAB_SIZE:]
        L(f"| Embedding (orig vocab) | {g_orig.norm().item():.6e} | {g_orig.mean().item():.6e} | {g_orig.abs().max().item():.6e} |")
        L(f"| Embedding (new vocab) | {g_new.norm().item():.6e} | {g_new.mean().item():.6e} | {g_new.abs().max().item():.6e} |")
    else:
        L(f"| Embedding | grad=None | — | — |")

    if head_weight.grad is not None:
        g = head_weight.grad.cpu().float()
        g_orig = g[:ORIGINAL_VOCAB_SIZE]
        g_new = g[ORIGINAL_VOCAB_SIZE:]
        tied_tag = " (tied to wte)" if weight_tied else ""
        L(f"| Head{tied_tag} (orig vocab) | {g_orig.norm().item():.6e} | {g_orig.mean().item():.6e} | {g_orig.abs().max().item():.6e} |")
        L(f"| Head{tied_tag} (new vocab) | {g_new.norm().item():.6e} | {g_new.mean().item():.6e} | {g_new.abs().max().item():.6e} |")
    else:
        L(f"| Head | grad=None | — | — |")

    if lora_a_param is not None and lora_a_param.grad is not None:
        g = lora_a_param.grad.cpu().float()
        L(f"| LoRA_A (`{lora_a_name[-50:]}`) | {g.norm().item():.6e} | {g.mean().item():.6e} | {g.abs().max().item():.6e} |")
    if lora_b_param is not None and lora_b_param.grad is not None:
        g = lora_b_param.grad.cpu().float()
        L(f"| LoRA_B (`{lora_b_name[-50:]}`) | {g.norm().item():.6e} | {g.mean().item():.6e} | {g.abs().max().item():.6e} |")
    L("")

    # ── Optimizer step ──
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)
    optimizer.step()

    # ── AFTER snapshot ──
    emb_after_orig = emb_weight[:ORIGINAL_VOCAB_SIZE].detach().cpu().clone()
    emb_after_new = emb_weight[ORIGINAL_VOCAB_SIZE:].detach().cpu().clone()
    head_after_orig = head_weight[:ORIGINAL_VOCAB_SIZE].detach().cpu().clone()
    head_after_new = head_weight[ORIGINAL_VOCAB_SIZE:].detach().cpu().clone()
    lora_a_after = lora_a_param.detach().cpu().clone() if lora_a_param is not None else None
    lora_b_after = lora_b_param.detach().cpu().clone() if lora_b_param is not None else None

    L(f"### Weight 변화량 (optimizer.step() 후)")
    L(f"")
    L(f"```")
    L(f"{'Layer':<40} | {'Before Norm':>14} | {'After Norm':>14} | {'Delta Norm':>14} | {'Δ/Before':>10}")
    L(f"{'─' * 100}")

    def report_change(name, before, after):
        b_cpu = before.cpu().float()
        a_cpu = after.cpu().float()
        b_norm = b_cpu.norm().item()
        a_norm = a_cpu.norm().item()
        delta = (a_cpu - b_cpu).norm().item()
        ratio = delta / max(b_norm, 1e-12) * 100
        L(f"{name:<40} | {b_norm:14.6f} | {a_norm:14.6f} | {delta:14.8f} | {ratio:8.4f}%")

    report_change("Embedding (orig, idx < 126349)", emb_before_orig, emb_after_orig)
    report_change("Embedding (new,  idx >= 126349)", emb_before_new, emb_after_new)
    head_label = "Head/wte (tied)" if weight_tied else "Head/ff_out"
    report_change(f"{head_label} (orig, idx < 126349)", head_before_orig, head_after_orig)
    report_change(f"{head_label} (new,  idx >= 126349)", head_before_new, head_after_new)
    if lora_a_before is not None:
        report_change(f"LoRA_A (first layer)", lora_a_before, lora_a_after)
    if lora_b_before is not None:
        report_change(f"LoRA_B (first layer)", lora_b_before, lora_b_after)

    L(f"```")
    L("")

    # ─────────────────────────────────────────
    # 7. Embedding 상세 비교 (orig vs new vocab)
    # ─────────────────────────────────────────
    L("## 7. Embedding & Head 상세 — Original vs New Vocab")
    L("")
    L("> Original vocab (idx 0 ~ 126348): LLaDA 기본 토큰")
    L("> New vocab (idx 126349 ~): 프로젝트 추가 토큰 (BOOL, FLOAT, SELFIES, ...)")
    L("")

    # Embedding
    L(f"### Input Embedding (wte)")
    L(f"")
    L(f"```")
    L(f"{'구분':<25} | {'Mean':>12} | {'Std':>12} | {'Norm':>14} | {'Δ Norm':>14}")
    L(f"{'─' * 85}")
    for tag, tensor, baseline in [
        ("Orig (before)", emb_before_orig, None),
        ("Orig (after)",  emb_after_orig,  emb_before_orig),
        ("New  (before)", emb_before_new,  None),
        ("New  (after)",  emb_after_new,   emb_before_new),
    ]:
        t = tensor.float()
        delta_str = "—"
        if baseline is not None:
            delta_str = fmt_float((t - baseline.float()).norm().item(), 8)
        L(f"{tag:<25} | {t.mean().item():12.6e} | {t.std().item():12.6e} | {t.norm().item():14.6f} | {delta_str}")
    L(f"```")
    L("")

    # 특정 토큰의 embedding 변화
    L(f"### 특정 토큰별 Embedding 변화")
    L(f"")
    sample_tokens = {
        "orig — 'the'": tokenizer.encode("the", add_special_tokens=False)[0],
        "orig — 'molecule'": tokenizer.encode("molecule", add_special_tokens=False)[0],
        "new  — '<BOOLEAN>'": tokenizer.convert_tokens_to_ids("<BOOLEAN>"),
        "new  — '<SELFIES>'": tokenizer.convert_tokens_to_ids("<SELFIES>"),
        "new  — '<FLOAT>'": tokenizer.convert_tokens_to_ids("<FLOAT>"),
        "new  — '<mol>'": tokenizer.convert_tokens_to_ids("<mol>"),
    }

    L(f"```")
    L(f"{'Token':<25} | {'ID':>7} | {'Vocab':>6} | {'Emb Δ Norm':>14} | {'Head Δ Norm':>14}")
    L(f"{'─' * 75}")
    for desc, tid in sample_tokens.items():
        is_new = tid >= ORIGINAL_VOCAB_SIZE
        emb_b = emb_before_new[tid - ORIGINAL_VOCAB_SIZE] if is_new else emb_before_orig[tid]
        emb_a = emb_after_new[tid - ORIGINAL_VOCAB_SIZE] if is_new else emb_after_orig[tid]
        head_b = head_before_new[tid - ORIGINAL_VOCAB_SIZE] if is_new else head_before_orig[tid]
        head_a = head_after_new[tid - ORIGINAL_VOCAB_SIZE] if is_new else head_after_orig[tid]
        emb_delta = (emb_a.float() - emb_b.float()).norm().item()
        head_delta = (head_a.float() - head_b.float()).norm().item()
        vocab_tag = "new" if is_new else "orig"
        L(f"{desc:<25} | {tid:7d} | {vocab_tag:>6} | {emb_delta:14.8f} | {head_delta:14.8f}")
    L(f"```")
    L("")

    # ─────────────────────────────────────────
    # 8. LoRA Weight 상세
    # ─────────────────────────────────────────
    L("## 8. LoRA Weight 변화")
    L("")

    L(f"```")
    L(f"{'Layer (last 60 chars)':<62} | {'Grad Norm':>12} | {'Δ Weight Norm':>14}")
    L(f"{'─' * 95}")
    lora_changes = []
    for name, param in lora_params_list:  # 전체 LoRA layer
        if param.grad is not None:
            g_norm = param.grad.float().norm().item()
        else:
            g_norm = 0.0
        # Weight change (need before snapshot — 첫 layer만 있음)
        short_name = name[-60:]
        lora_changes.append((short_name, g_norm))
        L(f"{short_name:<62} | {g_norm:12.6e} | —")
    L(f"```")
    L("")

    if lora_a_before is not None and lora_a_after is not None:
        L(f"### LoRA_A (첫 번째 layer) 변화 상세")
        L(f"- Layer: `{lora_a_name}`")
        L(f"- Shape: {list(lora_a_param.shape)}")
        delta_a = (lora_a_after.float() - lora_a_before.float())
        L(f"- Before norm: {lora_a_before.float().norm().item():.6f}")
        L(f"- After norm:  {lora_a_after.float().norm().item():.6f}")
        L(f"- Delta norm:  {delta_a.norm().item():.8f}")
        L(f"- Delta max:   {delta_a.abs().max().item():.8f}")
        L("")

    if lora_b_before is not None and lora_b_after is not None:
        L(f"### LoRA_B (첫 번째 layer) 변화 상세")
        L(f"- Layer: `{lora_b_name}`")
        L(f"- Shape: {list(lora_b_param.shape)}")
        delta_b = (lora_b_after.float() - lora_b_before.float())
        L(f"- Before norm: {lora_b_before.float().norm().item():.6f}")
        L(f"- After norm:  {lora_b_after.float().norm().item():.6f}")
        L(f"- Delta norm:  {delta_b.norm().item():.8f}")
        L(f"- Delta max:   {delta_b.abs().max().item():.8f}")
        L(f"- LoRA_B는 초기값이 0 → 첫 step에서 0이 아닌 값으로 변화 (정상)")
        L("")

    # ─────────────────────────────────────────
    # 9. GPU 메모리
    # ─────────────────────────────────────────
    L("## 9. GPU 메모리 사용량")
    L("")
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    L(f"| 항목 | GB |")
    L(f"|------|-----|")
    L(f"| Allocated | {mem_alloc:.2f} |")
    L(f"| Reserved | {mem_reserved:.2f} |")
    L(f"| Total GPU | {mem_total:.2f} |")
    L(f"| Free | {mem_total - mem_reserved:.2f} |")
    L("")

    # ─────────────────────────────────────────
    # Write
    # ─────────────────────────────────────────
    output_path = os.path.join(PROJECT_ROOT, "test", "diagnostic_report.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{'=' * 60}")
    print(f"Diagnostic report saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
