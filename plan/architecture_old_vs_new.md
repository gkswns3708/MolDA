# Old_MolDA vs New_MolDA — Architecture 대조

> 작성일: 2026-05-12
> Old: `/Public/11-MolDA/MolDA_miccai_Tpod/` (MICCAI 2026 Tpod, Mol-LLM Custom)
> New: `/opt/MolDA/` (현재 작업 트리, `feature/stage3_v_molpo` branch)

Old의 monolithic Blip2-stage3 구조를 New가 **Hydra config + per-stage freeze policy + V-MolPO infrastructure** 로 재구성한 변경을 정리. 코드 참고/리뷰/논문 architecture 섹션용.

---

## 1. 한눈에 보는 매핑 표

| Old (MolDA_miccai_Tpod) | New (opt/MolDA) | 관계 |
|---|---|---|
| `model/blip2_opt.py` (1400줄, base class) | `src/model/molda.py` + `src/model/llada_wrapper.py` 로 분리 | **분해 재설계** |
| `model/blip2_llada.py` (2772줄) | `src/model/molda.py` (~430줄) + `src/model/llada_wrapper.py` + `src/generation/generate.py` | **분해 재설계** — generation 분리, 단일 forward 책임 |
| `model/blip2_stage3.py` (2164줄, Lightning 통합) | `src/training/trainer.py` (mixin) + `src/training/{validation,optimizer,scheduler,checkpoint}.py` | **분해 재설계** — Mixin pattern 으로 책임 분리 |
| `model/blip2_qformer.py` | `src/model/qformer.py` (얇은 wrapper) + `src/model/adapter/blip2.py` (vendored Blip2Base) | **재포장** — `init_Qformer` classmethod 만 재사용 |
| `model/gin_model.py`, `model/tokenGT.py`, `model/gine_tokengt.py` | `src/model/adapter/{gin_model,tokenGT,gine_tokengt}.py` (vendored 그대로) + `src/model/gnn.py` (어댑터) | **그대로 vendoring + 얇은 wrapper** |
| `model/gine_loader.py` | `src/model/adapter/gine_loader.py` (vendored, **현재 미사용**) | **vendoring 후 dead** — gine_tokengt.py 내부 loader 통합 |
| `data_utils.py` (635줄, DataCollator + MolPO 로직 혼재) | `src/data/{datamodule,dataset,collator,molpo_collator}.py` (4 파일 분리) | **분해 재설계** — SFT collator / MolPO collator 분리, dataset/datamodule 분리 |
| `data_module.py` | `src/data/datamodule.py` | **재구성** — Hydra-friendly path resolution + stage-aware collator dispatch |
| `stage3.py` (900+줄, main script + Lightning Module + Trainer config) | `scripts/train.py` (~200줄, Hydra entry) + `src/training/trainer.py` | **분해 재설계** — main + Lightning 분리 |
| `configs/` (Hydra default.yaml + train_llada.yaml) | `src/configs/{default,data,gnn,trainer,experiment,download}/` (계층 구조) | **재구성** — defaults list / experiment override 패턴 |
| (Old 의 MolPO: `{epoch}-th_rejected_*` curriculum, batch 내 duplication) | `src/data/molpo_collator.py` (B → 2B/3B 배치) + `src/training/v_molpo_loss.py` + `src/training/vrpo_elbo.py` + `src/model/ref_llada_wrapper.py` | **완전 재설계** — VRPO 기반 새 아키텍처 |
| (Old 의 단일 freeze 정책) | `MolDA._apply_stage_freeze_policy()` (stage 1/2/3 분기) | **신규 도입** |
| (Old 의 단일 forward) | `MolDA.forward` → `_forward_with_graph` / `_molpo_forward` / `_forward_logits` dispatch | **신규 도입** |
| (Old 에 없음) | `src/training/vrpo_elbo.py` (n_t-MC + antithetic) | **신규 (V-MolPO)** |
| (Old 에 없음) | `src/training/v_molpo_loss.py` (DPO-E + per-task EMA anchor) | **신규 (V-MolPO)** |
| (Old 에 없음) | `src/model/ref_llada_wrapper.py` (frozen π_ref) | **신규 (V-MolPO)** |
| Old `validation_step` (Blip2Stage3 내) | `src/training/validation.py` (ValidationMixin) | **분리 + DDP-safe JSONL** |
| Old generation (blip2_llada.py 의 `generate` / `generate_semi_ar`) | `src/generation/generate.py` | **분리** |

---

## 2. 모델 forward 흐름 비교

### 2.1 Old: `Blip2LLaDA.forward(samples)` (단일 monolithic)

```
samples = { input_ids, attention_mask, labels, graphs, additional_graphs, is_mol_token }
       │
       ├─ Stochastic masking on answer region (is_answer = labels != -100)
       │    sample p_mask ~ U(0,1) per batch
       │    mask positions with prob p_mask, importance-weight 1/p_mask
       │
       ├─ Tokenizer noise:  input_ids → noisy_input_ids → text_embeds
       │
       ├─ if "graph" in mol_representation:
       │     graph_encoder(graphs) → graph_emb, graph_mask
       │     ln_graph(graph_emb)
       │     Qformer.bert(query_tokens, graph_emb) → q_out
       │     opt_proj(q_out) → mol_tokens
       │     inject_graph_embeds2input_embeds(text_embeds, mol_tokens, is_mol_token)
       │
       ├─ llm_model(inputs_embeds=text_embeds, attention_mask) → logits
       │
       └─ CE loss on masked positions, weighted by 1/p_mask, normalized
              → scalar loss
```

**한계**: stage 분리 없음, MolPO 분기는 collator + label로 처리 (forward 자체는 SFT만), Q-Former/GNN/LLM이 한 함수에 묶임.

### 2.2 New: `MolDA.forward(batch)` (dispatch + per-stage 분리)

```
batch = { input_ids, labels, attention_mask, prompt_lengths, tasks,
          graphs?, additional_graphs?, molpo_batch_size?, ... }
       │
       ├─ V-MolPO 분기: molpo_enabled && "molpo_batch_size" in batch
       │     → _molpo_forward(batch)
       │         [§5 신규 도입의 상세 흐름 참고]
       │
       └─ SFT 분기 (Stage 1/2):
            loss_fn.make_noisy(input_ids, labels) → noisy_ids, mask_indices, p_mask
            │
            ├─ use_graph = (stage>=2 && mol_representation=="string+graph" && "graphs" in batch)
            │     ├─ True: _forward_with_graph(noisy_ids, attention_mask, batch, input_ids)
            │     │           │ text_embeds = llada.model.get_input_embeddings()(noisy_ids)
            │     │           │ with torch.no_grad() if not tune_gnn:
            │     │           │     gnn(graphs.x, edge_index, edge_attr, batch.batch) → graph_emb, graph_mask
            │     │           │ qformer(graph_emb, graph_mask) → mol_tokens [B, n_query, D]
            │     │           │ inject at <mol> positions in text_embeds
            │     │           └─ llada.model(inputs_embeds=text_embeds, attention_mask) → outputs
            │     │
            │     └─ False:  llada.model(input_ids=noisy_ids, attention_mask) → outputs
            │
            ├─ loss_fn(logits, input_ids, labels, mask_indices, p_mask, tasks, ...)
            │     → loss + per-task metrics + NaN log
            │
            └─ return { loss, answer_length_mean, ... }
```

**차이점**:
- `_apply_stage_freeze_policy()` 가 init 시 한 번 호출되어 `requires_grad` 정책 고정 — forward 마다 분기 안 함
- `make_noisy` 가 `t_override` / `mask_indices_override` 받아서 외부 seed 주입 가능 (VRPO antithetic 호환)
- `_forward_logits` 가 ELBO 호출용 logits-only 서브루틴으로 분리 (loss 안 계산)

---

## 3. 데이터 파이프라인 비교

### 3.1 Old: `data_utils.py` 단일 책임

```
dataset (HF Arrow) → Stage3DM (custom DataModule)
      │
      └─ DataCollator(args).__call__(batch)
              │
              ├─ Tokenize prompts + targets (mol_representation 분기)
              ├─ Build is_mol_token mask
              ├─ if "graph": GraphCollater([], [])(list_graphs) → graphs Batch
              ├─ if MolPO enabled:
              │     for k in range(1, batch_division):
              │       duplicate prompt/target with `{current_epoch}-th_rejected_*` fields
              │       optional preference system_prompt prepend
              │       molpo_labels 별도 masking
              └─ return { input_ids, attention_mask, labels, graphs, is_mol_token, molpo_labels, tasks }
```

**한계**: MolPO 로직이 SFT collator에 섞여있음, dataset 추상화 부재 (collator 가 raw HF row 직접 처리), `{epoch}-th_rejected_*` 동적 컬럼 접근으로 brittle.

### 3.2 New: 4-component 분리

```
dataset/Processed/{root}/{Train,Val,Test}/  (HF Arrow)
      │
      ▼
MolDADataModule (datamodule.py)
      │ - _resolve_path(split): "dataset/Processed/{root}/{split}" + legacy fallback
      │ - dual tokenizers (right-pad train, left-pad eval)
      │
      ▼
MoleculeDataset (dataset.py)
      │ - dual-column 자동 detect: prompt_text_{selfies,smiles} → prompt_text 매핑
      │ - has_molpo_pair property: target_text_chosen/rejected 존재 여부
      │ - V-MolPO chosen/rejected dual-column 까지 처리
      │ - _val_idx 부여 (DDP dedup용)
      │
      ▼
Train collator selection (_build_train_collator):
   ├─ molpo.enabled=true → MolPOTrainCollator (molpo_collator.py)
   │     - B chosen/rejected pair → 2B (or 3B) batch
   │     - molpo_batch_size, molpo_batch_division 키 emit
   │     - answer_len 0 pre-filter
   │     - graph: same prompt 의 chosen/rejected 에 동일 graph replicate
   │
   └─ default → TrainCollator (collator.py)
         - right-pad input_ids/labels with EOS
         - _build_graph_batch helper (PyG Collater([], []))
         - if mol_representation includes "graph": emit graphs + additional_graphs

EvalCollator (collator.py)
   - left-pad PAD-only prompt_ids for generation
```

**개선**:
- MolPO 로직이 별도 collator 로 isolation (lazy import 로 stage 1/2 영향 차단)
- dataset 의 dual-column 처리가 명시적 (Old 의 raw 컬럼 접근 제거)
- graph batching helper (`_build_graph_batch`) 공통화 — train/eval 모두 같은 함수 사용

---

## 4. Stage 분리 정책 비교

### 4.1 Old: 단일 stage = "Stage 3 통합 SFT"
- `model/blip2_stage3.py` 가 모든 모듈 (Q-Former + GNN + LoRA + LLM) 한 번에 학습
- pretrained_ckpt_path 로 prior stage 가중치 로드 가능하나, **stage 내부 분기 없음**
- freeze 정책은 args 플래그 (`tune_gnn`, `tune_llm`)로 외부 제어

### 4.2 New: 3-stage 명시 분리

| | Stage 1 | Stage 2 | Stage 3 (V-MolPO) |
|---|---|---|---|
| `cfg.stage` | 1 | 2 | 3 |
| `mol_representation` | `string_only` | `string+graph` | `string+graph` |
| 학습 대상 | LoRA + wte (+ lm_head) | Q-Former + query_tokens + opt_proj + ln_graph | LoRA + wte + (GNN/QFormer if `tune_gnn=true`) |
| Frozen | base LLM | LLM, LoRA, GNN | base LLM, ref_model |
| MolPO | OFF | OFF | ON |
| 진입점 | `MolDA.forward()` SFT branch | 동 + `_forward_with_graph` | `MolDA.forward()` V-MolPO branch → `_molpo_forward` |
| ckpt 전환 | Stage 0 (HF) → Stage 1 ckpt | Stage 1 ckpt → Stage 2 ckpt | Stage 2 ckpt → Stage 3 ckpt |

**구현 메커니즘**: `MolDA._apply_stage_freeze_policy()` 가 `__init__` 마지막에 호출되어 stage 별 freeze 정책 일괄 적용. 학습 hot loop 에선 `requires_grad` 만 검사하면 됨.

```python
# 단순화한 흐름
if stage == 1:        # LoRA + wte 만
    requires_grad = "lora" in name or "modules_to_save" in name
elif stage == 2:      # Q-Former bridge 만
    STAGE2_TRAINABLE_KEYS = ("qformer", "query_tokens", "opt_proj", "ln_graph")
    requires_grad = any(k in name.lower() for k in STAGE2_TRAINABLE_KEYS)
elif stage == 3:      # LoRA + wte + (옵션) GNN/QFormer; ref_model 항상 frozen
    if name.startswith("ref_model."):
        requires_grad = False
    elif name.startswith("gnn.") or name.startswith("qformer."):
        requires_grad = tune_gnn   # cfg.model.tune_gnn
    elif "lora_" in name or "modules_to_save" in name:
        requires_grad = True
    else:
        requires_grad = False      # base LLM frozen
```

---

## 5. 신규 도입 — V-MolPO Infrastructure

Old 의 MICCAI MolPO 는 collator 에서 chosen/rejected batch duplication + label masking 으로만 구현됨 (preference reward 학습은 SFT loss 의 token-level CE 차이로 간접). New 는 **DPO-E + VRPO** 정식 도입:

### 5.1 신규 모듈
| 파일 | 역할 |
|---|---|
| `src/training/vrpo_elbo.py` | `sample_shared_TM(eps=1e-3)` — n_t × B (T, M) deterministic 샘플. `compute_elbo(forward_fn, ids, lab, n_t, seed)` — n_t-MC ELBO. `compute_dpo_e_score(...)` — DPO-E margin. |
| `src/training/v_molpo_loss.py` | `TaskAnchorEMA(alpha=0.99)` — per-task EMA, state_dict 직렬화 가능. `compute_v_molpo_loss(elbo_θ/ref × w/l, tasks, ema, β, λ, margin_clip*)` — sigmoid DPO-E + 옵션 anchor + margin clip (burn-in gated). `combine_total_loss(L_SFT, V-MolPO out)`. |
| `src/model/ref_llada_wrapper.py` | `RefMolDA` — Stage 2 ckpt 별도 인스턴스 load + freeze + `.train()` override. inner cfg 에서 `molpo.enabled=False` 로 RefMolDA 재귀 방지. `make_forward_fn(sub_batch)` — `compute_elbo` 가 받을 forward 함수 factory. |
| `src/data/molpo_collator.py` | `MolPOTrainCollator` — chosen/rejected pair → 2B (or 3B) batch. `molpo_batch_size/_division` 키 emit (MolDA forward 분기 신호). graph chosen/rejected 동일 graph replicate. |

### 5.2 `_molpo_forward` 흐름

```
batch (from MolPOTrainCollator) — input_ids[2B,L], labels[2B,L], tasks[2B],
                                  graphs (replicated), molpo_batch_size=B, _division=2
       │
       ├─ Slice: [0:B]=chosen, [B:2B]=rejected (div=3 면 [0:B]=sft, ...)
       │
       ├─ Antithetic seed:  seed_w = global_step*1000+7, seed_l = global_step*1000+13
       │                    (antithetic ON 이면 ref 가 같은 seed 사용)
       │
       ├─ ELBO 계산:
       │    ┌─ compute_elbo(make_theta_fwd(chosen_sub), ..., seed=seed_w)    → elbo_θ_w  [B] (grad)
       │    ├─ compute_elbo(make_theta_fwd(rejected_sub), ..., seed=seed_l)  → elbo_θ_l  [B] (grad)
       │    └─ with torch.no_grad():
       │         compute_elbo(ref_model.make_forward_fn(chosen_sub), ..., seed=seed_w_ref)    → elbo_ref_w
       │         compute_elbo(ref_model.make_forward_fn(rejected_sub), ..., seed=seed_l_ref) → elbo_ref_l
       │
       ├─ compute_v_molpo_loss(elbo_θ/ref × w/l, tasks_chosen, task_anchor_ema, β=0.1, ...)
       │     - rewards = β·(B̂_θ − B̂_ref)
       │     - margin = r_θ(y_w) − r_θ(y_l)
       │     - EMA update: anchor_per_task[task] += (1-α)·(r_w.detach() − anchor)
       │     - γ_i = λ · |EMA[task_i]|
       │     - L_pref = −logσ(margin − γ_i).mean()
       │     - (옵션) L_anchor = −logσ(−(r_l − λ_r·EMA)).mean()
       │     - (옵션) margin_clip if global_step < burn_in
       │
       ├─ (div=3) L_SFT = _sft_forward_internal(sft_sub) — standard masked diffusion loss
       │
       ├─ combine_total_loss(L_SFT, vmolpo_out, sft_weight, molpo_weight, anc_w) → L_total
       │
       └─ return {
              "loss": L_total,
              "v_molpo/loss_pref", "v_molpo/loss_anchor",
              "v_molpo/margin", "v_molpo/margin_unclipped",
              "v_molpo/rewards_chosen", "v_molpo/rewards_rejected",
              "v_molpo/gamma", "v_molpo/avg_chosen_reward",
              "v_molpo/elbo_θ/ref_w/l_mean",
              "v_molpo/margin_clipped_frac",
              ...
          }
       (trainer.training_step 이 "v_molpo/*" prefix 키를 자동으로 wandb logging)
```

### 5.3 Checkpoint 처리
- `on_save_checkpoint`: `model.ref_model.*` 모든 키 drop (어차피 ref_ckpt_path 에서 재로드), `task_anchor_ema.state_dict()` 를 `v_molpo_task_anchor_ema` 별도 키로 저장
- `on_load_checkpoint`: EMA 복원 (epoch boundary 보존, reset 안 함)

---

## 6. 제거 / 보류 / 단순화

| Old 항목 | New 처리 | 사유 |
|---|---|---|
| `model/blip2_qformer.py` 의 Blip2Qformer (ITM/ITC head 포함 contrastive variant) | `src/model/adapter/blip2qformer.py` 로 vendored, **dead** | New 는 forward에서 `Qformer.bert` 만 사용. ITM/ITC 는 contrastive pretraining 단계에서 부활 가능 (plan §8(d)) |
| `model/blip2_{llama,mistral,opt,t5}.py` (LLM 별 변종) | **삭제** (vendored 도 안 함) | LLaDA 단일 backbone 으로 집중. 다른 LLM 으로 옮기는 시점에 어댑터 재작성 |
| `model/help_funcs.py` 의 metric 함수들 | `src/training/metrics.py` 로 재작성 | OLD 의 BLEU/METEOR/ROUGE wrapper 를 task-categorized evaluator 로 재설계 |
| `model/added_tokens.py` | `src/model/adapter/added_tokens.py` (vendored, 사용 중) | 토큰 상수 정의 그대로 |
| Old 의 `{epoch}-th_rejected_*` curriculum (epoch 따라 reject 강도 변경) | **제거** | New 는 build_molpo_dataset_synthetic.py 로 pre-pair 데이터 생성 후 정적 chosen/rejected 사용. curriculum 은 별도 ablation 으로 도입 가능 |
| Old 의 `apply_preference_system_prompt` (rejected 에 system prompt prepend) | **보류** | New V-MolPO 는 ELBO 기반 reference-relative reward 라 system prompt 차이로 margin 만드는 우회 불필요 |
| Old generation: `generate_semi_ar` (teacher forcing 변종) | `src/generation/generate.py` 에 일부 보존 | semi_ar 옵션 cfg 로 노출 (`cfg.generation.semi_ar`) |
| Old 의 `model.train()` 호출 시 frozen GNN 도 train 모드로 전환되는 문제 | New `GINETokenGT.train()` override (class-level 메서드) 로 해결 | DDP spawn pickle 안전 |
| Old 의 단일 `lr` flat config | New `lr.{lora, embed_orig, embed_new, head_orig, head_new, other}` 5-group | per-component LR 미세조정 |

---

## 7. 디렉터리 구조 비교

### 7.1 Old
```
MolDA_miccai_Tpod/
├── model/                    # 모든 모델/loss/scheduler 가 여기
│   ├── blip2_opt.py          # base class
│   ├── blip2_llada.py        # LLaDA forward + generation (2772줄)
│   ├── blip2_stage3.py       # Lightning module (2164줄)
│   ├── blip2_{llama,...}.py  # LLM 변종
│   ├── gin_model.py / tokenGT.py / gine_tokengt.py   # GNN
│   ├── help_funcs.py         # 평가 metric
│   └── scheduler.py
├── data_module.py / data_utils.py   # 데이터 (2 files, mono)
├── stage3.py                  # main + Hydra
├── configs/                   # Hydra (flat)
├── dataset/, utils/, log/, debug_*.py
└── augment_dataset.py 등 dataset 생성 스크립트
```

### 7.2 New
```
opt/MolDA/
├── src/
│   ├── model/
│   │   ├── molda.py             # unified model (~430줄)
│   │   ├── llada_wrapper.py     # LLM + tokenizer + LoRA
│   │   ├── qformer.py           # Q-Former bridge wrapper
│   │   ├── gnn.py               # GNN wrapper
│   │   ├── ref_llada_wrapper.py # 신규 (V-MolPO)
│   │   ├── added_tokens.py
│   │   └── adapter/             # vendored Old code (read-only)
│   │       ├── blip2.py, blip2qformer.py
│   │       ├── gin_model.py, tokenGT.py, gine_tokengt.py
│   │       └── gine_loader.py (dead), help_funcs.py (dead)
│   ├── training/
│   │   ├── trainer.py           # MolDATrainer + mixin
│   │   ├── optimizer.py         # 5-group config_optimizers
│   │   ├── scheduler.py         # WarmupStableDecayLR
│   │   ├── loss.py              # MaskedDiffusionLoss
│   │   ├── v_molpo_loss.py      # 신규 (V-MolPO)
│   │   ├── vrpo_elbo.py         # 신규 (V-MolPO)
│   │   ├── validation.py        # ValidationMixin + DDP-safe JSONL
│   │   ├── checkpoint.py        # CheckpointMixin + EMA 보존
│   │   └── metrics.py
│   ├── data/
│   │   ├── datamodule.py
│   │   ├── dataset.py
│   │   ├── collator.py          # SFT collator + graph batch helper
│   │   └── molpo_collator.py    # 신규 (V-MolPO)
│   ├── generation/
│   │   └── generate.py          # 분리
│   ├── loggers/
│   ├── dataset_generation/      # dataset 생성 파이프라인
│   └── configs/                 # Hydra 계층 구조
│       ├── default.yaml
│       ├── data/                # 데이터셋별
│       ├── gnn/                 # GNN 별
│       ├── trainer/             # stage 별
│       └── experiment/          # 실험 별 (overrides)
├── scripts/                     # 실행 스크립트
│   ├── train.py
│   ├── train_stage3_v_molpo_*.sh
│   └── build_molpo_dataset_synthetic.py 등
├── test/                        # 28+ test files
├── checkpoint/                  # 학습 산출물 (.gitignore)
├── plan/                        # 설계 문서
└── venv/                        # 가상환경 (.gitignore)
```

---

## 8. 핵심 메시지

- **단일 책임 분해**: Old 의 `blip2_stage3.py` (2164줄) → New 의 mixin 4개 (`trainer.py`, `optimizer.py`, `scheduler.py`, `validation.py`, `checkpoint.py`). 각 mixin 이 자체 테스트 + 책임 명확.
- **Stage 명시화**: Old 는 stage 가 외부 데이터 / ckpt path 차이로만 구분. New 는 `cfg.stage` 가 first-class field 로 init / freeze / forward dispatch 에 직접 영향.
- **V-MolPO 정식 도입**: Old MICCAI MolPO 의 batch duplication 만의 한계 → DPO-E (reference-based) + VRPO (n_t-MC + antithetic, LLaDA-1.5) + per-task EMA anchor (MolPO 의 원래 의도) 결합한 새 정식 학습 알고리즘.
- **Hydra config 계층화**: Old `default.yaml + train_llada.yaml` flat → New `default + data/X + gnn/X + trainer/X + experiment/Y` 의 5개 축. `+experiment=stage3_v_molpo_chebi_only trainer=stage3` 처럼 한 줄 launch.
- **Old 의 vendored 유지 사유**: GINE+TokenGT, BLIP-2 Q-Former 의 구현이 작지 않고 패치 안 됨. 재구현하면 reproducibility 깨질 위험 → vendored adapter 로 격리 + 어댑터 wrapper 로 New schema 와 분리.
- **Vendored 미사용 부분 (blip2qformer.py, gine_loader.py, help_funcs.py)**: 단순 삭제 안 함 — contrastive pretraining / dynamic GNN backbone dispatch 같은 미래 작업에서 reference 로 살아남.

---

## 부록: 모듈별 줄 수 비교 (참고)

| 모듈 | Old | New |
|---|---|---|
| Core model | `blip2_llada.py` 2772 | `molda.py` ~430 + `llada_wrapper.py` ~200 |
| Lightning trainer | `blip2_stage3.py` 2164 | `trainer.py` + 4 mixin ≈ 1500 |
| Data | `data_utils.py` 635 + `data_module.py` ~150 | `datamodule.py` ~200 + `dataset.py` ~100 + `collator.py` ~200 + `molpo_collator.py` ~210 |
| V-MolPO | (없음) | `vrpo_elbo.py` 162 + `v_molpo_loss.py` 211 + `ref_llada_wrapper.py` 158 |
| Q-Former | `blip2_qformer.py` ~550 | `qformer.py` ~60 (+ vendored 550) |
| GNN | `gine_tokengt.py` ~73 + `gin_model.py`/`tokenGT.py` ~1100 | `gnn.py` ~70 (+ vendored 동일) |
| Configs | `default.yaml`/`train_llada.yaml` ~300 | 30+ yaml 파일 (계층) |
| Tests | 일부 debug 스크립트만 | 28+ test files |

New 는 line count 가 분산되어 있지만 각 파일 책임이 더 명확.
