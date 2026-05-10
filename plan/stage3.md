# Stage 3 V-MolPO 구현 plan

본 문서는 New_MolDA Stage 3 의 **V-MolPO (Variance-Reduced MolPO)** 도입을 위한 설계와 실행 계획. Phase 0 분산 측정 결과 (2026-05-10) 가 반영되어 n_t / antithetic / epoch 등 핵심 hyperparameter 가 정량 근거 위에 결정됨.

---

## 0. 한 줄 요약

V-MolPO = **MolPO 의 task-anchor 안정화** + **VRPO 의 reference-based ELBO + n_t-MC + antithetic** 결합. MICCAI MolPO 가 LLaDA 에서 폭주했던 분산 문제를 LLaDA 1.5 (arXiv:2505.19223) 의 VRPO 로 직접 해결. New_MolDA 에 **n_t=2, antithetic ON, ref=Stage 2 ckpt, 3 epoch** 로 구현.

---

## 1. 왜 V-MolPO 가 필요한가

### 1.1 Diffusion LM 의 log π(y|x) 본질 차이

```text
Causal LM (Mistral)                    Diffusion LM (LLaDA)
═══════════════════                    ════════════════════
log π(y|x) = Σ log p(y_t | y_<t, x)    log π(y|x) ≥ E_t E_q [ℓ(y_t, t)]  ← ELBO
            └─ deterministic ─┘                  └── stochastic ──┘

  Single fwd → variance = 0            Single fwd, 1 sample of t
                                       → variance = Θ(1)
                                       → 1/p_mask 가 작은 t 에서 폭증
```

DPO 는 "결정론적 log π" 를 가정하는데 LLaDA 는 그 가정을 깨버림. 단일 MC 샘플로 ELBO 를 추정하면 V[ŝ] = Θ(1) 이고, 그 결과 −logσ(β·s) 가 출렁이며 grad 가 폭주.

### 1.2 MICCAI MolPO 가 LLaDA 에서 broken 인 이유

```text
y_w ──→ [LLaDA fwd 1×] ──→ instance_loss_w ──→ logp_w
                              (rescale broken)        ↘
y_l ──→ [LLaDA fwd 1×] ──→ instance_loss_l ──→ logp_l ──→ margin
                                                            │
                                                loss = −logσ(β·margin)

Failure modes:
  ❌ no π_ref           → DPO theory 미충족
  ❌ single MC sample   → margin variance Θ(1) 그대로
  ❌ rescale 깨짐        → instance_loss × seq_len / token_count
                         (instance_loss 는 이미 answer_len 기준 평균)
  ❌ 1/p_mask amplifies → t≈ε 에서 1000× 까지 튐
```

학습 결과: loss 가 0.x → 60 → 1500 사이로 진동.

### 1.3 VRPO 가 분산을 줄이는 3 메커니즘

```text
[1] n_t-MC 평균            V[B̂] ∝ 1/n_t                  (Theorem 2)
[2] optimal allocation     n_t=n, n_yt=1                 (Theorem 2 corollary)
[3] antithetic sampling    πθ ↔ πref 가 (T,M) 공유         (Theorem 3)
                           V = V_θ + V_ref − 2·Cov  (Cov>0 → 차감)

세 기법 결합 → V[ŝ] 5–10× 감소 (LLaDA 1.5 검증)
```

---

## 2. Phase 0 측정 결과 (2026-05-10, 6 GPU × 60 trials)

Stage 2 ckpt 를 πθ 로 사용, weight perturb 0.001 로 πref 합성. ChEBI captioning 데이터 batch B=8.

### 2.1 Theorem 2 — V[B̂] vs n_t

```text
   eps   n_t      V[B̂]    ratio   ideal 1/n_t
─────────────────────────────────────────────
  0.001    1   0.00377   1.0000        1.0000
  0.001    2   0.00229   0.6088        0.5000   ← sweet spot
  0.001    4   0.00197   0.5227        0.2500
  0.001    8   0.00120   0.3194        0.1250

  0.05     1   0.00411   1.0000        1.0000
  0.05     2   0.00232   0.5637        0.5000
  0.05     4   0.00207   0.5025        0.2500
  0.05     8   0.00133   0.3242        0.1250

  0.1      1   0.00439   1.0000        1.0000
  0.1      2   0.00246   0.5613        0.5000
  0.1      4   0.00210   0.4775        0.2500
  0.1      8   0.00149   0.3387        0.1250
```

### 2.2 Theorem 3 — antithetic V[ŝ] (n_t=4, eps=0.05)

```text
shared seeds (antithetic ON):  V[ŝ] = 1.4e-6
indep seeds  (antithetic OFF): V[ŝ] = 1.7e-4
ratio shared / indep         = 0.003   ← 99.7% 분산 제거
```

### 2.3 해석 / 결정

- **n_t=2 가 비용/효과 sweet spot**: n_t=1→2 에서 V 가 44% 감소, n_t=2→4 는 9% 만 추가 감소 → n_t=4 의 추가 fwd 비용 정당화 안 됨
- **분산 감소가 ideal 1/n_t 보다 약함**: data variance (token 분포) 가 추가 분산 source. 이론은 보존되지만 effective n_t 는 약 0.5× 수준
- **eps 영향 미미**: 1e-3 / 0.05 / 0.1 모두 비슷한 V → 표준 LLaDA eps=1e-3 그대로 사용
- **antithetic 효과 압도적**: 99.7% 분산 제거. 절대 빠뜨리면 안 됨 (구현 필수)
- caveat: weight_perturb=0.001 이 학습 후반 πθ–πref 거리보다 작아 antithetic 효과가 과대평가 가능. 학습 후 ratio 가 0.1–0.3 정도로 상승 예상되지만 그래도 강력

---

## 3. V-MolPO 정식화

### 3.1 Loss 구성

```text
L_total = sft_weight · L_SFT(y_w|x) + molpo_weight · L_pref + anc_weight · L_anchor

  B̂_θ(y; T, M)   = (1/n_t) Σ_j ℓ_πθ(y, t_j, m_j)
  B̂_ref(y; T, M) = (1/n_t) Σ_j ℓ_πref(y, t_j, m_j)            ← 동일 (T, M)
  r_θ(y) = β · (B̂_θ(y) − B̂_ref(y))                            ← antithetic

  margin = r_θ(y_w) − r_θ(y_l)
  L_pref   = −logσ(margin − γ_i)             γ_i = molpo.lambda · |E[r_w,task]|
  L_anchor = −logσ(−(r_θ(y_l) − rejected_lambda · avg_chosen_r̄))
  L_SFT    = MaskedDiffusionLoss(y_w | x)                     ← molpo.batch_division=3 일 때만
```

### 3.2 매 training step 의 흐름 (Phase 1+ 에서 구현)

```text
─────────────── 매 training step (V-MolPO) ───────────────

Input batch (mol_div=2):
  y_w[B], y_l[B]   # B = chosen, B = rejected (페어 정렬)

Step 1 — shared (T, M) 샘플
  seed = global_step
  T[n_t, B], M[n_t, B, L] = sample_shared_TM(seed, n_t=2)
  ↑ θ 와 ref 가 모두 같은 (T, M) 사용 (antithetic, Theorem 3)

Step 2 — π_θ ELBO (gradient 흐름)
  B̂_θ(y_w) = (1/n_t) Σ_j ℓ_θ(y_w, t_j, m_j)    # [B]   ← grad
  B̂_θ(y_l) = (1/n_t) Σ_j ℓ_θ(y_l, t_j, m_j)    # [B]   ← grad

Step 3 — π_ref ELBO (no_grad)
  B̂_ref(y_w) = (1/n_t) Σ_j ℓ_ref(y_w, t_j, m_j)  # [B]
  B̂_ref(y_l) = (1/n_t) Σ_j ℓ_ref(y_l, t_j, m_j)  # [B]

Step 4 — reference-relative reward
  r_θ(y_w) = β · (B̂_θ(y_w) − B̂_ref(y_w))   # [B]
  r_θ(y_l) = β · (B̂_θ(y_l) − B̂_ref(y_l))   # [B]

Step 5 — task-anchor (MolPO 고유, EMA per task)
  per_task_EMA[task] = α · per_task_EMA[task] + (1-α) · r_θ(y_w).detach()
  γ_i = molpo_lambda · |per_task_EMA[task_i]|

Step 6 — Loss 합성
  margin = r_θ(y_w) − r_θ(y_l)                            # [B]
  margin_clipped = min(margin, margin_clip · |r_θ(y_w)|)  # 옵션 (burn-in 1k step)

  L_pref   = −logσ(margin_clipped − γ_i).mean()           # DPO-E core
  L_anchor = −logσ(−(r_θ(y_l) − λ_r · EMA)).mean()        # 옵션 (anc_w=0 default)
  L_SFT    = MaskedDiffusionLoss(y_w | x)                 # mol_div=3 에서만

  L_total = sft_weight · L_SFT
          + molpo_weight · L_pref
          + anc_weight · L_anchor

Step 7 — backward
  L_total.backward()        # gradient 는 θ 로만 흐름 (ref 는 no_grad)
  optimizer.step()
─────────────────────────────────────────────────────────
```

step 당 forward 비용:

- n_t=2 × 2 (chosen+rejected) × 2 (θ+ref) = **8 forwards / step**
- n_t=2 batch dim stack 으로 실제 fwd 호출은 2 회 (θ-pass, ref-pass), batch 첫차원만 4 배

### 3.3 MICCAI vs V-MolPO 비교

```text
┌──────────────────────┬──────────────────────┬────────────────────────┐
│ 항목                  │ MICCAI (broken)       │ V-MolPO                │
├──────────────────────┼──────────────────────┼────────────────────────┤
│ reward                │ β·log π_θ             │ β·(log π_θ − log π_ref)│
│ ELBO 추정             │ 단일 MC               │ n_t-MC 평균             │
│ noise sharing         │ N/A (ref 없음)        │ antithetic (필수)       │
│ instance_loss rescale │ × seq_len / tc 깨짐   │ 직접 ℓ 합산             │
│ NaN guard             │ post-hoc filter       │ collator pre-filter     │
│ task-anchor           │ EMA noise 큼          │ EMA noise 작음 → 신뢰↑ │
│ 1/p_mask 처리         │ 그대로                │ n_t 평균이 흡수         │
└──────────────────────┴──────────────────────┴────────────────────────┘
```

---

## 4. 결정 사항 (Phase 0 결과 반영)

```text
┌──────────────────────┬──────────────────────────────────────────────┐
│ 결정 항목            │ 권장값 / 근거                                │
├──────────────────────┼──────────────────────────────────────────────┤
│ πref ckpt            │ Stage 2 ckpt (last.ckpt)                     │
│                      │ — Stage 3 가 Stage 2 에서 시작하므로 자연 선택│
├──────────────────────┼──────────────────────────────────────────────┤
│ n_t                  │ 2 (Phase 0 sweet spot, 4→2 로 비용 절반)      │
│                      │ — V[B̂] 0.0023 (n_t=1 의 56%)                 │
│                      │ — n_t=4 는 추가 9% 만 감소, 비용 정당화 안 됨 │
│                      │ — n_t=8 ablation 으로만 검토                  │
├──────────────────────┼──────────────────────────────────────────────┤
│ antithetic           │ 필수 ON (Phase 0: 99.7% 분산 제거)            │
├──────────────────────┼──────────────────────────────────────────────┤
│ β                    │ 0.1 (LLaDA 1.5 기본, 현 stage3.yaml=1.0)     │
│                      │ → yaml 변경 필요                             │
├──────────────────────┼──────────────────────────────────────────────┤
│ epochs (Stage 3)     │ 3 → 6 ablation (현 yaml=6)                   │
│                      │ — DPO 류는 일반적으로 1-3 epoch              │
│                      │ — Stage 2 가 이미 수렴된 출발점이라 6 과대 가능│
├──────────────────────┼──────────────────────────────────────────────┤
│ batch_size /         │ stage3.yaml 의 batch_size=1 유지             │
│ effective batch      │ collator 가 2B 만들고 n_t=2 stack →          │
│                      │ effective tensor 첫차원 = 2 × 2 = 4          │
│                      │ accumulate 조정으로 global_batch=1024 유지   │
├──────────────────────┼──────────────────────────────────────────────┤
│ ref_model 메모리     │ LoRA gate=0 toggle 로 backbone 공유          │
│                      │ → 추가 weight 0, activation 만 별도 fwd      │
├──────────────────────┼──────────────────────────────────────────────┤
│ tune_gnn (stage 3)   │ 초기엔 false (Q-Former+graph 정렬은 stage2 끝)│
│                      │ → preference signal 만으로 LoRA 업데이트     │
├──────────────────────┼──────────────────────────────────────────────┤
│ molpo.batch_division │ 2 (순수 DPO-E) → 검증 후 3 ablation          │
├──────────────────────┼──────────────────────────────────────────────┤
│ margin_clip          │ burn-in 1k step 만 ±10                       │
├──────────────────────┼──────────────────────────────────────────────┤
│ NaN guard 위치       │ molpo_collator pre-filter (answer_len ≥ 1)   │
├──────────────────────┼──────────────────────────────────────────────┤
│ EMA anchor 보존      │ checkpoint 에 저장, epoch 시작 시 reset 안함 │
└──────────────────────┴──────────────────────────────────────────────┘
```

---

## 5. 학습 비용 / 속도 최적화

### 5.1 Stage 2 SFT 와 비교

```text
Stage 2 SFT:    1 fwd / step,  batch_dim = B
V-MolPO (n_t=2): 2 fwd / step,  batch_dim = 4·B (chosen+rejected, n_t stack)

step 당 wall-clock ≈ 2 × Stage 2 step  (fwd 횟수만 봤을 때 — activation
메모리는 4× 늘어나므로 실제로는 2-3× 까지 갈 수 있음)

3 epoch 학습:  Stage 2 의 ~6-9 epoch 분량
6 epoch 학습:  Stage 2 의 ~12-18 epoch 분량
```

### 5.2 추가 최적화 옵션 (Phase 1 검증 후 도입)

```text
[A] π_ref ELBO 캐싱 (가장 큰 효과, +50% 빠름)
    학습 시작 전 1회: 각 (sample, fixed step-seed) 에 대해 B̂_ref pre-compute,
    NumPy memmap 또는 HDF5 로 디스크 저장.
    학습 루프: θ forward 만 + 캐시 read.
    제약: (T, M) seed 가 epoch 간 고정 → 변동성 약간 감소
          (B̂_ref 가 fixed 해도 antithetic 효과는 유지됨)

[B] LoRA-toggle vs ref forward 별도 측정
    π_ref = π_θ + LoRA(gate=0). 즉 backbone 공유.
    PEFT 의 gate toggle 은 forward 마다 호출 비용이 있어
    실제로는 π_θ 와 π_ref 를 같은 batch 에 concat 해서 한 번에 forward 가
    더 빠를 수도 있음 → Phase 1 에서 둘 다 측정 후 결정

[C] mol_div=3 (SFT 동시) skip
    초기엔 mol_div=2 만 (L_SFT 항 없음). +33% 빠름 (3분기 vs 2분기).
```

### 5.3 권장 출발 설정

```text
n_t=2, antithetic=true, mol_div=2, 3 epoch, ref 캐싱은 Phase 1 후 검토
→ Stage 2 SFT 의 ~6-9 epoch 분량
→ ref 캐싱 추가 시 ~3-5 epoch 분량
→ 6 epoch 까지 가더라도 ~12-15 epoch 분량 (감당 가능)
```

---

## 6. 변경 대상 파일

### 6.1 신규 파일 (일부 Phase 0 에서 이미 commit 완료)

```text
New_MolDA/
├── src/
│   ├── training/
│   │   ├── vrpo_elbo.py              ✅ Phase 0 commit 완료
│   │   └── v_molpo_loss.py           ← 신규: L_pref + L_anchor (per-task EMA)
│   ├── model/
│   │   └── ref_llada_wrapper.py      ← 신규: π_ref wrapper (LoRA gate=0 toggle)
│   ├── data/
│   │   └── molpo_collator.py         ← 신규: chosen/rejected pair → 2B batch
│   └── configs/
│       └── experiment/
│           └── stage3_v_molpo_chebi_only.yaml   ← 신규: ChEBI 단일 task 검증용
├── test/
│   ├── test_vrpo_elbo.py             ✅ 12 passed
│   ├── test_v_molpo_loss.py          ← 신규
│   └── test_molpo_collator.py        ← 신규
└── scripts/
    ├── measure_vrpo_variance.py      ✅ Phase 0 commit 완료
    ├── train_stage3_v_molpo_phase0.sh ✅ Phase 0 commit 완료
    └── train_stage3_v_molpo_chebi.sh ← Phase 3 단일 task launcher
```

### 6.2 기존 파일 수정

```text
[A] src/training/loss.py
    L38-71  make_noisy() : 외부 (T, M) seed 옵션 받도록 refactor
    L73-217 forward()    : 그대로 (V-MolPO 의 L_SFT 로 재사용)

[B] src/model/molda.py
    L23  __init__       : self.ref_model 추가 (cfg.molpo.enabled and stage==3)
    L65-109 forward     : stage==3 + molpo 분기 → _molpo_forward 호출
    L144 _apply_stage_freeze_policy : stage 3 freeze (LoRA trainable, GNN/Q-Former 옵션)

[C] src/training/trainer.py
    L111-120 training_step : 변경 없음 (loss_dict 기반 구조 유지)
    ADD logging: train/v_molpo/{margin, logps_chosen, logps_rejected,
                                 rewards_chosen, rewards_rejected, B_var}

[D] src/data/datamodule.py
    setup() : stage==3 and cfg.molpo.enabled 면 MolPOTrainCollator 사용

[E] src/data/dataset.py
    ADD: chosen/rejected 컬럼 인식

[F] src/configs/trainer/stage3.yaml
    ADD: n_t=2, antithetic=true, ref_ckpt_path, margin_clip_burn_in=1000
    UPDATE: beta 1.0 → 0.1, max_epochs 6 → 3

[G] src/training/checkpoint.py
    ADD: ref_model state_dict 분리 저장/로드
    ADD: per_task_EMA 보존 (epoch 시작 시 reset 안 함)
```

---

## 7. Implementation Phase 의존 그래프

```text
Phase 0: 분산 측정 실험 (가설 정량 검증, 1 GPU 30분)
─────────────────────────────────────────────────────
  ✅ 완료 (commit 0fffbd8 on feature/stage3_v_molpo)
     - src/training/vrpo_elbo.py + tests/test_vrpo_elbo.py (12 pass)
     - scripts/measure_vrpo_variance.py + train_stage3_v_molpo_phase0.sh
     - 결과: §2 의 표 → n_t=2 채택, antithetic ON 필수
                              │
                              ▼
Phase 1: Reference policy + 데이터 인프라
─────────────────────────────────────────────────────
  ▸ 신규: src/model/ref_llada_wrapper.py (LoRA gate=0 toggle 또는 별도 ckpt 로드)
  ▸ 신규: src/data/molpo_collator.py (chosen/rejected → 2B batch)
  ▸ 수정: src/data/datamodule.py (cfg.molpo.enabled 분기)
  ▸ 수정: src/data/dataset.py (chosen/rejected 컬럼 인식)
  ▸ 수정: src/configs/trainer/stage3.yaml (n_t=2, antithetic, ref_ckpt_path 추가)
  ▸ 신규: tests/test_molpo_collator.py
  ▸ 검증: πθ=πref 일 때 margin=0 수렴, collator 가 정확히 2B 출력
                              │
                              ▼
Phase 2: V-MolPO loss 통합
─────────────────────────────────────────────────────
  ▸ 신규: src/training/v_molpo_loss.py (L_pref + L_anchor + per-task EMA)
  ▸ 수정: src/model/molda.py (_molpo_forward 메서드 + ref_model 멤버)
  ▸ 수정: src/training/loss.py (make_noisy 가 외부 seed 수용)
  ▸ 수정: src/training/checkpoint.py (EMA 보존)
  ▸ 신규: tests/test_v_molpo_loss.py
  ▸ 검증: molpo_batch_division=2 (순수 DPO-E) 우선 검증
                              │
                              ▼
Phase 3: Stage 3 단일 task 검증 (ChEBI captioning)
─────────────────────────────────────────────────────
  ▸ 신규: src/configs/experiment/stage3_v_molpo_chebi_only.yaml
  ▸ 신규: scripts/train_stage3_v_molpo_chebi.sh
  ▸ pretrained_ckpt = stage 2 ckpt
  ▸ ref_ckpt        = stage 2 ckpt (frozen)
  ▸ 1k step burn-in (margin clip 활성화) + 5k step 학습
  ▸ 성공 기준: train/loss 0.x ~ 5 안정, ChEBI exact_match 향상
                              │
                              ▼
Phase 4: Full Stage 3 + Ablation
─────────────────────────────────────────────────────
  ▸ 30+ task full Stage 3 (multi_task)
  ▸ Ablation lineup:
    (i)   MICCAI ref-free baseline (Old_MolDA 결과 재현)
    (ii)  ref-based VRPO simple (no MolPO anchor)
    (iii) V-MolPO 정식 (= ii + per-task EMA anchor)
    (iv)  + Graph-conditioned antithetic (논문 차별화)
  ▸ MoleculeNet/ChEBI/SMolInstruct 전 metric 비교
```

---

## 8. V-MolPO 차별화 포인트 (논문 차별화 후보)

```text
(a) Task-conditioned MC budget
    captioning (긴 답변) → n_t = 4
    property   (짧은 답변) → n_t = 1 또는 2
    Theorem 2 의 unbiased property 는 task별 다른 n_t 에서도 유지

(b) Graph-conditioned antithetic
    Prompt 영역만 (T_p, M_p) 공유, answer 영역은 독립
    
              ┌──────────────┬────────────────┐
      chosen: │ shared noise │ INDEPENDENT    │
              │  (prompt)    │ (answer y_w)   │
              ├──────────────┼────────────────┤
    rejected: │ shared noise │ INDEPENDENT    │
              │  (prompt)    │ (answer y_l)   │
              └──────────────┴────────────────┘
    ⇒ y_w/y_l 길이 mismatch 회피하면서 prompt-side noise cancel

(c) MolPO EMA anchor as control variate
    r_θ(y) − avg_chosen_r̄ ≈ 0 (mean per task)
    ⇒ 평균 0 근방으로 scale 줄여 추가 분산 감소
    ⇒ MolPO 의 task-conditional 안정화가 VRPO 위에서 시너지

(d) MolDA-rephrase 10× 데이터셋과 결합
    같은 분자 y 에 대한 10가지 prompt rephrase
    ⇒ 자연스러운 antithetic batch (prompt diversity)
    ⇒ prompt-noise marginalize, Stage 1 자산 재사용
```

---

## 9. Verification

### 9.1 단위 테스트 (완료 / 예정)

```text
✅ test_vrpo_elbo.py        12 passed
   - reproducibility, Theorem 2/3, NaN safety, shape/API

⏳ test_molpo_collator.py
   - 입력 B sample → 출력 2B tensor (chosen 0:B, rejected B:2B)
   - answer_len ≥ 1 pre-filter 작동
   - batch_division=3 시 3B 출력

⏳ test_v_molpo_loss.py
   - πθ = πref → margin = 0
   - rejected/chosen 답변 swap → margin 부호 반전
   - EMA per-task 가 epoch boundary 통과 시 보존
   - margin_clip burn-in 동안 ±10, 이후 unclipped
```

### 9.2 Phase 3 학습 검증 (ChEBI 단일 task)

```text
WandB 모니터링:
  ✓ train/loss              0.x ~ 5 안정 (이전 0.x → 60 → 1500 폭주 X)
  ✓ train/v_molpo/margin    분산 안정 (std/mean < 1)
  ✓ train/v_molpo/logps_chosen  로깅됨
  ✓ train/v_molpo/logps_rejected 로깅됨 (현재 미로깅 issue 해결됨)
  ✓ train/loss/grad_norm    1.0 이하 안정

생성 품질 (val):
  ✓ ChEBI exact_match    Stage 2 baseline 대비 향상
  ✓ ChEBI MACCS_FTS      Stage 2 baseline 대비 향상
  ✓ ChEBI BLEU-4         Stage 2 baseline 대비 향상
```

---

## 10. Branch / Workflow

```text
develop (current)
   │
   └─→ feature/stage3_v_molpo  ← Phase 0-4 commit 누적
        │
        ├─→ ✅ Phase 0 (commit 0fffbd8): vrpo_elbo + tests + Phase 0 launcher
        ├─→ Phase 1: ref infra + collator (의존 없음, 다음 단계)
        ├─→ Phase 2: loss + molda forward 통합
        ├─→ Phase 3: chebi-only 학습 + 결과
        └─→ Phase 4: full stage 3 + ablation

각 Phase 종료 시점에 develop 으로 PR 가능 (incremental review).
```
