# Alignment 문제? MolPO 문제? — V-MolPO 도입 결정 보고

> 작성일: 2026-05-13
> 작성자: gyuang
> 대상: MolDA 팀 (alignment 관련 자료 조사 요청에 대한 응답)
> 결론: **alignment 자체가 아닌 MICCAI MolPO 알고리즘이 LLaDA 위에서 broken** — VRPO (LLaDA 1.5) 기반의 V-MolPO 로 재설계 완료

---

## 0. TL;DR (한 문단 요약)

팀 가설은 "Stage 3 alignment (= preference fine-tuning) 단계 어딘가에 문제가 있다"였지만, 자료 조사 결과 **alignment 단계 자체가 아니라 그 안에서 사용한 MICCAI MolPO 알고리즘이 LLaDA(Diffusion LM) 가정을 깨버려서 학습이 broken**임을 확인. 원인은 (1) Diffusion LM 의 `log π(y|x)` 가 ELBO 형식이라 단일 MC 샘플로 추정할 수 없는데 MolPO 는 그걸 그대로 DPO 에 넣음, (2) reference policy 부재로 DPO theory 자체 미충족. 해결책으로 LLaDA 1.5 의 **VRPO (Variance-Reduced Preference Optimization)** + MolPO 의 **per-task EMA anchor** 를 결합한 **V-MolPO** 를 설계·구현 완료. Phase 0 분산 측정으로 `n_t=2 + antithetic ON` 의 비용/효과 sweet spot 정량 검증, Phase 1-2 인프라 구현 완료, Phase 3 ChEBI 단일 task 학습 launcher까지 준비됨.

---

## 1. 팀이 본 문제 — "alignment 어딘가에 문제가 있다"

MolDA Stage 3 V-MolPO 학습 시 다음 증상 보고됨 (MICCAI 2026 Tpod 결과 + 직전 wandb runs):

```text
train/loss     0.x → 60 → 1500 사이로 폭주
train/grad_norm  발산
margin (logp_w − logp_l)  std 매우 큼 (mean 의 10× 이상)
generation 품질  Stage 2 baseline 대비 향상 없음 또는 악화
```

팀 가설: 
- "alignment (preference) 학습 자체가 LLaDA 와 안 맞을 수 있다"
- "Stage 2→3 ckpt 전환에서 모델 분포가 깨졌을 수 있다"
- "GNN/Q-Former bridge 가 alignment 신호를 제대로 못 받을 수 있다"

조사 요청: alignment 관련 reference 자료 + 현 구현 점검.

## 2. 내가 본 문제 — "alignment 가 아니라 MolPO 알고리즘이 broken"

가설: alignment 단계 자체는 LLaDA + diffusion 위에서도 가능. 다만 우리가 쓰는 **MICCAI MolPO 가 causal LM 을 가정한 DPO derivative** 라서 LLaDA 같은 masked diffusion LM 에서 깨짐.

근거 1. Diffusion LM 의 `log π(y|x)` 본질 차이:

```text
Causal LM (Mistral 등)                  Diffusion LM (LLaDA)
─────────────────────                  ────────────────────
log π(y|x) = Σ log p(y_t | y_<t, x)    log π(y|x) ≥ E_t E_q [ℓ(y_t, t)]   ← ELBO
            └─ deterministic ─┘                   └── stochastic ──┘

Single fwd → variance = 0              Single fwd, 1 sample of t
                                       → variance = Θ(1)
                                       → 1/p_mask 가 작은 t 에서 폭증
```

DPO 류는 **결정론적 `log π`** 를 가정. LLaDA 는 ELBO 라 단일 MC 추정 사용 시 분산 Θ(1). 그 분산이 그대로 `−logσ(β·s)` 의 `s` (margin) 에 들어가 grad 폭주.

근거 2. MICCAI MolPO 의 구체적 구현 결함:

```text
y_w ──→ [LLaDA fwd 1×] ──→ instance_loss_w ──→ logp_w
                              (rescale broken)        ↘
y_l ──→ [LLaDA fwd 1×] ──→ instance_loss_l ──→ logp_l ──→ margin
                                                            │
                                                loss = −logσ(β·margin)

Failure modes:
  ❌ no π_ref           → DPO theory 미충족 (β·log π 만 사용, reference 없음)
  ❌ single MC sample   → margin variance Θ(1) 그대로
  ❌ rescale 깨짐        → instance_loss × seq_len / token_count
                         (instance_loss 는 이미 answer_len 기준 평균)
  ❌ 1/p_mask amplifies → t≈ε 에서 1000× 까지 튐
```

이 4가지가 서로 곱해져서 학습이 0.x → 60 → 1500 진동.

→ **alignment 단계 자체가 broken 인 게 아니라, 그 안에서 쓴 MolPO 가 LLaDA gradients 를 폭주시키고 있다는 결론**.

---

## 3. 참고 자료

| # | 출처 | 핵심 포인트 |
|---|---|---|
| **A1** | **LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models** (Zhu et al., arXiv:2505.19223v2, 2025) — `/opt/Reference/LLADA 1.5_*.pdf` | DPO 의 `log π` 를 ELBO 로 대체하면 분산 폭주 → **n_t-MC 평균 (Theorem 2)** + **antithetic noise sharing (Theorem 3)** + **reference policy 명시적 사용**으로 분산 5–10× 감소. 우리 V-MolPO 의 backbone. |
| **A2** | **LLaDA: Large Language Diffusion Models** (`/opt/Reference/LLaDA_*.pdf`) | Masked diffusion LM 의 forward process 정의. ELBO objective + `1/p_mask` rescale. eps=1e-3 timestep floor 의 출처. |
| **A3** | **Mol-LLM: Multimodal Generalist Molecular LLM with Improved Graph Utilization** (`/opt/Reference/Mol-LLM_*.pdf`) | MolPO 원형: per-task anchor + chosen/rejected pair 학습. **task-conditional 안정화** 아이디어는 살릴 가치 있음. 단, 원본 MolPO 는 causal Mistral 에서 디자인됐고 reference policy 없음 → LLaDA 로 그대로 옮기면 §2 의 4가지 결함 발생. |
| A4 | LLaDA-V (`/opt/Reference/LLaDA-V_*.pdf`) | LLaDA 위에서 multimodal instruction tuning 한 사례. SFT 단계까지는 잘 동작 → "alignment 자체가 LLaDA 에서 불가능"한 게 아님을 보여주는 반례. |
| A5 | Direct Preference Optimization (Rafailov et al., 2023) — DPO 원논문 | DPO 의 reference-relative reward `r_θ = β·(log π_θ − log π_ref)` 정의. **reference 없이 β·log π 만 쓰면 DPO 가 아님** — 우리 MICCAI MolPO 가 정확히 그 결함. |
| A6 | LLaDA 1.5 official repo (논문 링크) | n_t-MC 의 reference 구현. n=8 default 라고 명시 (우리 Phase 0 측정으로 n_t=2 가 비용/효과 sweet spot 결정). |

추가로 검증 가능한 것:
- 직전 MICCAI 2026 Tpod 학습의 wandb run logs (loss 폭주 패턴)
- **Phase 0 분산 측정 실험 결과** (n_t / antithetic 효과 정량) — `/opt/MolDA/plan/stage3.md` §2

---

## 4. 진단 → 검증 (Phase 0)

가설을 코드로 정량 검증 (2026-05-10, 6 GPU × 60 trials):

### 4.1 Theorem 2 — `V[B̂]` vs `n_t` 검증

```text
   eps   n_t      V[B̂]    ratio   ideal 1/n_t
─────────────────────────────────────────────
  0.001    1   0.00377   1.0000        1.0000
  0.001    2   0.00229   0.6088        0.5000   ← sweet spot
  0.001    4   0.00197   0.5227        0.2500
  0.001    8   0.00120   0.3194        0.1250
```

### 4.2 Theorem 3 — antithetic 효과

```text
shared seeds (antithetic ON):  V[ŝ] = 1.4e-6
indep seeds  (antithetic OFF): V[ŝ] = 1.7e-4
ratio shared / indep         = 0.003   ← 99.7% 분산 제거
```

### 4.3 결론

- **n_t=2** 가 cost/효과 sweet spot (n_t=1→2 에서 V 44% 감소, n_t=2→4 는 +9% 만 추가, 비용 정당화 안 됨)
- **antithetic 필수 ON** (단순히 같은 seed 공유로 99.7% 분산 제거 — 거의 free lunch)
- eps 1e-3 / 0.05 / 0.1 모두 비슷한 V → 표준 LLaDA eps=1e-3 그대로 사용

→ "alignment 가 LLaDA 에서 어렵다"가 아니라 **"n_t=2 + antithetic 만 갖추면 분산이 잡힌다"** 는 정량 근거.

코드: [src/training/vrpo_elbo.py](../src/training/vrpo_elbo.py), 측정 스크립트: [scripts/measure_vrpo_variance.py](../scripts/measure_vrpo_variance.py), launcher: [scripts/train_stage3_v_molpo_phase0.sh](../scripts/train_stage3_v_molpo_phase0.sh).

---

## 5. 현재 구현 상태 — V-MolPO

### 5.1 한 줄 정의

> **V-MolPO = MolPO 의 task-anchor 안정화 + VRPO 의 reference-based ELBO + n_t-MC + antithetic sampling**

MICCAI MolPO 의 의도 (per-task 안정화) 는 살리고, LLaDA 호환성 결함 4가지 (no ref, single MC, rescale, 1/p_mask 폭주) 는 VRPO 도구로 전부 차단.

### 5.2 Loss 구성

```text
L_total = sft_weight · L_SFT(y_w|x) + molpo_weight · L_pref + anc_weight · L_anchor

  B̂_θ(y; T, M)   = (1/n_t) Σ_j ℓ_πθ(y, t_j, m_j)              ← n_t-MC ELBO
  B̂_ref(y; T, M) = (1/n_t) Σ_j ℓ_πref(y, t_j, m_j)            ← antithetic 동일 (T, M)
  r_θ(y) = β · (B̂_θ(y) − B̂_ref(y))                            ← reference-relative reward

  margin    = r_θ(y_w) − r_θ(y_l)
  L_pref    = −logσ(margin − γ_i)            γ_i = molpo_lambda · |E[r_w,task]|
  L_anchor  = −logσ(−(r_θ(y_l) − rejected_lambda · avg_chosen_r̄))
```

### 5.3 매 step 흐름

```text
Step 1 — shared (T, M) 샘플 (seed = global_step·1000+offset)
         T[n_t, B], M[n_t, B, L] = sample_shared_TM(seed, n_t=2)
         ↑ θ 와 ref 가 같은 (T, M) 사용 (antithetic, Theorem 3)

Step 2 — π_θ ELBO (gradient 흐름)
         B̂_θ(y_w), B̂_θ(y_l)   ← grad

Step 3 — π_ref ELBO (no_grad)
         B̂_ref(y_w), B̂_ref(y_l)

Step 4 — reference-relative reward
         r_θ(y_w) = β · (B̂_θ(y_w) − B̂_ref(y_w))
         r_θ(y_l) = β · (B̂_θ(y_l) − B̂_ref(y_l))

Step 5 — per-task EMA anchor (MolPO 고유)
         per_task_EMA[task] = α · per_task_EMA[task] + (1-α) · r_θ(y_w).detach()
         γ_i = molpo_lambda · |per_task_EMA[task_i]|

Step 6 — Loss 합성
         L_pref   = −logσ(margin − γ_i).mean()
         L_anchor = −logσ(−(r_θ(y_l) − λ_r · EMA)).mean()         (anc_weight=0 default → off)
         L_SFT    = MaskedDiffusionLoss(y_w | x)                  (mol_div=3 에서만)

Step 7 — backward
         L_total.backward()        # gradient 는 θ 로만 흐름 (ref 는 no_grad)
```

step 당 fwd 비용: `n_t=2 × 2 (chosen+rejected) × 2 (θ+ref) = 8 forwards / step` (실제로는 n_t batch dim stack 으로 fwd 호출 4회).

### 5.4 MICCAI vs V-MolPO 비교

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

### 5.5 코드 인벤토리

| 파일 | 역할 | LOC | 테스트 |
|---|---|---|---|
| [src/training/vrpo_elbo.py](../src/training/vrpo_elbo.py) | n_t-MC + antithetic ELBO | 162 | test_vrpo_elbo.py (12 pass) |
| [src/training/v_molpo_loss.py](../src/training/v_molpo_loss.py) | DPO-E loss + per-task EMA anchor | 211 | test_v_molpo_loss.py (17 pass) |
| [src/model/ref_llada_wrapper.py](../src/model/ref_llada_wrapper.py) | frozen π_ref (Stage 2 ckpt 별도 load) | 158 | test_ref_llada_wrapper.py (6 pass) |
| [src/data/molpo_collator.py](../src/data/molpo_collator.py) | chosen/rejected → 2B/3B batch | 208 | test_molpo_collator.py |
| [src/model/molda.py](../src/model/molda.py) | `_molpo_forward`/`_forward_logits`/`_slice_batch`/freeze policy stage 3 | (전체 ~430) | test_molpo_forward_dispatch.py + test_stage3_freeze_policy.py |
| [src/training/checkpoint.py](../src/training/checkpoint.py) | EMA persist + ref_model state filter | (mixin 일부) | test_audit_regression.py (round-trip) |

config: [src/configs/trainer/stage3.yaml](../src/configs/trainer/stage3.yaml) (`n_t=2, antithetic=true, beta=0.1, ema_alpha=0.99, margin_clip_burn_in=0`).

launcher: [scripts/train_stage3_v_molpo_chebi.sh](../scripts/train_stage3_v_molpo_chebi.sh) — ChEBI 단일 task 검증용 (Phase 3).

총 테스트 회귀 가드: V-MolPO 관련 47+ cases (모두 pass 확인).

### 5.6 audit 결과 (직전 일감)

세부: [plan/stage3_audit_summary.md](stage3_audit_summary.md), [plan/validation_generation_audit.md](validation_generation_audit.md).

핵심:
- CRITICAL bug 1개 fix (`_build_pyg_batch` → `_build_graph_batch` rename)
- margin clipping 비활성 (VRPO 논문 미사용 + MolPO 측 optional)
- ema_alpha yaml 명시
- vendored Old_MolDA 미사용 모듈 docstring 보강
- validation.py config default mismatch 1건 수정 + 회귀 테스트 추가

→ 현 시점에 V-MolPO 학습 진입 가능 상태.

---

## 6. 앞으로의 방향

### Phase 3 (다음 단계, 1-2 주)

**ChEBI 단일 task 검증** — V-MolPO 가 실제로 분산 안정성을 가져오는지 정성/정량 확인.

```bash
# launcher 그대로 실행 가능
bash scripts/train_stage3_v_molpo_chebi.sh
```

성공 기준 (wandb 모니터링):
- `train/loss` 0.x ~ 5 안정 (이전 0.x → 60 → 1500 폭주 X)
- `train/v_molpo/margin` 분산 안정 (std/mean < 1)
- `train/v_molpo/logps_chosen` + `logps_rejected` 둘 다 로깅됨 (MICCAI 미로깅 issue 해결됨)
- `train/loss/grad_norm` 1.0 이하 안정
- ChEBI exact_match / MACCS_FTS / BLEU-4: Stage 2 baseline 대비 향상

### Phase 4 (1-2 개월)

**Full Stage 3 + ablation lineup**:

```text
30+ task multi_task 학습:
  (i)   MICCAI ref-free baseline (Old_MolDA 결과 재현)
  (ii)  ref-based VRPO simple (per-task EMA anchor 끔)
  (iii) V-MolPO 정식 (= ii + per-task EMA anchor)
  (iv)  + Graph-conditioned antithetic (논문 차별화)
  
지표: MoleculeNet / ChEBI / SMolInstruct 전 metric 비교
```

### 추가 차별화 카드 (논문 contribution 후보)

`plan/stage3.md §8`:
1. **Task-conditioned MC budget** — captioning(긴 답변)은 n_t=4, property(짧은 답변)은 n_t=1~2 — Theorem 2 의 unbiased property 는 task별 다른 n_t 에서도 유지
2. **Graph-conditioned antithetic** — Prompt 영역만 (T_p, M_p) 공유, answer 영역은 독립. y_w/y_l 길이 mismatch 회피하면서 prompt-side noise cancel
3. **MolPO EMA anchor as control variate** — `r_θ(y) − avg_chosen_r̄` 로 scale 줄여 추가 분산 감소
4. **MolDA-rephrase 10× 데이터셋과 결합** — 같은 분자 y 에 대한 10가지 prompt rephrase 자체가 자연스러운 antithetic batch

### 기술 부채 처리 (병행)

`plan/validation_generation_audit.md` 의 deferred WARN 항목 13개:
- `_val_custom_chart_history` thread safety lock 도입 (Stage 2/3 validation phase 진입 직전)
- `cfg.generation` 직접 접근 → `.get()` 패턴 통일
- `mask_id=126336` 3개 파일 hardcode → `MASK_TOKEN_ID` import 통일
- `print()` → `logger` 일괄 refactor (별도 PR)

---

## 7. 팀 의사결정 요청

이 보고를 기반으로 합의가 필요한 항목:

1. **방향성**: alignment 단계 자체 재설계 (예: SFT-only 로 회귀, RLHF 로 전환)가 아니라 **MolPO → V-MolPO 교체**로 가는 방향에 동의?
2. **자원**: Phase 3 ChEBI 단일 task 검증을 위한 GPU 시간 배정 (6×Blackwell × ~ 1일 예상). 이미 launcher 준비됨.
3. **publish 전략**: V-MolPO 자체를 contribution 으로 publish 할지 vs MolDA architecture 의 학습 trick 으로 footnote 처리할지. §6의 차별화 카드 (특히 graph-conditioned antithetic) 가 살아있다면 별도 paper 가능.

---

## 부록 A. 한 페이지 cheat sheet

```text
문제 가설 (팀):  "alignment 단계 어딘가에 문제 있다"
실제 원인 (조사): MolPO 가 LLaDA 의 ELBO objective 와 호환 안 됨 (4가지 결함)
해결책:          VRPO (n_t-MC + antithetic + reference) + MolPO (per-task anchor) 결합
                = V-MolPO

핵심 reference:  LLaDA 1.5 paper (arXiv:2505.19223), Mol-LLM paper, DPO 원논문, LLaDA 원논문
구현 위치:       src/training/{vrpo_elbo.py, v_molpo_loss.py}, src/model/ref_llada_wrapper.py, src/data/molpo_collator.py
검증:            Phase 0 분산 측정 (n_t=2 sweet spot, antithetic 99.7% 감소) + 47+ unit tests
다음 단계:       Phase 3 ChEBI 단일 task 학습 (launcher: scripts/train_stage3_v_molpo_chebi.sh)
```

---

## 부록 B. 관련 문서

- [plan/stage3.md](stage3.md) — V-MolPO 도입 plan + Phase 0 결과
- [plan/stage3_audit_summary.md](stage3_audit_summary.md) — audit fix 결과 정리
- [plan/architecture_old_vs_new.md](architecture_old_vs_new.md) — Old_MolDA vs New_MolDA 비교
- [plan/validation_generation_audit.md](validation_generation_audit.md) — validation/generation 코드 감사
- 논문 4개: [/opt/Reference/](/opt/Reference/) (LLaDA, LLaDA 1.5, LLaDA-V, Mol-LLM)
