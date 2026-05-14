# Stage 3 V-MolPO — Audit & Fix 결과 정리

> 작성일: 2026-05-12
> 대상 brunch: `feature/stage3_v_molpo`
> origin HEAD: `3b93c1e` (직전 audit fix 3개 push 후 상태)

`plan/stage3.md` 명세와 실제 코드를 대조한 audit + 수정 작업 결과 요약. **무엇이 바뀌었나** + **각 모듈이 지금 어떤 기능을 하나** 두 축으로 정리.

---

## 1. 무엇이 바뀌었나 (이번 audit fix push 3 commits)

### Commit 1 — `b1360e6` **fix(data): rename `_build_pyg_batch` → `_build_graph_batch`**
- **파일**: [src/data/molpo_collator.py](../src/data/molpo_collator.py) L31, L204
- **원인**: V-MolPO Phase 1 초기 commit (5/10, `b3291ad`)이 collator에 `_build_graph_batch`로 추가될 helper를 `_build_pyg_batch`라는 이름으로 import. helper 함수 자체는 collator에 추가되지 않은 채로 push되어 stage 1/2 module load에서도 `ImportError`로 죽었음. 우리가 직전 stage 2 작업에서 datamodule.py에 lazy import 박아서 stage 1/2는 살렸지만 **molpo_collator 자체의 이름 오류는 그대로**였음. Stage 3 + `mol_representation=string+graph` 진입 시 `NameError` 즉시 crash.
- **영향**: Stage 3 학습 가능 상태로 unlock. Stage 1/2 무영향.

### Commit 2 — `dda728e` **chore(stage3): expose `ema_alpha`, disable margin clip, comment eps source**
- **파일**: [src/configs/trainer/stage3.yaml](../src/configs/trainer/stage3.yaml), [src/configs/experiment/stage3_v_molpo_chebi_only.yaml](../src/configs/experiment/stage3_v_molpo_chebi_only.yaml), [src/training/vrpo_elbo.py](../src/training/vrpo_elbo.py)
- **변경**:
  - `ema_alpha: 0.99` yaml에 명시 (기존엔 `molda.py:84`에서 silent default). 같은 값이지만 ablation 시 yaml로 조정 가능 + 의도 명시.
  - `margin_clip_burn_in: 1000 → 0` (yaml 둘 다). VRPO 논문(LLaDA-1.5, arxiv 2505.19223)에 clipping 없음(WebFetch 확인), Mol-LLM MolPO 측 clipping은 optional anchor. **user 결정**으로 빼기. 코드 경로는 보존 — `v_molpo_loss.py:152`가 `global_step < clip_burn_in`을 gate로 쓰므로 burn_in=0이면 자연스럽게 비활성. ablation 시 yaml에 큰 수치 다시 넣으면 됨.
  - `DEFAULT_EPS = 1e-3 # LLaDA paper standard (plan/stage3.md §4)` 주석 추가. 값 출처 traceability 확보.

### Commit 3 — `3b93c1e` **docs(adapter): document why `blip2qformer/gine_loader/help_funcs` are kept unused**
- **파일**: [src/model/adapter/blip2qformer.py](../src/model/adapter/blip2qformer.py), [src/model/adapter/gine_loader.py](../src/model/adapter/gine_loader.py), [src/model/adapter/help_funcs.py](../src/model/adapter/help_funcs.py)
- **사유**: 위 3개 파일은 Old_MolDA에서 vendoring했지만 현재 코드 경로 어디서도 import 안 됨 (dead). 단순 삭제 대신 module-level docstring으로 **(a) 왜 dead인지**, **(b) 어떤 시나리오에 부활하는지** 명시.
  - `blip2qformer.py`: contrastive pretraining (ITM/ITC head) 도입 시 부활 (plan §8(d) Mol-rephrase 10× contrastive 안)
  - `gine_loader.py`: 현재는 `gine_tokengt.py`의 `_strip_prefix` 통합 loader로 충분. 동적 GNN backbone dispatch 필요 시 부활
  - `help_funcs.py`: `blip2qformer.py` 부활 시 transitively 살아남
- **영향**: 코드베이스 의도 명확화. git blame/grep으로 "이거 왜 있냐" 질문 차단.

---

## 2. 의도적으로 변경 안 한 항목

- **`nn.LayerNorm` (gine_tokengt 측) vs custom `LayerNorm` (qformer 측) 혼용**: working wandb run `lfi4r8cv`(5/9)가 같은 구성으로 학습 성공했음. 변경 시 dtype precision 회귀 위험. 별도 issue로 분리.
- **`print()` → `logger.info()` refactor**: trainer.py / validation.py에 DDP debug print 다수. 별도 cleanup PR.
- **`lr.X: ???` mandatory placeholders**: 현재 `stage3_v_molpo_chebi_only.yaml`이 override. 다른 stage 3 experiment 추가 시 잊을 위험 있지만 현재 다른 experiment 없으므로 보류.

---

## 3. 현재 각 모듈이 하는 일 (Stage 3 기준)

### 3.1 학습 진입점
- [scripts/train_stage3_v_molpo_chebi.sh](../scripts/train_stage3_v_molpo_chebi.sh): Phase 3 ChEBI 단일 task launcher. `pretrained_ckpt`/`molpo.ref_ckpt_path` 둘 다 Stage 2 ckpt로 세팅. 6 GPU DDP, batch_size=4 per GPU, accum 10 → global 240.
- [scripts/train_stage3_v_molpo_phase0.sh](../scripts/train_stage3_v_molpo_phase0.sh): Phase 0 variance 측정 launcher (학습 아님). `measure_vrpo_variance.py` 호출. 이미 완료 상태.

### 3.2 모델 forward 구조
```
MolDA.forward(batch)
  ├─ molpo_enabled & "molpo_batch_size" in batch → _molpo_forward(batch)   [Stage 3 V-MolPO 경로]
  └─ 그 외 → 표준 SFT forward                                                [Stage 1/2 경로]
```

**Stage 3 V-MolPO 경로** ([src/model/molda.py:178-322](../src/model/molda.py#L178)):
1. `batch_division=2`이면 `[0:B]=chosen`, `[B:2B]=rejected`로 슬라이스 (3이면 sft도 0:B)
2. `vrpo_elbo.compute_elbo(..., n_t=2, seed=...)`로 πθ ELBO 계산 (gradient 흐름)
3. 같은 seed로 `ref_model.make_forward_fn(...)` 통해 πref ELBO 계산 (no_grad)
4. `compute_v_molpo_loss(...)` → `r_θ(y) = β·(B̂_θ − B̂_ref)`, margin, EMA-based γ, sigmoid loss
5. `combine_total_loss(...)` → `L_total = sft_weight·L_SFT + molpo_weight·L_pref + anc_w·L_anchor`

### 3.3 핵심 모듈

| 파일 | 역할 |
|---|---|
| [src/training/vrpo_elbo.py](../src/training/vrpo_elbo.py) | **VRPO ELBO 추정기**. `sample_shared_TM(eps=1e-3)` — n_t × B 개 timestep + mask를 deterministic seed로 sample (antithetic용 공유). `compute_elbo(forward_fn, ids, lab, n_t, seed, ...)` — n_t-MC 평균 ELBO. `compute_dpo_e_score(...)` — β·(B̂_θ_w − B̂_ref_w) − β·(B̂_θ_l − B̂_ref_l) margin. **162줄, test 12 cases pass.** |
| [src/training/v_molpo_loss.py](../src/training/v_molpo_loss.py) | **V-MolPO loss + per-task EMA**. `TaskAnchorEMA` (alpha=0.99, state_dict 지원 — checkpoint round-trip OK). `compute_v_molpo_loss(elbo_w/l × θ/ref, tasks, ema, β, λ, margin_clip_*)` — DPO-E sigmoid + 옵션 anchor + 옵션 margin clip. `combine_total_loss(...)` — L_total. **211줄, test 17 cases pass.** |
| [src/model/ref_llada_wrapper.py](../src/model/ref_llada_wrapper.py) | **π_ref wrapper**. Stage 2 ckpt를 별도 MolDA 인스턴스로 로드, 모든 param freeze + `.train()` override로 eval mode 고정. inner cfg에서 `molpo.enabled=False`로 RefMolDA 재귀 방지. `make_forward_fn(sub_batch) → callable` — `compute_elbo`가 받을 forward 함수 factory. string-only/string+graph 둘 다 지원. **158줄, test 6 cases pass.** |
| [src/model/molda.py](../src/model/molda.py) | **MolDA 통합 모델**. `__init__`에서 stage≥2면 GNN/QFormer, stage==3 & molpo.enabled면 ref_model + task_anchor_ema 추가 설치. `_forward_with_graph` — GNN → QFormer → `<mol>` placeholder 자리에 inject → LLaDA. `_molpo_forward` — Stage 3 V-MolPO 메인 루프. `_forward_logits` — VRPO ELBO 호출용 logits-only 서브루틴. `_slice_batch` — chosen/rejected 분리 (PyG Batch `to_data_list` → slice → `from_data_list`). `_apply_stage_freeze_policy` — stage 3에서 LoRA 학습 + GNN/QFormer는 `tune_gnn`에 따라 + ref_model은 항상 frozen. |
| [src/data/molpo_collator.py](../src/data/molpo_collator.py) | **chosen/rejected pair → 2B (or 3B) batch**. 답변 길이 0 sample pre-filter (NaN 방어), `molpo_batch_size`/`molpo_batch_division` 키 emit (MolDA forward 분기 신호), graph 시 chosen/rejected에 동일 graph replicate. `_build_graph_batch` (collator.py 공유 helper) 사용. |
| [src/data/collator.py](../src/data/collator.py) | **표준 SFT collator + graph batching helper**. `_build_graph_batch(samples, x_key, ei_key, ea_key)` — PyG `Data` 리스트 → `Collater`. TrainCollator/EvalCollator가 `mol_representation`에 "graph" 포함 시 호출. |
| [src/data/datamodule.py](../src/data/datamodule.py) | **train/val/test loader 구성**. `_build_train_collator`가 `molpo.enabled` 시 MolPOTrainCollator (lazy import) 아니면 TrainCollator 반환. |
| [src/training/checkpoint.py](../src/training/checkpoint.py) | **state_dict 필터 + EMA persist**. `on_save_checkpoint`: `ref_model.*` 키 drop, trainable + 의미있는 frozen만 보존; `task_anchor_ema.state_dict()`를 `v_molpo_task_anchor_ema` 별도 키로 저장. `on_load_checkpoint`: EMA 복원 (epoch boundary 보존, reset 안 함). |
| [src/training/loss.py](../src/training/loss.py) | **MaskedDiffusionLoss**. SFT path. `make_noisy(t_override, mask_indices_override)`로 외부 (T, M) 주입 가능 — VRPO antithetic 호환. |
| [src/training/trainer.py](../src/training/trainer.py) | **Lightning module**. `training_step`이 model output dict에서 `v_molpo/*` prefix 키 (loss_pref, margin, rewards_chosen/rejected, gamma 등)를 wandb로 자동 logging. |

### 3.4 vendored adapter (Old_MolDA)
모두 [src/model/adapter/](../src/model/adapter/) 아래.

| 파일 | 상태 | 역할 |
|---|---|---|
| `blip2.py` | **사용 중** | `Blip2Base.init_Qformer/init_tokenizer` + custom `LayerNorm` (fp16 안전). `src/model/qformer.py`가 사용. |
| `gin_model.py` | **사용 중** | `GNN_MoleculeSTM` (GINE backbone). `gine_tokengt.py` 내부. |
| `tokenGT.py` | **사용 중** | `BERTTokenGT` (TokenGT graph transformer, flash-attn BertEncoder). `gine_tokengt.py` 내부. |
| `gine_tokengt.py` | **사용 중** | `GINE_TokenGT` — GINE+TokenGT 병렬 인코더 + LayerNorm + concat. `_strip_prefix` helper로 raw `gnn.*` ckpt와 stage 2-wrapped `blip2model.graph_encoder.*` ckpt 둘 다 로드. `src/model/gnn.py`가 wrap. |
| `added_tokens.py` | **사용 중** | special token 상수. `llada_wrapper.py`, `dataset_generation/generator.py`가 import. |
| `blip2qformer.py` | **dead (의도)** | Old_MolDA contrastive `Blip2Qformer` (ITM/ITC). 부활 시나리오: contrastive pretraining 도입. |
| `gine_loader.py` | **dead (의도)** | 동등 GNN loader. 현재는 `gine_tokengt.py`의 통합 loader가 충분. |
| `help_funcs.py` | **dead (의도)** | `blip2qformer.py`만 사용. transitively dead. |

### 3.5 config (Stage 3 기준 resolved 값)
```
stage: 3
model.mol_representation: string+graph
model.tune_gnn: true                 # Stage 3에선 GNN/QFormer도 학습
training.max_epochs: 3 (default) / 20 (chebi_only experiment override)
training.batch_size: 1 (default) / 4 (chebi_only override)
training.global_batch_size: 1024 / 240

molpo.enabled: true
molpo.n_t: 2                         # Phase 0 sweet spot
molpo.antithetic: true               # 99.7% 분산 제거
molpo.beta: 0.1                      # LLaDA 1.5 표준
molpo.batch_division: 2              # 순수 DPO-E (sft 슬라이스 없음)
molpo.ema_alpha: 0.99                # ← 이번에 yaml에 명시
molpo.margin_clip_burn_in: 0         # ← 이번에 비활성화
molpo.margin_clip_scale: 1.0         # 무관 (burn_in=0이라 발동 안 함)
molpo.sft_weight: 1.0
molpo.molpo_weight: 0.25
molpo.anc_rejected_weight: 0.0       # anchor loss off
molpo.ref_ckpt_path: ???             # CLI override 필수

lr.lora: 1.25e-4 / embed_orig: 1.25e-5 / embed_new: 1.25e-5
lr.head_orig/new: 0.0                # weight_tying=true → 미사용
lr.other: 2.5e-5                     # Q-Former + GNN
```

---

## 4. 검증 상태

| 검증 | 결과 |
|---|---|
| `pytest test/test_molpo_collator.py test/test_v_molpo_loss.py test/test_vrpo_elbo.py test/test_ref_llada_wrapper.py` | **47 passed** |
| Hydra dry compose (`+experiment=stage3_v_molpo_chebi_only trainer=stage3`) | OK — `margin_clip_burn_in=0`, `ema_alpha=0.99`, `n_t=2`, `antithetic=True`, `beta=0.1`, lr 전부 resolve |
| `git push origin feature/stage3_v_molpo` | OK — `eb7de9a..3b93c1e` |

---

## 5. 다음 단계

학습 진입 가능 상태. 다음 작업 후보:

1. **Phase 3 ChEBI 학습 시작**: `bash scripts/train_stage3_v_molpo_chebi.sh` 또는 같은 명령어 inline. Stage 2 ckpt를 πθ + πref로 사용. 1k step 학습해서 train/loss 안정성 확인 후 본격 학습.
2. **남은 minor 항목 처리** (별도 PR로):
   - `nn.LayerNorm` vs custom `LayerNorm` 혼용 정밀 검토
   - `print()` → `logger` refactor
   - `lr.X: ???` placeholder defensive default
3. **Phase 4 full Stage 3 + ablation**: 30+ task multi_task, V-MolPO vs MICCAI baseline vs ref-based VRPO 비교

---

## 6. 작업 외 남은 working tree

- [scripts/fetch_wandb_reference.py](../scripts/fetch_wandb_reference.py) — `D` 상태 (우리가 안 만진 파일, pod 복구 직후부터 deleted). 의도 불명 → 별도 처리 필요 시 user 판단.
- `nohup.out` — cruft, `rm` 권장.

---

## 부록: 직전 push log

```
3b93c1e docs(adapter): document why blip2qformer/gine_loader/help_funcs are kept unused
dda728e chore(stage3): expose ema_alpha, disable margin clip burn-in, comment eps source
b1360e6 fix(data): rename _build_pyg_batch -> _build_graph_batch in molpo_collator
─── 이상 이번 audit fix ───
eb7de9a chore: ignore local venv/ directory
7233c7b feat(experiment): 10xRephrase data + experiment configs for Stage 2
41f1815 fix(data): lazy import of MolPOTrainCollator in datamodule
0e88530 feat(data): PyG graph batching in TrainCollator/EvalCollator for string+graph
7dc7129 feat(stage2): real QFormer + GINETokenGT adapters wrapping vendored Mol-LLM
a7bded1 feat(model): vendor Mol-LLM legacy implementation for Stage 2 bridge
─── 이상 Stage 2 push ───
b016e24 tune(stage3): val every 5 epoch (epoch-based, simpler)  ← 원래 origin HEAD
```
