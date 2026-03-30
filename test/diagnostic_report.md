# MolDA Training Step Diagnostic Report

> Generated: 2026-03-28 01:31:25
> Device: CUDA NVIDIA L40
> max_length=512, gen_max_len=256 (프로덕션 동일)

## 1. 모델 로딩

- LLM: `GSAI-ML/LLaDA-8B-Instruct`
- Original vocab size: **126,349**
- Expanded vocab size: **129,325** (+2,976 tokens)
- LoRA: r=64, alpha=32
- Precision: bfloat16

### Parameter Summary
| 구분 | 파라미터 수 | 비율 |
|------|-----------|------|
| Total | 8,122,904,576 | 100% |
| Trainable | 613,601,280 | 7.55% |
| LoRA | 83,886,080 | 1.03% |
| Embedding (wte, tied to output) | 529,715,200 | 6.52% |
| Head (ff_out, 0 if weight_tied) | 0 | 0.00% |
| Frozen | 7,509,303,296 | 92.45% |

## 2. 실제 데이터 로딩 (Train_toy100)

- Batch size: **4**
- Sequence length: **512** (max_length=512)
- Tasks in batch: `['smol-property_prediction-esol', 'reagent_prediction', 'chebi-20-text2mol', 'qm9_homo_lumo_gap']`

### Sample 0 상세
- Task: `smol-property_prediction-esol`
- Prompt length: **76** tokens
- Answer length: **12** tokens
- Padding length: **424** tokens (EOS, id=128001)
- Total: 76 (prompt) + 12 (answer) + 424 (pad) = 512

#### Prompt 전체 (76 tokens)

```
  Pos | Token ID |    Label | Decoded              | Region
───────────────────────────────────────────────────────────────
    0 |   126080 |     -100 | <|startoftext|>      | prompt
    1 |   126346 |     -100 | <|start_header_id…   | prompt
    2 |    18621 |     -100 | system               | prompt
    3 |   126347 |     -100 | <|end_header_id|>    | prompt
    4 |      198 |     -100 | \n                   | prompt
    5 |      198 |     -100 | \n                   | prompt
    6 |     2496 |     -100 | You                  | prompt
    7 |      449 |     -100 |  are                 | prompt
    8 |      259 |     -100 |  a                   | prompt
    9 |     9031 |     -100 |  helpful             | prompt
   10 |    16841 |     -100 |  assistant           | prompt
   11 |      352 |     -100 |  for                 | prompt
   12 |    17473 |     -100 |  molecular           | prompt
   13 |    25241 |     -100 |  chemistry           | prompt
   14 |       11 |     -100 | ,                    | prompt
   15 |      297 |     -100 |  to                  | prompt
   16 |     3265 |     -100 |  address             | prompt
   17 |     9947 |     -100 |  tasks               | prompt
   18 |     2524 |     -100 |  including           | prompt
   19 |    17473 |     -100 |  molecular           | prompt
   20 |     3809 |     -100 |  property            | prompt
   21 |    18188 |     -100 |  classification      | prompt
   22 |       11 |     -100 | ,                    | prompt
   23 |    17473 |     -100 |  molecular           | prompt
   24 |     3809 |     -100 |  property            | prompt
   25 |    20109 |     -100 |  regression          | prompt
   26 |       11 |     -100 | ,                    | prompt
   27 |     9977 |     -100 |  chemical            | prompt
   28 |    11226 |     -100 |  reaction            | prompt
   29 |    19185 |     -100 |  prediction          | prompt
   30 |       11 |     -100 | ,                    | prompt
   31 |    31051 |     -100 |  molecule            | prompt
   32 |    36636 |     -100 |  caption             | prompt
   33 |      283 |     -100 | ing                  | prompt
   34 |       11 |     -100 | ,                    | prompt
   35 |    31051 |     -100 |  molecule            | prompt
   36 |     9611 |     -100 |  generation          | prompt
   37 |       13 |     -100 | .                    | prompt
   38 |   126348 |     -100 | <|eot_id|>           | prompt
   39 |   126346 |     -100 | <|start_header_id…   | prompt
   40 |     3840 |     -100 | user                 | prompt
   41 |   126347 |     -100 | <|end_header_id|>    | prompt
   42 |      198 |     -100 | \n                   | prompt
   43 |      198 |     -100 | \n                   | prompt
   44 |     7584 |     -100 | Can                  | prompt
   45 |      362 |     -100 |  you                 | prompt
   46 |     6935 |     -100 |  predict             | prompt
   47 |      268 |     -100 |  the                 | prompt
   48 |     2599 |     -100 |  water               | prompt
   49 |     2843 |     -100 |  log                 | prompt
   50 |    98544 |     -100 |  solubility          | prompt
   51 |      300 |     -100 |  of                  | prompt
   52 |      220 |     -100 |                      | prompt
   53 |   126380 |     -100 | <SELFIES>            | prompt
   54 |      220 |     -100 |                      | prompt
   55 |   128074 |     -100 | [C]                  | prompt
   56 |   128074 |     -100 | [C]                  | prompt
   57 |   128074 |     -100 | [C]                  | prompt
   58 |   128369 |     -100 | [Branch1]            | prompt
   59 |   128074 |     -100 | [C]                  | prompt
   60 |   128074 |     -100 | [C]                  | prompt
   61 |   128369 |     -100 | [Branch1]            | prompt
   62 |   128074 |     -100 | [C]                  | prompt
   63 |   128074 |     -100 | [C]                  | prompt
   64 |   128074 |     -100 | [C]                  | prompt
   65 |   128837 |     -100 | [O]                  | prompt
   66 |      220 |     -100 |                      | prompt
   67 |   126381 |     -100 | </SELFIES>           | prompt
   68 |       30 |     -100 | ?                    | prompt
   69 |   126348 |     -100 | <|eot_id|>           | prompt
   70 |   126346 |     -100 | <|start_header_id…   | prompt
   71 |      598 |     -100 | ass                  | prompt
   72 |    10450 |     -100 | istant               | prompt
   73 |   126347 |     -100 | <|end_header_id|>    | prompt
   74 |      198 |     -100 | \n                   | prompt
   75 |      198 |     -100 | \n                   | prompt
```

#### Answer 전체 (12 tokens)

```
  Pos | Token ID |    Label | Decoded              | Region
───────────────────────────────────────────────────────────────
   76 |   126351 |   126351 | <FLOAT>              | answer  OK
   77 |      220 |      220 |                      | answer  OK
   78 |   126371 |   126371 | <|-|>                | answer  OK
   79 |   126361 |   126361 | <|1|>                | answer  OK
   80 |   126372 |   126372 | <|.|>                | answer  OK
   81 |   126360 |   126360 | <|0|>                | answer  OK
   82 |   126364 |   126364 | <|4|>                | answer  OK
   83 |   126360 |   126360 | <|0|>                | answer  OK
   84 |   126360 |   126360 | <|0|>                | answer  OK
   85 |      220 |      220 |                      | answer  OK
   86 |   126352 |   126352 | </FLOAT>             | answer  OK
   87 |   126348 |   126348 | <|eot_id|>           | answer  OK
```

#### Padding (첫 5 / 424 tokens)

```
  Pos | Token ID |    Label | Decoded              | Region
───────────────────────────────────────────────────────────────
   88 |   126081 |     -100 | <|endoftext|>        | padding (EOS)
   89 |   126081 |     -100 | <|endoftext|>        | padding (EOS)
   90 |   126081 |     -100 | <|endoftext|>        | padding (EOS)
   91 |   126081 |     -100 | <|endoftext|>        | padding (EOS)
   92 |   126081 |     -100 | <|endoftext|>        | padding (EOS)
  ... (419 more padding tokens, all id=128001)
```

## 3. Forward Process — `make_noisy()`

LLaDA Masked Diffusion: `t ~ U(0,1)` → `p_mask = (1-eps)*t + eps` → answer 토큰을 확률 p_mask로 MASK 교체

### Masking 결과
| Sample | p_mask | Answer 길이 | Masked 수 | Mask 비율 |
|--------|--------|------------|----------|-----------|
| 0 | 0.6133 | 12 | 4 | 33.3% |
| 1 | 0.0110 | 20 | 1 | 5.0% |
| 2 | 0.3990 | 30 | 12 | 40.0% |
| 3 | 0.0413 | 12 | 1 | 8.3% |

### Sample 0 마스킹 시각화 (answer 영역, 첫 40 tokens)

```
Position  : 원본 ID → Noisy ID  [MASK?]  Decoded
──────────────────────────────────────────────────────────────────────
  [  76] : 126351 → 126351           '<FLOAT>'
  [  77] :    220 → 126336  ██ MASK  ' '
  [  78] : 126371 → 126371           '<|-|>'
  [  79] : 126361 → 126361           '<|1|>'
  [  80] : 126372 → 126336  ██ MASK  '<|.|>'
  [  81] : 126360 → 126336  ██ MASK  '<|0|>'
  [  82] : 126364 → 126364           '<|4|>'
  [  83] : 126360 → 126360           '<|0|>'
  [  84] : 126360 → 126336  ██ MASK  '<|0|>'
  [  85] :    220 →    220           ' '
  [  86] : 126352 → 126352           '</FLOAT>'
  [  87] : 126348 → 126348           '<|eot_id|>'
```

- Prompt 영역 보존: **OK** (noisy_ids[:prompt_len] == input_ids[:prompt_len])
- MASK token ID: **126336** (`<|mdm_mask|>`)

## 4. Model Forward Pass

- Input: `noisy_ids` [4, 512]
- Output: `logits` [4, 512, 129325] (B, L, Vocab=129325)

## 5. Loss 계산 — `MaskedDiffusionLoss.forward()`

공식: `loss = Σ [ CE(logit, target) / p_mask / answer_length ] / batch_size`

### 계산 결과
| 항목 | 값 |
|------|-----|
| **Loss** | **92.011215** |
| Answer length mean | 18.50 |
| Loss is finite | YES |
| Loss is positive | YES |

### Sample 0 — 전체 시퀀스 Prediction & Loss (p_mask=0.6133, ans_len=12)

범례: `Region` = P(prompt), A(answer-보존), **M**(answer-MASKED), pad(패딩)
Masked 위치만 loss에 기여. 나머지는 `—`.

```
  Pos | Region |    정답ID | 정답Token          |    예측ID | 예측Token          |     CE Loss |     /p_mask |    /ans_len
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    0 |      P |  126080 | <|startoftext|>  |       — | —                |           — |           — |           —
    1 |      P |  126346 | <|start_header_… |       — | —                |           — |           — |           —
    2 |      P |   18621 | system           |       — | —                |           — |           — |           —
    3 |      P |  126347 | <|end_header_id… |       — | —                |           — |           — |           —
    4 |      P |     198 | \n               |       — | —                |           — |           — |           —
    5 |      P |     198 | \n               |       — | —                |           — |           — |           —
    6 |      P |    2496 | You              |       — | —                |           — |           — |           —
    7 |      P |     449 |  are             |       — | —                |           — |           — |           —
    8 |      P |     259 |  a               |       — | —                |           — |           — |           —
    9 |      P |    9031 |  helpful         |       — | —                |           — |           — |           —
   10 |      P |   16841 |  assistant       |       — | —                |           — |           — |           —
   11 |      P |     352 |  for             |       — | —                |           — |           — |           —
   12 |      P |   17473 |  molecular       |       — | —                |           — |           — |           —
   13 |      P |   25241 |  chemistry       |       — | —                |           — |           — |           —
   14 |      P |      11 | ,                |       — | —                |           — |           — |           —
   15 |      P |     297 |  to              |       — | —                |           — |           — |           —
   16 |      P |    3265 |  address         |       — | —                |           — |           — |           —
   17 |      P |    9947 |  tasks           |       — | —                |           — |           — |           —
   18 |      P |    2524 |  including       |       — | —                |           — |           — |           —
   19 |      P |   17473 |  molecular       |       — | —                |           — |           — |           —
   20 |      P |    3809 |  property        |       — | —                |           — |           — |           —
   21 |      P |   18188 |  classification  |       — | —                |           — |           — |           —
   22 |      P |      11 | ,                |       — | —                |           — |           — |           —
   23 |      P |   17473 |  molecular       |       — | —                |           — |           — |           —
   24 |      P |    3809 |  property        |       — | —                |           — |           — |           —
   25 |      P |   20109 |  regression      |       — | —                |           — |           — |           —
   26 |      P |      11 | ,                |       — | —                |           — |           — |           —
   27 |      P |    9977 |  chemical        |       — | —                |           — |           — |           —
   28 |      P |   11226 |  reaction        |       — | —                |           — |           — |           —
   29 |      P |   19185 |  prediction      |       — | —                |           — |           — |           —
   30 |      P |      11 | ,                |       — | —                |           — |           — |           —
   31 |      P |   31051 |  molecule        |       — | —                |           — |           — |           —
   32 |      P |   36636 |  caption         |       — | —                |           — |           — |           —
   33 |      P |     283 | ing              |       — | —                |           — |           — |           —
   34 |      P |      11 | ,                |       — | —                |           — |           — |           —
   35 |      P |   31051 |  molecule        |       — | —                |           — |           — |           —
   36 |      P |    9611 |  generation      |       — | —                |           — |           — |           —
   37 |      P |      13 | .                |       — | —                |           — |           — |           —
   38 |      P |  126348 | <|eot_id|>       |       — | —                |           — |           — |           —
   39 |      P |  126346 | <|start_header_… |       — | —                |           — |           — |           —
   40 |      P |    3840 | user             |       — | —                |           — |           — |           —
   41 |      P |  126347 | <|end_header_id… |       — | —                |           — |           — |           —
   42 |      P |     198 | \n               |       — | —                |           — |           — |           —
   43 |      P |     198 | \n               |       — | —                |           — |           — |           —
   44 |      P |    7584 | Can              |       — | —                |           — |           — |           —
   45 |      P |     362 |  you             |       — | —                |           — |           — |           —
   46 |      P |    6935 |  predict         |       — | —                |           — |           — |           —
   47 |      P |     268 |  the             |       — | —                |           — |           — |           —
   48 |      P |    2599 |  water           |       — | —                |           — |           — |           —
   49 |      P |    2843 |  log             |       — | —                |           — |           — |           —
   50 |      P |   98544 |  solubility      |       — | —                |           — |           — |           —
   51 |      P |     300 |  of              |       — | —                |           — |           — |           —
   52 |      P |     220 |                  |       — | —                |           — |           — |           —
   53 |      P |  126380 | <SELFIES>        |       — | —                |           — |           — |           —
   54 |      P |     220 |                  |       — | —                |           — |           — |           —
   55 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   56 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   57 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   58 |      P |  128369 | [Branch1]        |       — | —                |           — |           — |           —
   59 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   60 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   61 |      P |  128369 | [Branch1]        |       — | —                |           — |           — |           —
   62 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   63 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   64 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   65 |      P |  128837 | [O]              |       — | —                |           — |           — |           —
   66 |      P |     220 |                  |       — | —                |           — |           — |           —
   67 |      P |  126381 | </SELFIES>       |       — | —                |           — |           — |           —
   68 |      P |      30 | ?                |       — | —                |           — |           — |           —
   69 |      P |  126348 | <|eot_id|>       |       — | —                |           — |           — |           —
   70 |      P |  126346 | <|start_header_… |       — | —                |           — |           — |           —
   71 |      P |     598 | ass              |       — | —                |           — |           — |           —
   72 |      P |   10450 | istant           |       — | —                |           — |           — |           —
   73 |      P |  126347 | <|end_header_id… |       — | —                |           — |           — |           —
   74 |      P |     198 | \n               |       — | —                |           — |           — |           —
   75 |      P |     198 | \n               |       — | —                |           — |           — |           —
   76 |      A |  126351 | <FLOAT>          |  126081 | <|endoftext|>    |           — |           — |           —
   77 |  **M** |     220 |                  |  126081 | <|endoftext|>    |     30.8125 |     50.2367 |    4.186389
   78 |      A |  126371 | <|-|>            |  126081 | <|endoftext|>    |           — |           — |           —
   79 |      A |  126361 | <|1|>            |  126081 | <|endoftext|>    |           — |           — |           —
   80 |  **M** |  126372 | <|.|>            |  126081 | <|endoftext|>    |     51.8750 |     84.5769 |    7.048078
   81 |  **M** |  126360 | <|0|>            |  126081 | <|endoftext|>    |     53.8125 |     87.7358 |    7.311320
   82 |      A |  126364 | <|4|>            |  126081 | <|endoftext|>    |           — |           — |           —
   83 |      A |  126360 | <|0|>            |  126081 | <|endoftext|>    |           — |           — |           —
   84 |  **M** |  126360 | <|0|>            |  126081 | <|endoftext|>    |     52.3750 |     85.3921 |    7.116011
   85 |      A |     220 |                  |  126081 | <|endoftext|>    |           — |           — |           —
   86 |      A |  126352 | </FLOAT>         |  126081 | <|endoftext|>    |           — |           — |           —
   87 |      A |  126348 | <|eot_id|>       |  126081 | <|endoftext|>    |           — |           — |           —
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      |        |         | TOTAL            |         |                  |             |             |   25.661798
```

- Padding (424 tokens, all EOS id=128001) 생략
- **Sample 0 기여도 합계**: 25.661798
- **최종 loss** = (Σ all samples) / batch_size = 92.011215

> 학습 전이므로 예측 Token이 정답과 무관한 것이 정상. 학습이 진행되면 정답Token과 예측Token이 일치하기 시작.

## 6. Backward + Weight Update

### Gradient 통계 (backward 후)

| Layer | Grad Norm | Grad Mean | Grad Max |
|-------|-----------|-----------|----------|
| Embedding (orig vocab) | 1.842799e+03 | -2.182640e-05 | 2.330000e+02 |
| Embedding (new vocab) | 1.069631e+03 | -3.038730e-04 | 1.530000e+02 |
| Head (tied to wte) (orig vocab) | 1.842799e+03 | -2.182640e-05 | 2.330000e+02 |
| Head (tied to wte) (new vocab) | 1.069631e+03 | -3.038730e-04 | 1.530000e+02 |
| LoRA_A (`.transformer.blocks.0.q_proj.lora_A.default.weight`) | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| LoRA_B (`.transformer.blocks.0.q_proj.lora_B.default.weight`) | 6.060567e+00 | -3.272399e-05 | 2.002731e-01 |

### Weight 변화량 (optimizer.step() 후)

```
Layer                                    |    Before Norm |     After Norm |     Delta Norm |   Δ/Before
────────────────────────────────────────────────────────────────────────────────────────────────────
Embedding (orig, idx < 126349)           |     274.079102 |     274.079102 |     0.01528922 |   0.0056%
Embedding (new,  idx >= 126349)          |      45.487965 |      45.487968 |     0.00734499 |   0.0161%
Head/wte (tied) (orig, idx < 126349)     |     274.079102 |     274.079102 |     0.01528922 |   0.0056%
Head/wte (tied) (new,  idx >= 126349)    |      45.487965 |      45.487968 |     0.00734499 |   0.0161%
LoRA_A (first layer)                     |       4.622340 |       4.621142 |     0.00115548 |   0.0250%
LoRA_B (first layer)                     |       0.000000 |       1.247851 |     1.24785149 | 124785149097442.6250%
```

## 7. Embedding & Head 상세 — Original vs New Vocab

> Original vocab (idx 0 ~ 126348): LLaDA 기본 토큰
> New vocab (idx 126349 ~): 프로젝트 추가 토큰 (BOOL, FLOAT, SELFIES, ...)

### Input Embedding (wte)

```
구분                        |         Mean |          Std |           Norm |         Δ Norm
─────────────────────────────────────────────────────────────────────────────────────
Orig (before)             | -2.118877e-05 | 1.322869e-02 |     274.079102 | —
Orig (after)              | -2.118865e-05 | 1.322869e-02 |     274.079102 | 0.01528922
New  (before)             | -1.791623e-05 | 1.303103e-02 |      45.487965 | —
New  (after)              | -1.791579e-05 | 1.303103e-02 |      45.487968 | 0.00734499
```

### 특정 토큰별 Embedding 변화

```
Token                     |      ID |  Vocab |     Emb Δ Norm |    Head Δ Norm
───────────────────────────────────────────────────────────────────────────
orig — 'the'              |    1614 |   orig |     0.00000192 |     0.00000192
orig — 'molecule'         |      76 |   orig |     0.00000325 |     0.00000325
new  — '<BOOLEAN>'        |  126349 |    new |     0.00000000 |     0.00000000
new  — '<SELFIES>'        |  126380 |    new |     0.00149502 |     0.00149502
new  — '<FLOAT>'          |  126351 |    new |     0.00149534 |     0.00149534
new  — '<mol>'            |  126359 |    new |     0.00000000 |     0.00000000
```

## 8. LoRA Weight 변화

```
Layer (last 60 chars)                                          |    Grad Norm |  Δ Weight Norm
───────────────────────────────────────────────────────────────────────────────────────────────
odel.model.transformer.blocks.0.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.0.q_proj.lora_B.default.weight   | 2.843082e-03 | —
odel.model.transformer.blocks.0.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.0.k_proj.lora_B.default.weight   | 2.157944e-03 | —
odel.model.transformer.blocks.0.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.0.v_proj.lora_B.default.weight   | 1.819731e-02 | —
del.model.transformer.blocks.0.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.0.up_proj.lora_B.default.weight   | 4.844633e-03 | —
odel.model.transformer.blocks.1.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.1.q_proj.lora_B.default.weight   | 2.845712e-04 | —
odel.model.transformer.blocks.1.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.1.k_proj.lora_B.default.weight   | 3.251531e-04 | —
odel.model.transformer.blocks.1.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.1.v_proj.lora_B.default.weight   | 3.451256e-02 | —
del.model.transformer.blocks.1.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.1.up_proj.lora_B.default.weight   | 9.527080e-03 | —
odel.model.transformer.blocks.2.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.2.q_proj.lora_B.default.weight   | 5.278363e-04 | —
odel.model.transformer.blocks.2.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.2.k_proj.lora_B.default.weight   | 8.216352e-04 | —
odel.model.transformer.blocks.2.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.2.v_proj.lora_B.default.weight   | 1.997631e-02 | —
del.model.transformer.blocks.2.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.2.up_proj.lora_B.default.weight   | 1.531112e-02 | —
odel.model.transformer.blocks.3.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.3.q_proj.lora_B.default.weight   | 1.248033e-04 | —
odel.model.transformer.blocks.3.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.3.k_proj.lora_B.default.weight   | 1.296620e-04 | —
odel.model.transformer.blocks.3.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.3.v_proj.lora_B.default.weight   | 5.208375e-03 | —
del.model.transformer.blocks.3.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.3.up_proj.lora_B.default.weight   | 6.883623e-03 | —
odel.model.transformer.blocks.4.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.4.q_proj.lora_B.default.weight   | 2.551714e-04 | —
odel.model.transformer.blocks.4.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.4.k_proj.lora_B.default.weight   | 4.177570e-04 | —
odel.model.transformer.blocks.4.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.4.v_proj.lora_B.default.weight   | 6.094194e-03 | —
del.model.transformer.blocks.4.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.4.up_proj.lora_B.default.weight   | 5.881395e-03 | —
odel.model.transformer.blocks.5.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.5.q_proj.lora_B.default.weight   | 3.182384e-04 | —
odel.model.transformer.blocks.5.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.5.k_proj.lora_B.default.weight   | 3.658222e-04 | —
odel.model.transformer.blocks.5.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.5.v_proj.lora_B.default.weight   | 6.313999e-03 | —
del.model.transformer.blocks.5.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.5.up_proj.lora_B.default.weight   | 4.275882e-03 | —
odel.model.transformer.blocks.6.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.6.q_proj.lora_B.default.weight   | 1.497768e-04 | —
odel.model.transformer.blocks.6.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.6.k_proj.lora_B.default.weight   | 2.093824e-04 | —
odel.model.transformer.blocks.6.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.6.v_proj.lora_B.default.weight   | 7.725629e-03 | —
del.model.transformer.blocks.6.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.6.up_proj.lora_B.default.weight   | 4.651194e-03 | —
odel.model.transformer.blocks.7.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.7.q_proj.lora_B.default.weight   | 1.515605e-04 | —
odel.model.transformer.blocks.7.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.7.k_proj.lora_B.default.weight   | 1.587243e-04 | —
odel.model.transformer.blocks.7.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.7.v_proj.lora_B.default.weight   | 6.599010e-03 | —
del.model.transformer.blocks.7.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.7.up_proj.lora_B.default.weight   | 3.637719e-03 | —
odel.model.transformer.blocks.8.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.8.q_proj.lora_B.default.weight   | 1.875851e-04 | —
odel.model.transformer.blocks.8.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.8.k_proj.lora_B.default.weight   | 1.857006e-04 | —
odel.model.transformer.blocks.8.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.8.v_proj.lora_B.default.weight   | 7.393851e-03 | —
del.model.transformer.blocks.8.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.8.up_proj.lora_B.default.weight   | 3.399561e-03 | —
odel.model.transformer.blocks.9.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.9.q_proj.lora_B.default.weight   | 1.286443e-04 | —
odel.model.transformer.blocks.9.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.9.k_proj.lora_B.default.weight   | 1.337423e-04 | —
odel.model.transformer.blocks.9.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.9.v_proj.lora_B.default.weight   | 7.113528e-03 | —
del.model.transformer.blocks.9.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.9.up_proj.lora_B.default.weight   | 2.448487e-03 | —
del.model.transformer.blocks.10.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.10.q_proj.lora_B.default.weight   | 1.035828e-04 | —
del.model.transformer.blocks.10.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.10.k_proj.lora_B.default.weight   | 7.996411e-05 | —
del.model.transformer.blocks.10.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.10.v_proj.lora_B.default.weight   | 6.414422e-03 | —
el.model.transformer.blocks.10.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.10.up_proj.lora_B.default.weight   | 1.994828e-03 | —
del.model.transformer.blocks.11.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.11.q_proj.lora_B.default.weight   | 2.036014e-04 | —
del.model.transformer.blocks.11.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.11.k_proj.lora_B.default.weight   | 1.557309e-04 | —
del.model.transformer.blocks.11.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.11.v_proj.lora_B.default.weight   | 7.473971e-03 | —
el.model.transformer.blocks.11.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.11.up_proj.lora_B.default.weight   | 2.786549e-03 | —
del.model.transformer.blocks.12.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.12.q_proj.lora_B.default.weight   | 2.555155e-04 | —
del.model.transformer.blocks.12.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.12.k_proj.lora_B.default.weight   | 2.405578e-04 | —
del.model.transformer.blocks.12.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.12.v_proj.lora_B.default.weight   | 7.694049e-03 | —
el.model.transformer.blocks.12.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.12.up_proj.lora_B.default.weight   | 2.520162e-03 | —
del.model.transformer.blocks.13.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.13.q_proj.lora_B.default.weight   | 2.538413e-04 | —
del.model.transformer.blocks.13.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.13.k_proj.lora_B.default.weight   | 1.928777e-04 | —
del.model.transformer.blocks.13.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.13.v_proj.lora_B.default.weight   | 9.929477e-03 | —
el.model.transformer.blocks.13.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.13.up_proj.lora_B.default.weight   | 3.132602e-03 | —
del.model.transformer.blocks.14.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.14.q_proj.lora_B.default.weight   | 6.138839e-04 | —
del.model.transformer.blocks.14.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.14.k_proj.lora_B.default.weight   | 5.213243e-04 | —
del.model.transformer.blocks.14.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.14.v_proj.lora_B.default.weight   | 7.648591e-03 | —
el.model.transformer.blocks.14.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.14.up_proj.lora_B.default.weight   | 3.177757e-03 | —
del.model.transformer.blocks.15.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.15.q_proj.lora_B.default.weight   | 1.267438e-03 | —
del.model.transformer.blocks.15.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.15.k_proj.lora_B.default.weight   | 1.058281e-03 | —
del.model.transformer.blocks.15.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.15.v_proj.lora_B.default.weight   | 7.272935e-03 | —
el.model.transformer.blocks.15.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.15.up_proj.lora_B.default.weight   | 2.466960e-03 | —
del.model.transformer.blocks.16.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.16.q_proj.lora_B.default.weight   | 1.404028e-03 | —
del.model.transformer.blocks.16.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.16.k_proj.lora_B.default.weight   | 9.357333e-04 | —
del.model.transformer.blocks.16.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.16.v_proj.lora_B.default.weight   | 5.471078e-03 | —
el.model.transformer.blocks.16.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.16.up_proj.lora_B.default.weight   | 2.082182e-03 | —
del.model.transformer.blocks.17.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.17.q_proj.lora_B.default.weight   | 1.573818e-03 | —
del.model.transformer.blocks.17.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.17.k_proj.lora_B.default.weight   | 1.094069e-03 | —
del.model.transformer.blocks.17.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.17.v_proj.lora_B.default.weight   | 4.438640e-03 | —
el.model.transformer.blocks.17.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.17.up_proj.lora_B.default.weight   | 1.852948e-03 | —
del.model.transformer.blocks.18.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.18.q_proj.lora_B.default.weight   | 8.880341e-04 | —
del.model.transformer.blocks.18.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.18.k_proj.lora_B.default.weight   | 6.302053e-04 | —
del.model.transformer.blocks.18.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.18.v_proj.lora_B.default.weight   | 4.471664e-03 | —
el.model.transformer.blocks.18.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.18.up_proj.lora_B.default.weight   | 1.856979e-03 | —
del.model.transformer.blocks.19.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.19.q_proj.lora_B.default.weight   | 1.177805e-03 | —
del.model.transformer.blocks.19.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.19.k_proj.lora_B.default.weight   | 8.257242e-04 | —
del.model.transformer.blocks.19.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.19.v_proj.lora_B.default.weight   | 4.767774e-03 | —
el.model.transformer.blocks.19.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.19.up_proj.lora_B.default.weight   | 2.220907e-03 | —
del.model.transformer.blocks.20.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.20.q_proj.lora_B.default.weight   | 1.809399e-03 | —
del.model.transformer.blocks.20.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.20.k_proj.lora_B.default.weight   | 1.267418e-03 | —
del.model.transformer.blocks.20.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.20.v_proj.lora_B.default.weight   | 4.859246e-03 | —
el.model.transformer.blocks.20.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.20.up_proj.lora_B.default.weight   | 2.123009e-03 | —
del.model.transformer.blocks.21.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.21.q_proj.lora_B.default.weight   | 1.001741e-03 | —
del.model.transformer.blocks.21.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.21.k_proj.lora_B.default.weight   | 8.405473e-04 | —
del.model.transformer.blocks.21.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.21.v_proj.lora_B.default.weight   | 4.739574e-03 | —
el.model.transformer.blocks.21.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.21.up_proj.lora_B.default.weight   | 1.960562e-03 | —
del.model.transformer.blocks.22.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.22.q_proj.lora_B.default.weight   | 1.143369e-03 | —
del.model.transformer.blocks.22.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.22.k_proj.lora_B.default.weight   | 8.007419e-04 | —
del.model.transformer.blocks.22.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.22.v_proj.lora_B.default.weight   | 2.945042e-03 | —
el.model.transformer.blocks.22.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.22.up_proj.lora_B.default.weight   | 2.000297e-03 | —
del.model.transformer.blocks.23.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.23.q_proj.lora_B.default.weight   | 1.592387e-03 | —
del.model.transformer.blocks.23.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.23.k_proj.lora_B.default.weight   | 9.040792e-04 | —
del.model.transformer.blocks.23.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.23.v_proj.lora_B.default.weight   | 3.347012e-03 | —
el.model.transformer.blocks.23.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.23.up_proj.lora_B.default.weight   | 3.402858e-03 | —
del.model.transformer.blocks.24.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.24.q_proj.lora_B.default.weight   | 1.190436e-03 | —
del.model.transformer.blocks.24.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.24.k_proj.lora_B.default.weight   | 8.263155e-04 | —
del.model.transformer.blocks.24.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.24.v_proj.lora_B.default.weight   | 3.426549e-03 | —
el.model.transformer.blocks.24.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.24.up_proj.lora_B.default.weight   | 1.828419e-03 | —
del.model.transformer.blocks.25.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.25.q_proj.lora_B.default.weight   | 6.336702e-04 | —
del.model.transformer.blocks.25.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.25.k_proj.lora_B.default.weight   | 5.127744e-04 | —
del.model.transformer.blocks.25.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.25.v_proj.lora_B.default.weight   | 2.802582e-03 | —
el.model.transformer.blocks.25.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.25.up_proj.lora_B.default.weight   | 1.465528e-03 | —
del.model.transformer.blocks.26.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.26.q_proj.lora_B.default.weight   | 9.522123e-04 | —
del.model.transformer.blocks.26.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.26.k_proj.lora_B.default.weight   | 8.782339e-04 | —
del.model.transformer.blocks.26.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.26.v_proj.lora_B.default.weight   | 2.274609e-03 | —
el.model.transformer.blocks.26.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.26.up_proj.lora_B.default.weight   | 1.611298e-03 | —
del.model.transformer.blocks.27.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.27.q_proj.lora_B.default.weight   | 9.691210e-04 | —
del.model.transformer.blocks.27.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.27.k_proj.lora_B.default.weight   | 6.759044e-04 | —
del.model.transformer.blocks.27.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.27.v_proj.lora_B.default.weight   | 2.440756e-03 | —
el.model.transformer.blocks.27.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.27.up_proj.lora_B.default.weight   | 2.055508e-03 | —
del.model.transformer.blocks.28.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.28.q_proj.lora_B.default.weight   | 1.362828e-03 | —
del.model.transformer.blocks.28.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.28.k_proj.lora_B.default.weight   | 9.810869e-04 | —
del.model.transformer.blocks.28.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.28.v_proj.lora_B.default.weight   | 1.991642e-03 | —
el.model.transformer.blocks.28.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.28.up_proj.lora_B.default.weight   | 1.434262e-03 | —
del.model.transformer.blocks.29.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.29.q_proj.lora_B.default.weight   | 1.550448e-03 | —
del.model.transformer.blocks.29.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.29.k_proj.lora_B.default.weight   | 1.320723e-03 | —
del.model.transformer.blocks.29.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.29.v_proj.lora_B.default.weight   | 2.150793e-03 | —
el.model.transformer.blocks.29.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.29.up_proj.lora_B.default.weight   | 1.414891e-03 | —
del.model.transformer.blocks.30.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.30.q_proj.lora_B.default.weight   | 9.606101e-04 | —
del.model.transformer.blocks.30.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.30.k_proj.lora_B.default.weight   | 7.552379e-04 | —
del.model.transformer.blocks.30.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.30.v_proj.lora_B.default.weight   | 1.841277e-03 | —
el.model.transformer.blocks.30.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.30.up_proj.lora_B.default.weight   | 1.616992e-03 | —
del.model.transformer.blocks.31.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.31.q_proj.lora_B.default.weight   | 1.276743e-03 | —
del.model.transformer.blocks.31.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.31.k_proj.lora_B.default.weight   | 8.998436e-04 | —
del.model.transformer.blocks.31.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.31.v_proj.lora_B.default.weight   | 1.483509e-03 | —
el.model.transformer.blocks.31.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.31.up_proj.lora_B.default.weight   | 3.460666e-03 | —
```

### LoRA_A (첫 번째 layer) 변화 상세
- Layer: `llada._model.base_model.model.model.transformer.blocks.0.q_proj.lora_A.default.weight`
- Shape: [64, 4096]
- Before norm: 4.622340
- After norm:  4.621142
- Delta norm:  0.00115548
- Delta max:   0.00000391

### LoRA_B (첫 번째 layer) 변화 상세
- Layer: `llada._model.base_model.model.model.transformer.blocks.0.q_proj.lora_B.default.weight`
- Shape: [4096, 64]
- Before norm: 0.000000
- After norm:  1.247851
- Delta norm:  1.24785149
- Delta max:   0.00249973
- LoRA_B는 초기값이 0 → 첫 step에서 0이 아닌 값으로 변화 (정상)

## 9. GPU 메모리 사용량

| 항목 | GB |
|------|-----|
| Allocated | 21.16 |
| Reserved | 39.17 |
| Total GPU | 47.59 |
| Free | 8.42 |

## 10. Step-wise Denoising Logging — 4가지 전략 조합

> `generate_with_logging()`을 remasking × sampling 4가지 조합으로 실행하여 denoising 과정을 시각화.
> 학습 1 step 직후의 모델 — 무작위에 가까운 생성이 정상.
> **Prompt A**: 분자 Task (batch에서 추출). **Prompt B**: 일반 질문 (base LLaDA 능력 확인).
>
> **확률 표기**: `N.N%` = 이번 step에서 새로 확정된 토큰의 softmax 확률, `prev` = 이전 step에서 이미 확정 (LLaDA는 한 번 unmask된 토큰을 re-mask하지 않음).

- gen_length=48, steps=8, block_length=16 (semi_ar용)

---

### Prompt A: 분자 Task (batch sample)

- Prompt: `<|startoftext|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant for molecular...`
- Prompt length: 76 tokens
- Target: `<FLOAT> <|-|><|1|><|.|><|0|><|4|><|0|><|0|> </FLOAT><|eot_id|>`

#### 10.1. [A] low_confidence + standard

| Config | Value |
|--------|-------|
| Remasking | low_confidence |
| Sampling | standard |
| Steps (requested) | 8 |
| gen_length | 48 |

- Actual steps: **8** (requested: 8)
- Actual gen_length: **48**

| Step | Unmasked | Total | Unmasked % |
|------|----------|-------|------------|
| 1/8 | 6 | 48 | 12.5% |
| 2/8 | 12 | 48 | 25.0% |
| 3/8 | 18 | 48 | 37.5% |
| 4/8 | 24 | 48 | 50.0% |
| 5/8 | 30 | 48 | 62.5% |
| 6/8 | 36 | 48 | 75.0% |
| 7/8 | 42 | 48 | 87.5% |
| 8/8 | 48 | 48 | 100.0% |

**Step 1/8** — Unmasked: 6/48 (12.5%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] [MASK]                      [MASK]                      [MASK]                   
  [  3-  5] [MASK]                      [MASK]                      [MASK]                   
  [  6-  8] [MASK]                      [MASK]                      [MASK]                   
  [  9- 11] [MASK]                      [MASK]                      [MASK]                   
  [ 12- 14] [MASK]                      [MASK]                      [MASK]                   
  [ 15- 17] [MASK]                      [MASK]                      [MASK]                   
  [ 18- 20] [MASK]                      [MASK]                      [MASK]                   
  [ 21- 23] [MASK]                      [MASK]                      <|endoftext|>        4.8%
  [ 24- 26] [MASK]                      [MASK]                      [MASK]                   
  [ 27- 29] [MASK]                      [MASK]                      [MASK]                   
  [ 30- 32] [MASK]                      [MASK]                      [MASK]                   
  [ 33- 35] <|endoftext|>        4.9%   <|endoftext|>        4.9%   [MASK]                   
  [ 36- 38] [MASK]                      [MASK]                      [MASK]                   
  [ 39- 41] [MASK]                      <|endoftext|>        4.9%   [MASK]                   
  [ 42- 44] [MASK]                      <|endoftext|>        4.9%   [MASK]                   
  [ 45- 47] [MASK]                      <|endoftext|>        4.9%   [MASK]                   
```

**Step 5/8** — Unmasked: 30/48 (62.5%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   [MASK]                      <|endoftext|>        prev
  [  3-  5] <|endoftext|>        5.1%   [MASK]                      [MASK]                   
  [  6-  8] <|endoftext|>        prev   [MASK]                      [MASK]                   
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   [MASK]                   
  [ 12- 14] <|endoftext|>        5.1%   [MASK]                      <|endoftext|>        5.1%
  [ 15- 17] <|endoftext|>        prev   <|endoftext|>        5.1%   <|endoftext|>        5.1%
  [ 18- 20] [MASK]                      <|endoftext|>        prev   [MASK]                   
  [ 21- 23] <|endoftext|>        prev   [MASK]                      <|endoftext|>        prev
  [ 24- 26] <|endoftext|>        prev   <|endoftext|>        prev   [MASK]                   
  [ 27- 29] [MASK]                      <|endoftext|>        prev   <|endoftext|>        prev
  [ 30- 32] <|endoftext|>        5.2%   [MASK]                      <|endoftext|>        prev
  [ 33- 35] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 36- 38] [MASK]                      <|endoftext|>        prev   <|endoftext|>        prev
  [ 39- 41] [MASK]                      <|endoftext|>        prev   <|endoftext|>        prev
  [ 42- 44] [MASK]                      <|endoftext|>        prev   <|endoftext|>        prev
  [ 45- 47] [MASK]                      <|endoftext|>        prev   [MASK]                   
```

**Step 8/8** — Unmasked: 48/48 (100.0%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   <|endoftext|>        6.3%   <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   <|endoftext|>        5.5%   <|endoftext|>        6.5%
  [  6-  8] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        6.9%
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 12- 14] <|endoftext|>        prev   <|endoftext|>        6.7%   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 18- 20] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 21- 23] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 24- 26] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 27- 29] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 30- 32] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 33- 35] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 36- 38] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 39- 41] <|endoftext|>        7.0%   <|endoftext|>        prev   <|endoftext|>        prev
  [ 42- 44] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 45- 47] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
```

**Generated**: `<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|e...`
**Target**: `<FLOAT> <|-|><|1|><|.|><|0|><|4|><|0|><|0|> </FLOAT><|eot_id|>`

#### 10.2. [A] low_confidence + semi_ar

| Config | Value |
|--------|-------|
| Remasking | low_confidence |
| Sampling | semi_ar |
| Steps (requested) | 8 |
| gen_length | 48 |
| block_length | 16 |

- Actual steps: **9** (requested: 8)
- Actual gen_length: **48**

| Step | Unmasked | Total | Unmasked % |
|------|----------|-------|------------|
| 1/9 | 6 | 48 | 12.5% |
| 2/9 | 11 | 48 | 22.9% |
| 3/9 | 16 | 48 | 33.3% |
| 4/9 | 22 | 48 | 45.8% |
| 5/9 | 27 | 48 | 56.2% |
| 6/9 | 32 | 48 | 66.7% |
| 7/9 | 38 | 48 | 79.2% |
| 8/9 | 43 | 48 | 89.6% |
| 9/9 | 48 | 48 | 100.0% |

**Step 1/9** — Unmasked: 6/48 (12.5%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        4.8%   [MASK]                      [MASK]                   
  [  3-  5] <|endoftext|>        4.7%   [MASK]                      [MASK]                   
  [  6-  8] <|endoftext|>        4.7%   [MASK]                      [MASK]                   
  [  9- 11] [MASK]                      [MASK]                      [MASK]                   
  [ 12- 14] [MASK]                      <|endoftext|>        4.8%   <|endoftext|>        4.8%
  [ 15- 17] <|endoftext|>        4.8%   [MASK]                      [MASK]                   
  [ 18- 20] [MASK]                      [MASK]                      [MASK]                   
  [ 21- 23] [MASK]                      [MASK]                      [MASK]                   
  [ 24- 26] [MASK]                      [MASK]                      [MASK]                   
  [ 27- 29] [MASK]                      [MASK]                      [MASK]                   
  [ 30- 32] [MASK]                      [MASK]                      [MASK]                   
  [ 33- 35] [MASK]                      [MASK]                      [MASK]                   
  [ 36- 38] [MASK]                      [MASK]                      [MASK]                   
  [ 39- 41] [MASK]                      [MASK]                      [MASK]                   
  [ 42- 44] [MASK]                      [MASK]                      [MASK]                   
  [ 45- 47] [MASK]                      [MASK]                      [MASK]                   
```

**Step 5/9** — Unmasked: 27/48 (56.2%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  6-  8] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 12- 14] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   <|endoftext|>        5.2%   <|endoftext|>        prev
  [ 18- 20] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        5.2%
  [ 21- 23] <|endoftext|>        5.2%   <|endoftext|>        5.2%   [MASK]                   
  [ 24- 26] [MASK]                      <|endoftext|>        prev   <|endoftext|>        prev
  [ 27- 29] [MASK]                      <|endoftext|>        prev   [MASK]                   
  [ 30- 32] <|endoftext|>        5.2%   [MASK]                      [MASK]                   
  [ 33- 35] [MASK]                      [MASK]                      [MASK]                   
  [ 36- 38] [MASK]                      [MASK]                      [MASK]                   
  [ 39- 41] [MASK]                      [MASK]                      [MASK]                   
  [ 42- 44] [MASK]                      [MASK]                      [MASK]                   
  [ 45- 47] [MASK]                      [MASK]                      [MASK]                   
```

**Step 9/9** — Unmasked: 48/48 (100.0%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  6-  8] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 12- 14] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 18- 20] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 21- 23] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 24- 26] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 27- 29] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 30- 32] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 33- 35] <|endoftext|>        prev   <|endoftext|>        7.1%   <|endoftext|>        prev
  [ 36- 38] <|endoftext|>        6.7%   <|endoftext|>        6.6%   <|endoftext|>        prev
  [ 39- 41] <|endoftext|>        prev   <|endoftext|>        6.6%   <|endoftext|>        6.9%
  [ 42- 44] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 45- 47] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
```

**Generated**: `<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|e...`
**Target**: `<FLOAT> <|-|><|1|><|.|><|0|><|4|><|0|><|0|> </FLOAT><|eot_id|>`

#### 10.3. [A] random + standard

| Config | Value |
|--------|-------|
| Remasking | random |
| Sampling | standard |
| Steps (requested) | 8 |
| gen_length | 48 |

- Actual steps: **8** (requested: 8)
- Actual gen_length: **48**

| Step | Unmasked | Total | Unmasked % |
|------|----------|-------|------------|
| 1/8 | 6 | 48 | 12.5% |
| 2/8 | 12 | 48 | 25.0% |
| 3/8 | 18 | 48 | 37.5% |
| 4/8 | 24 | 48 | 50.0% |
| 5/8 | 30 | 48 | 62.5% |
| 6/8 | 36 | 48 | 75.0% |
| 7/8 | 42 | 48 | 87.5% |
| 8/8 | 48 | 48 | 100.0% |

**Step 1/8** — Unmasked: 6/48 (12.5%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] [MASK]                      [MASK]                      [MASK]                   
  [  3-  5] [MASK]                      <|endoftext|>        4.6%   [MASK]                   
  [  6-  8] [MASK]                      [MASK]                      [MASK]                   
  [  9- 11] [MASK]                      <|endoftext|>        4.7%   [MASK]                   
  [ 12- 14] [MASK]                      [MASK]                      [MASK]                   
  [ 15- 17] <|endoftext|>        4.8%   [MASK]                      [MASK]                   
  [ 18- 20] [MASK]                      [MASK]                      <|endoftext|>        4.7%
  [ 21- 23] [MASK]                      [MASK]                      [MASK]                   
  [ 24- 26] [MASK]                      [MASK]                      [MASK]                   
  [ 27- 29] [MASK]                      [MASK]                      <|endoftext|>        4.7%
  [ 30- 32] [MASK]                      [MASK]                      [MASK]                   
  [ 33- 35] [MASK]                      [MASK]                      [MASK]                   
  [ 36- 38] [MASK]                      [MASK]                      [MASK]                   
  [ 39- 41] [MASK]                      [MASK]                      <|endoftext|>        4.8%
  [ 42- 44] [MASK]                      [MASK]                      [MASK]                   
  [ 45- 47] [MASK]                      [MASK]                      [MASK]                   
```

**Step 5/8** — Unmasked: 30/48 (62.5%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        5.1%   <|endoftext|>        prev   [MASK]                   
  [  3-  5] [MASK]                      <|endoftext|>        prev   <|endoftext|>        5.1%
  [  6-  8] <|endoftext|>        4.9%   <|endoftext|>        prev   <|endoftext|>        prev
  [  9- 11] [MASK]                      <|endoftext|>        prev   <|endoftext|>        prev
  [ 12- 14] [MASK]                      <|endoftext|>        4.9%   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   [MASK]                      <|endoftext|>        prev
  [ 18- 20] <|endoftext|>        prev   [MASK]                      <|endoftext|>        prev
  [ 21- 23] <|endoftext|>        prev   [MASK]                      [MASK]                   
  [ 24- 26] [MASK]                      <|endoftext|>        prev   <|endoftext|>        prev
  [ 27- 29] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 30- 32] [MASK]                      <|endoftext|>        prev   [MASK]                   
  [ 33- 35] <|endoftext|>        prev   [MASK]                      [MASK]                   
  [ 36- 38] [MASK]                      [MASK]                      <|endoftext|>        prev
  [ 39- 41] <|endoftext|>        prev   [MASK]                      <|endoftext|>        prev
  [ 42- 44] [MASK]                      [MASK]                      <|endoftext|>        5.0%
  [ 45- 47] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        5.1%
```

**Step 8/8** — Unmasked: 48/48 (100.0%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  6-  8] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 12- 14] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 18- 20] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 21- 23] <|endoftext|>        prev   <|endoftext|>        5.8%   <|endoftext|>        prev
  [ 24- 26] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 27- 29] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 30- 32] <|endoftext|>        5.6%   <|endoftext|>        prev   <|endoftext|>        prev
  [ 33- 35] <|endoftext|>        prev   <|endoftext|>        4.7%   <|endoftext|>        5.6%
  [ 36- 38] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 39- 41] <|endoftext|>        prev   <|endoftext|>        5.5%   <|endoftext|>        prev
  [ 42- 44] <|endoftext|>        prev   <|endoftext|>        5.6%   <|endoftext|>        prev
  [ 45- 47] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
```

**Generated**: `<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|e...`
**Target**: `<FLOAT> <|-|><|1|><|.|><|0|><|4|><|0|><|0|> </FLOAT><|eot_id|>`

#### 10.4. [A] random + semi_ar

| Config | Value |
|--------|-------|
| Remasking | random |
| Sampling | semi_ar |
| Steps (requested) | 8 |
| gen_length | 48 |
| block_length | 16 |

- Actual steps: **9** (requested: 8)
- Actual gen_length: **48**

| Step | Unmasked | Total | Unmasked % |
|------|----------|-------|------------|
| 1/9 | 6 | 48 | 12.5% |
| 2/9 | 11 | 48 | 22.9% |
| 3/9 | 16 | 48 | 33.3% |
| 4/9 | 22 | 48 | 45.8% |
| 5/9 | 27 | 48 | 56.2% |
| 6/9 | 32 | 48 | 66.7% |
| 7/9 | 38 | 48 | 79.2% |
| 8/9 | 43 | 48 | 89.6% |
| 9/9 | 48 | 48 | 100.0% |

**Step 1/9** — Unmasked: 6/48 (12.5%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        4.8%   [MASK]                      <|endoftext|>        4.7%
  [  3-  5] [MASK]                      [MASK]                      [MASK]                   
  [  6-  8] <|endoftext|>        4.7%   <|endoftext|>        4.6%   <|endoftext|>        4.7%
  [  9- 11] [MASK]                      [MASK]                      [MASK]                   
  [ 12- 14] [MASK]                      <|endoftext|>        4.8%   [MASK]                   
  [ 15- 17] [MASK]                      [MASK]                      [MASK]                   
  [ 18- 20] [MASK]                      [MASK]                      [MASK]                   
  [ 21- 23] [MASK]                      [MASK]                      [MASK]                   
  [ 24- 26] [MASK]                      [MASK]                      [MASK]                   
  [ 27- 29] [MASK]                      [MASK]                      [MASK]                   
  [ 30- 32] [MASK]                      [MASK]                      [MASK]                   
  [ 33- 35] [MASK]                      [MASK]                      [MASK]                   
  [ 36- 38] [MASK]                      [MASK]                      [MASK]                   
  [ 39- 41] [MASK]                      [MASK]                      [MASK]                   
  [ 42- 44] [MASK]                      [MASK]                      [MASK]                   
  [ 45- 47] [MASK]                      [MASK]                      [MASK]                   
```

**Step 5/9** — Unmasked: 27/48 (56.2%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  6-  8] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 12- 14] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   <|endoftext|>        5.1%   [MASK]                   
  [ 18- 20] <|endoftext|>        prev   [MASK]                      [MASK]                   
  [ 21- 23] <|endoftext|>        5.1%   <|endoftext|>        prev   <|endoftext|>        prev
  [ 24- 26] [MASK]                      <|endoftext|>        5.0%   <|endoftext|>        5.1%
  [ 27- 29] <|endoftext|>        prev   [MASK]                      <|endoftext|>        prev
  [ 30- 32] <|endoftext|>        4.9%   <|endoftext|>        prev   [MASK]                   
  [ 33- 35] [MASK]                      [MASK]                      [MASK]                   
  [ 36- 38] [MASK]                      [MASK]                      [MASK]                   
  [ 39- 41] [MASK]                      [MASK]                      [MASK]                   
  [ 42- 44] [MASK]                      [MASK]                      [MASK]                   
  [ 45- 47] [MASK]                      [MASK]                      [MASK]                   
```

**Step 9/9** — Unmasked: 48/48 (100.0%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  6-  8] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 12- 14] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 18- 20] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 21- 23] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 24- 26] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 27- 29] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 30- 32] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        6.7%
  [ 33- 35] <|endoftext|>        6.9%   <|endoftext|>        prev   <|endoftext|>        7.3%
  [ 36- 38] <|endoftext|>        prev   <|endoftext|>        6.8%   <|endoftext|>        prev
  [ 39- 41] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 42- 44] <|endoftext|>        prev   <|endoftext|>        7.0%   <|endoftext|>        prev
  [ 45- 47] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
```

**Generated**: `<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|e...`
**Target**: `<FLOAT> <|-|><|1|><|.|><|0|><|4|><|0|><|0|> </FLOAT><|eot_id|>`

---

### Prompt B: 일반 질문 (base LLaDA 능력 확인)

- Prompt: `What is the chemical formula of water? The answer is`
- Prompt length: 11 tokens
- Target: `(expected: H2O)`

#### 10.5. [B] low_confidence + standard

| Config | Value |
|--------|-------|
| Remasking | low_confidence |
| Sampling | standard |
| Steps (requested) | 8 |
| gen_length | 48 |

- Actual steps: **8** (requested: 8)
- Actual gen_length: **48**

| Step | Unmasked | Total | Unmasked % |
|------|----------|-------|------------|
| 1/8 | 6 | 48 | 12.5% |
| 2/8 | 12 | 48 | 25.0% |
| 3/8 | 18 | 48 | 37.5% |
| 4/8 | 24 | 48 | 50.0% |
| 5/8 | 30 | 48 | 62.5% |
| 6/8 | 36 | 48 | 75.0% |
| 7/8 | 42 | 48 | 87.5% |
| 8/8 | 48 | 48 | 100.0% |

**Step 1/8** — Unmasked: 6/48 (12.5%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] [MASK]                      [MASK]                      [MASK]                   
  [  3-  5] [MASK]                      [MASK]                      [MASK]                   
  [  6-  8] [MASK]                      [MASK]                      [MASK]                   
  [  9- 11] [MASK]                      [MASK]                      <|endoftext|>        4.8%
  [ 12- 14] [MASK]                      [MASK]                      [MASK]                   
  [ 15- 17] [MASK]                      [MASK]                      [MASK]                   
  [ 18- 20] [MASK]                      [MASK]                      [MASK]                   
  [ 21- 23] [MASK]                      [MASK]                      [MASK]                   
  [ 24- 26] [MASK]                      [MASK]                      [MASK]                   
  [ 27- 29] [MASK]                      [MASK]                      [MASK]                   
  [ 30- 32] [MASK]                      [MASK]                      [MASK]                   
  [ 33- 35] [MASK]                      <|endoftext|>        4.9%   [MASK]                   
  [ 36- 38] <|endoftext|>        4.9%   [MASK]                      [MASK]                   
  [ 39- 41] [MASK]                      [MASK]                      <|endoftext|>        4.9%
  [ 42- 44] [MASK]                      [MASK]                      [MASK]                   
  [ 45- 47] [MASK]                      <|endoftext|>        4.9%   <|endoftext|>        4.9%
```

**Step 5/8** — Unmasked: 30/48 (62.5%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] [MASK]                      <|endoftext|>        prev   <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   [MASK]                      <|endoftext|>        5.5%
  [  6-  8] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 12- 14] [MASK]                      <|endoftext|>        prev   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        5.3%   <|endoftext|>        prev   [MASK]                   
  [ 18- 20] [MASK]                      [MASK]                      <|endoftext|>        5.3%
  [ 21- 23] [MASK]                      <|endoftext|>        5.5%   [MASK]                   
  [ 24- 26] [MASK]                      [MASK]                      [MASK]                   
  [ 27- 29] [MASK]                      <|endoftext|>        prev   [MASK]                   
  [ 30- 32] [MASK]                      [MASK]                      [MASK]                   
  [ 33- 35] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 36- 38] <|endoftext|>        prev   [MASK]                      <|endoftext|>        5.5%
  [ 39- 41] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 42- 44] [MASK]                      <|endoftext|>        5.4%   <|endoftext|>        prev
  [ 45- 47] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
```

**Step 8/8** — Unmasked: 48/48 (100.0%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  6-  8] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 12- 14] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 18- 20] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 21- 23] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        5.3%
  [ 24- 26] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        5.2%
  [ 27- 29] <|endoftext|>        5.3%   <|endoftext|>        prev   <|endoftext|>        prev
  [ 30- 32] <|endoftext|>        4.9%   <|endoftext|>        5.2%   <|endoftext|>        5.2%
  [ 33- 35] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 36- 38] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 39- 41] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 42- 44] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 45- 47] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
```

**Generated**: `<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|e...`
**Target**: `(expected: H2O)`

#### 10.6. [B] low_confidence + semi_ar

| Config | Value |
|--------|-------|
| Remasking | low_confidence |
| Sampling | semi_ar |
| Steps (requested) | 8 |
| gen_length | 48 |
| block_length | 16 |

- Actual steps: **9** (requested: 8)
- Actual gen_length: **48**

| Step | Unmasked | Total | Unmasked % |
|------|----------|-------|------------|
| 1/9 | 6 | 48 | 12.5% |
| 2/9 | 11 | 48 | 22.9% |
| 3/9 | 16 | 48 | 33.3% |
| 4/9 | 22 | 48 | 45.8% |
| 5/9 | 27 | 48 | 56.2% |
| 6/9 | 32 | 48 | 66.7% |
| 7/9 | 38 | 48 | 79.2% |
| 8/9 | 43 | 48 | 89.6% |
| 9/9 | 48 | 48 | 100.0% |

**Step 1/9** — Unmasked: 6/48 (12.5%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] [MASK]                      [MASK]                      <|endoftext|>        4.6%
  [  3-  5] [MASK]                      <|endoftext|>        4.6%   [MASK]                   
  [  6-  8] [MASK]                      [MASK]                      [MASK]                   
  [  9- 11] <|endoftext|>        4.6%   <|endoftext|>        4.6%   <|endoftext|>        4.8%
  [ 12- 14] <|endoftext|>        4.6%   [MASK]                      [MASK]                   
  [ 15- 17] [MASK]                      [MASK]                      [MASK]                   
  [ 18- 20] [MASK]                      [MASK]                      [MASK]                   
  [ 21- 23] [MASK]                      [MASK]                      [MASK]                   
  [ 24- 26] [MASK]                      [MASK]                      [MASK]                   
  [ 27- 29] [MASK]                      [MASK]                      [MASK]                   
  [ 30- 32] [MASK]                      [MASK]                      [MASK]                   
  [ 33- 35] [MASK]                      [MASK]                      [MASK]                   
  [ 36- 38] [MASK]                      [MASK]                      [MASK]                   
  [ 39- 41] [MASK]                      [MASK]                      [MASK]                   
  [ 42- 44] [MASK]                      [MASK]                      [MASK]                   
  [ 45- 47] [MASK]                      [MASK]                      [MASK]                   
```

**Step 5/9** — Unmasked: 27/48 (56.2%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  6-  8] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 12- 14] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 18- 20] [MASK]                      <|endoftext|>        prev   <|endoftext|>        prev
  [ 21- 23] [MASK]                      [MASK]                      [MASK]                   
  [ 24- 26] <|endoftext|>        6.9%   <|endoftext|>        prev   <|endoftext|>        6.7%
  [ 27- 29] <|endoftext|>        6.9%   <|endoftext|>        6.9%   [MASK]                   
  [ 30- 32] <|endoftext|>        6.9%   <|endoftext|>        prev   [MASK]                   
  [ 33- 35] [MASK]                      [MASK]                      [MASK]                   
  [ 36- 38] [MASK]                      [MASK]                      [MASK]                   
  [ 39- 41] [MASK]                      [MASK]                      [MASK]                   
  [ 42- 44] [MASK]                      [MASK]                      [MASK]                   
  [ 45- 47] [MASK]                      [MASK]                      [MASK]                   
```

**Step 9/9** — Unmasked: 48/48 (100.0%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  6-  8] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 12- 14] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 18- 20] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 21- 23] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 24- 26] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 27- 29] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 30- 32] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 33- 35] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 36- 38] <|endoftext|>        5.5%   <|endoftext|>        4.9%   <|endoftext|>        5.1%
  [ 39- 41] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        5.3%
  [ 42- 44] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        4.2%
  [ 45- 47] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
```

**Generated**: `<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|e...`
**Target**: `(expected: H2O)`

#### 10.7. [B] random + standard

| Config | Value |
|--------|-------|
| Remasking | random |
| Sampling | standard |
| Steps (requested) | 8 |
| gen_length | 48 |

- Actual steps: **8** (requested: 8)
- Actual gen_length: **48**

| Step | Unmasked | Total | Unmasked % |
|------|----------|-------|------------|
| 1/8 | 6 | 48 | 12.5% |
| 2/8 | 12 | 48 | 25.0% |
| 3/8 | 18 | 48 | 37.5% |
| 4/8 | 24 | 48 | 50.0% |
| 5/8 | 30 | 48 | 62.5% |
| 6/8 | 36 | 48 | 75.0% |
| 7/8 | 42 | 48 | 87.5% |
| 8/8 | 48 | 48 | 100.0% |

**Step 1/8** — Unmasked: 6/48 (12.5%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        4.5%   [MASK]                      [MASK]                   
  [  3-  5] [MASK]                      [MASK]                      <|endoftext|>        4.5%
  [  6-  8] [MASK]                      <|endoftext|>        4.5%   [MASK]                   
  [  9- 11] [MASK]                      <|endoftext|>        4.6%   [MASK]                   
  [ 12- 14] [MASK]                      [MASK]                      [MASK]                   
  [ 15- 17] [MASK]                      [MASK]                      [MASK]                   
  [ 18- 20] <|endoftext|>        4.5%   [MASK]                      [MASK]                   
  [ 21- 23] [MASK]                      [MASK]                      [MASK]                   
  [ 24- 26] [MASK]                      [MASK]                      [MASK]                   
  [ 27- 29] [MASK]                      [MASK]                      [MASK]                   
  [ 30- 32] [MASK]                      [MASK]                      [MASK]                   
  [ 33- 35] [MASK]                      [MASK]                      [MASK]                   
  [ 36- 38] <|endoftext|>        4.9%   [MASK]                      [MASK]                   
  [ 39- 41] [MASK]                      [MASK]                      [MASK]                   
  [ 42- 44] [MASK]                      [MASK]                      [MASK]                   
  [ 45- 47] [MASK]                      [MASK]                      [MASK]                   
```

**Step 5/8** — Unmasked: 30/48 (62.5%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   [MASK]                      <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   <|endoftext|>        7.7%   <|endoftext|>        prev
  [  6-  8] [MASK]                      <|endoftext|>        prev   [MASK]                   
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   [MASK]                   
  [ 12- 14] [MASK]                      [MASK]                      <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   [MASK]                      [MASK]                   
  [ 18- 20] <|endoftext|>        prev   [MASK]                      [MASK]                   
  [ 21- 23] <|endoftext|>        prev   <|endoftext|>        7.4%   <|endoftext|>        prev
  [ 24- 26] <|endoftext|>        prev   <|endoftext|>        7.6%   [MASK]                   
  [ 27- 29] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        7.9%
  [ 30- 32] <|endoftext|>        7.6%   <|endoftext|>        prev   <|endoftext|>        prev
  [ 33- 35] [MASK]                      [MASK]                      <|endoftext|>        prev
  [ 36- 38] <|endoftext|>        prev   <|endoftext|>        prev   [MASK]                   
  [ 39- 41] <|endoftext|>        prev   [MASK]                      [MASK]                   
  [ 42- 44] [MASK]                      <|endoftext|>        prev   <|endoftext|>        prev
  [ 45- 47] [MASK]                      <|endoftext|>        prev   <|endoftext|>        7.7%
```

**Step 8/8** — Unmasked: 48/48 (100.0%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   <|endoftext|>        5.8%   <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  6-  8] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        5.6%
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        5.6%
  [ 12- 14] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 18- 20] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        5.7%
  [ 21- 23] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 24- 26] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 27- 29] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 30- 32] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 33- 35] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 36- 38] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 39- 41] <|endoftext|>        prev   <|endoftext|>        5.4%   <|endoftext|>        prev
  [ 42- 44] <|endoftext|>        5.6%   <|endoftext|>        prev   <|endoftext|>        prev
  [ 45- 47] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
```

**Generated**: `<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|e...`
**Target**: `(expected: H2O)`

#### 10.8. [B] random + semi_ar

| Config | Value |
|--------|-------|
| Remasking | random |
| Sampling | semi_ar |
| Steps (requested) | 8 |
| gen_length | 48 |
| block_length | 16 |

- Actual steps: **9** (requested: 8)
- Actual gen_length: **48**

| Step | Unmasked | Total | Unmasked % |
|------|----------|-------|------------|
| 1/9 | 6 | 48 | 12.5% |
| 2/9 | 11 | 48 | 22.9% |
| 3/9 | 16 | 48 | 33.3% |
| 4/9 | 22 | 48 | 45.8% |
| 5/9 | 27 | 48 | 56.2% |
| 6/9 | 32 | 48 | 66.7% |
| 7/9 | 38 | 48 | 79.2% |
| 8/9 | 43 | 48 | 89.6% |
| 9/9 | 48 | 48 | 100.0% |

**Step 1/9** — Unmasked: 6/48 (12.5%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] [MASK]                      <|endoftext|>        4.5%   [MASK]                   
  [  3-  5] [MASK]                      [MASK]                      <|endoftext|>        4.5%
  [  6-  8] <|endoftext|>        4.5%   <|endoftext|>        4.5%   [MASK]                   
  [  9- 11] [MASK]                      <|endoftext|>        4.6%   [MASK]                   
  [ 12- 14] [MASK]                      <|endoftext|>        4.6%   [MASK]                   
  [ 15- 17] [MASK]                      [MASK]                      [MASK]                   
  [ 18- 20] [MASK]                      [MASK]                      [MASK]                   
  [ 21- 23] [MASK]                      [MASK]                      [MASK]                   
  [ 24- 26] [MASK]                      [MASK]                      [MASK]                   
  [ 27- 29] [MASK]                      [MASK]                      [MASK]                   
  [ 30- 32] [MASK]                      [MASK]                      [MASK]                   
  [ 33- 35] [MASK]                      [MASK]                      [MASK]                   
  [ 36- 38] [MASK]                      [MASK]                      [MASK]                   
  [ 39- 41] [MASK]                      [MASK]                      [MASK]                   
  [ 42- 44] [MASK]                      [MASK]                      [MASK]                   
  [ 45- 47] [MASK]                      [MASK]                      [MASK]                   
```

**Step 5/9** — Unmasked: 27/48 (56.2%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  6-  8] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 12- 14] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   <|endoftext|>        prev   [MASK]                   
  [ 18- 20] <|endoftext|>        prev   [MASK]                      [MASK]                   
  [ 21- 23] <|endoftext|>        4.9%   <|endoftext|>        prev   [MASK]                   
  [ 24- 26] <|endoftext|>        prev   <|endoftext|>        5.5%   <|endoftext|>        prev
  [ 27- 29] [MASK]                      <|endoftext|>        5.6%   <|endoftext|>        5.6%
  [ 30- 32] <|endoftext|>        prev   <|endoftext|>        5.6%   [MASK]                   
  [ 33- 35] [MASK]                      [MASK]                      [MASK]                   
  [ 36- 38] [MASK]                      [MASK]                      [MASK]                   
  [ 39- 41] [MASK]                      [MASK]                      [MASK]                   
  [ 42- 44] [MASK]                      [MASK]                      [MASK]                   
  [ 45- 47] [MASK]                      [MASK]                      [MASK]                   
```

**Step 9/9** — Unmasked: 48/48 (100.0%)

```
  Pos         Token                Prob   Token                Prob   Token                Prob
  ────────────────────────────────────────────────────────────────────────────────────────────
  [  0-  2] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  3-  5] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  6-  8] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [  9- 11] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 12- 14] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 15- 17] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 18- 20] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 21- 23] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 24- 26] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 27- 29] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 30- 32] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
  [ 33- 35] <|endoftext|>        prev   <|endoftext|>        5.8%   <|endoftext|>        prev
  [ 36- 38] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        5.9%
  [ 39- 41] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        6.0%
  [ 42- 44] <|endoftext|>        prev   <|endoftext|>        5.9%   <|endoftext|>        5.9%
  [ 45- 47] <|endoftext|>        prev   <|endoftext|>        prev   <|endoftext|>        prev
```

**Generated**: `<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|e...`
**Target**: `(expected: H2O)`
