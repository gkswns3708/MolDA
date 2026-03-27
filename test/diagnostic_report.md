# MolDA Training Step Diagnostic Report

> Generated: 2026-03-27 11:41:05
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

- Batch size: **2**
- Sequence length: **512** (max_length=512)
- Tasks in batch: `['bace', 'smol-property_prediction-hiv']`

### Sample 0 상세
- Task: `bace`
- Prompt length: **123** tokens
- Answer length: **5** tokens
- Padding length: **384** tokens (EOS, id=128001)
- Total: 123 (prompt) + 5 (answer) + 384 (pad) = 512

#### Prompt 전체 (123 tokens)

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
   44 |   110799 |     -100 | Predict              | prompt
   45 |      268 |     -100 |  the                 | prompt
   46 |    18492 |     -100 |  biological          | prompt
   47 |     5367 |     -100 |  activity            | prompt
   48 |      300 |     -100 |  of                  | prompt
   49 |      268 |     -100 |  the                 | prompt
   50 |    31051 |     -100 |  molecule            | prompt
   51 |      220 |     -100 |                      | prompt
   52 |   126355 |     -100 | <SELFIES>            | prompt
   53 |      220 |     -100 |                      | prompt
   54 |   128074 |     -100 | [C]                  | prompt
   55 |   128074 |     -100 | [C]                  | prompt
   56 |   128369 |     -100 | [Branch1]            | prompt
   57 |   128074 |     -100 | [C]                  | prompt
   58 |   128074 |     -100 | [C]                  | prompt
   59 |   128369 |     -100 | [Branch1]            | prompt
   60 |   128074 |     -100 | [C]                  | prompt
   61 |   128074 |     -100 | [C]                  | prompt
   62 |   128074 |     -100 | [C]                  | prompt
   63 |   127213 |     -100 | [=C]                 | prompt
   64 |   128074 |     -100 | [C]                  | prompt
   65 |   127213 |     -100 | [=C]                 | prompt
   66 |   128074 |     -100 | [C]                  | prompt
   67 |   127158 |     -100 | [Branch2]            | prompt
   68 |   129174 |     -100 | [Ring2]              | prompt
   69 |   126707 |     -100 | [Ring1]              | prompt
   70 |   128074 |     -100 | [C]                  | prompt
   71 |   128496 |     -100 | [NH2+1]              | prompt
   72 |   128074 |     -100 | [C]                  | prompt
   73 |   128074 |     -100 | [C]                  | prompt
   74 |   126933 |     -100 | [S]                  | prompt
   75 |   128787 |     -100 | [=Branch1]           | prompt
   76 |   128074 |     -100 | [C]                  | prompt
   77 |   127564 |     -100 | [=O]                 | prompt
   78 |   128787 |     -100 | [=Branch1]           | prompt
   79 |   128074 |     -100 | [C]                  | prompt
   80 |   127564 |     -100 | [=O]                 | prompt
   81 |   128074 |     -100 | [C]                  | prompt
   82 |   128074 |     -100 | [C]                  | prompt
   83 |   128369 |     -100 | [Branch1]            | prompt
   84 |   126933 |     -100 | [S]                  | prompt
   85 |   128074 |     -100 | [C]                  | prompt
   86 |   128074 |     -100 | [C]                  | prompt
   87 |   127213 |     -100 | [=C]                 | prompt
   88 |   128074 |     -100 | [C]                  | prompt
   89 |   127213 |     -100 | [=C]                 | prompt
   90 |   128369 |     -100 | [Branch1]            | prompt
   91 |   128074 |     -100 | [C]                  | prompt
   92 |   128785 |     -100 | [N]                  | prompt
   93 |   128074 |     -100 | [C]                  | prompt
   94 |   128369 |     -100 | [Branch1]            | prompt
   95 |   128074 |     -100 | [C]                  | prompt
   96 |   128155 |     -100 | [F]                  | prompt
   97 |   127213 |     -100 | [=C]                 | prompt
   98 |   126707 |     -100 | [Ring1]              | prompt
   99 |   127158 |     -100 | [Branch2]            | prompt
  100 |   128074 |     -100 | [C]                  | prompt
  101 |   126707 |     -100 | [Ring1]              | prompt
  102 |   129008 |     -100 | [P]                  | prompt
  103 |   128837 |     -100 | [O]                  | prompt
  104 |   127213 |     -100 | [=C]                 | prompt
  105 |   129174 |     -100 | [Ring2]              | prompt
  106 |   126707 |     -100 | [Ring1]              | prompt
  107 |   126990 |     -100 | [#Branch2]           | prompt
  108 |      220 |     -100 |                      | prompt
  109 |   126356 |     -100 | </SELFIES>           | prompt
  110 |     2864 |     -100 |  against             | prompt
  111 |      413 |     -100 |  B                   | prompt
  112 |    10595 |     -100 | ACE                  | prompt
  113 |       12 |     -100 | -                    | prompt
  114 |       16 |     -100 | 1                    | prompt
  115 |       13 |     -100 | .                    | prompt
  116 |   126348 |     -100 | <|eot_id|>           | prompt
  117 |   126346 |     -100 | <|start_header_id…   | prompt
  118 |      598 |     -100 | ass                  | prompt
  119 |    10450 |     -100 | istant               | prompt
  120 |   126347 |     -100 | <|end_header_id|>    | prompt
  121 |      198 |     -100 | \n                   | prompt
  122 |      198 |     -100 | \n                   | prompt
```

#### Answer 전체 (5 tokens)

```
  Pos | Token ID |    Label | Decoded              | Region
───────────────────────────────────────────────────────────────
  123 |   126349 |   126349 | <BOOLEAN>            | answer  OK
  124 |    10158 |    10158 |  False               | answer  OK
  125 |      220 |      220 |                      | answer  OK
  126 |   126350 |   126350 | </BOOLEAN>           | answer  OK
  127 |   126348 |   126348 | <|eot_id|>           | answer  OK
```

#### Padding (첫 5 / 384 tokens)

```
  Pos | Token ID |    Label | Decoded              | Region
───────────────────────────────────────────────────────────────
  128 |   126081 |     -100 | <|endoftext|>        | padding (EOS)
  129 |   126081 |     -100 | <|endoftext|>        | padding (EOS)
  130 |   126081 |     -100 | <|endoftext|>        | padding (EOS)
  131 |   126081 |     -100 | <|endoftext|>        | padding (EOS)
  132 |   126081 |     -100 | <|endoftext|>        | padding (EOS)
  ... (379 more padding tokens, all id=128001)
```

## 3. Forward Process — `make_noisy()`

LLaDA Masked Diffusion: `t ~ U(0,1)` → `p_mask = (1-eps)*t + eps` → answer 토큰을 확률 p_mask로 MASK 교체

### Masking 결과
| Sample | p_mask | Answer 길이 | Masked 수 | Mask 비율 |
|--------|--------|------------|----------|-----------|
| 0 | 0.6133 | 5 | 4 | 80.0% |
| 1 | 0.0110 | 5 | 1 | 20.0% |

### Sample 0 마스킹 시각화 (answer 영역, 첫 40 tokens)

```
Position  : 원본 ID → Noisy ID  [MASK?]  Decoded
──────────────────────────────────────────────────────────────────────
  [ 123] : 126349 → 126336  ██ MASK  '<BOOLEAN>'
  [ 124] :  10158 → 126336  ██ MASK  ' False'
  [ 125] :    220 → 126336  ██ MASK  ' '
  [ 126] : 126350 → 126336  ██ MASK  '</BOOLEAN>'
  [ 127] : 126348 → 126348           '<|eot_id|>'
```

- Prompt 영역 보존: **OK** (noisy_ids[:prompt_len] == input_ids[:prompt_len])
- MASK token ID: **126336** (`<|mdm_mask|>`)

## 4. Model Forward Pass

- Input: `noisy_ids` [2, 512]
- Output: `logits` [2, 512, 129325] (B, L, Vocab=129325)

## 5. Loss 계산 — `MaskedDiffusionLoss.forward()`

공식: `loss = Σ [ CE(logit, target) / p_mask / answer_length ] / batch_size`

### 계산 결과
| 항목 | 값 |
|------|-----|
| **Loss** | **172.008636** |
| Answer length mean | 5.00 |
| Loss is finite | YES |
| Loss is positive | YES |

### Sample 0 — 전체 시퀀스 Prediction & Loss (p_mask=0.6133, ans_len=5)

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
   44 |      P |  110799 | Predict          |       — | —                |           — |           — |           —
   45 |      P |     268 |  the             |       — | —                |           — |           — |           —
   46 |      P |   18492 |  biological      |       — | —                |           — |           — |           —
   47 |      P |    5367 |  activity        |       — | —                |           — |           — |           —
   48 |      P |     300 |  of              |       — | —                |           — |           — |           —
   49 |      P |     268 |  the             |       — | —                |           — |           — |           —
   50 |      P |   31051 |  molecule        |       — | —                |           — |           — |           —
   51 |      P |     220 |                  |       — | —                |           — |           — |           —
   52 |      P |  126355 | <SELFIES>        |       — | —                |           — |           — |           —
   53 |      P |     220 |                  |       — | —                |           — |           — |           —
   54 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   55 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   56 |      P |  128369 | [Branch1]        |       — | —                |           — |           — |           —
   57 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   58 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   59 |      P |  128369 | [Branch1]        |       — | —                |           — |           — |           —
   60 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   61 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   62 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   63 |      P |  127213 | [=C]             |       — | —                |           — |           — |           —
   64 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   65 |      P |  127213 | [=C]             |       — | —                |           — |           — |           —
   66 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   67 |      P |  127158 | [Branch2]        |       — | —                |           — |           — |           —
   68 |      P |  129174 | [Ring2]          |       — | —                |           — |           — |           —
   69 |      P |  126707 | [Ring1]          |       — | —                |           — |           — |           —
   70 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   71 |      P |  128496 | [NH2+1]          |       — | —                |           — |           — |           —
   72 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   73 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   74 |      P |  126933 | [S]              |       — | —                |           — |           — |           —
   75 |      P |  128787 | [=Branch1]       |       — | —                |           — |           — |           —
   76 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   77 |      P |  127564 | [=O]             |       — | —                |           — |           — |           —
   78 |      P |  128787 | [=Branch1]       |       — | —                |           — |           — |           —
   79 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   80 |      P |  127564 | [=O]             |       — | —                |           — |           — |           —
   81 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   82 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   83 |      P |  128369 | [Branch1]        |       — | —                |           — |           — |           —
   84 |      P |  126933 | [S]              |       — | —                |           — |           — |           —
   85 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   86 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   87 |      P |  127213 | [=C]             |       — | —                |           — |           — |           —
   88 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   89 |      P |  127213 | [=C]             |       — | —                |           — |           — |           —
   90 |      P |  128369 | [Branch1]        |       — | —                |           — |           — |           —
   91 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   92 |      P |  128785 | [N]              |       — | —                |           — |           — |           —
   93 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   94 |      P |  128369 | [Branch1]        |       — | —                |           — |           — |           —
   95 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
   96 |      P |  128155 | [F]              |       — | —                |           — |           — |           —
   97 |      P |  127213 | [=C]             |       — | —                |           — |           — |           —
   98 |      P |  126707 | [Ring1]          |       — | —                |           — |           — |           —
   99 |      P |  127158 | [Branch2]        |       — | —                |           — |           — |           —
  100 |      P |  128074 | [C]              |       — | —                |           — |           — |           —
  101 |      P |  126707 | [Ring1]          |       — | —                |           — |           — |           —
  102 |      P |  129008 | [P]              |       — | —                |           — |           — |           —
  103 |      P |  128837 | [O]              |       — | —                |           — |           — |           —
  104 |      P |  127213 | [=C]             |       — | —                |           — |           — |           —
  105 |      P |  129174 | [Ring2]          |       — | —                |           — |           — |           —
  106 |      P |  126707 | [Ring1]          |       — | —                |           — |           — |           —
  107 |      P |  126990 | [#Branch2]       |       — | —                |           — |           — |           —
  108 |      P |     220 |                  |       — | —                |           — |           — |           —
  109 |      P |  126356 | </SELFIES>       |       — | —                |           — |           — |           —
  110 |      P |    2864 |  against         |       — | —                |           — |           — |           —
  111 |      P |     413 |  B               |       — | —                |           — |           — |           —
  112 |      P |   10595 | ACE              |       — | —                |           — |           — |           —
  113 |      P |      12 | -                |       — | —                |           — |           — |           —
  114 |      P |      16 | 1                |       — | —                |           — |           — |           —
  115 |      P |      13 | .                |       — | —                |           — |           — |           —
  116 |      P |  126348 | <|eot_id|>       |       — | —                |           — |           — |           —
  117 |      P |  126346 | <|start_header_… |       — | —                |           — |           — |           —
  118 |      P |     598 | ass              |       — | —                |           — |           — |           —
  119 |      P |   10450 | istant           |       — | —                |           — |           — |           —
  120 |      P |  126347 | <|end_header_id… |       — | —                |           — |           — |           —
  121 |      P |     198 | \n               |       — | —                |           — |           — |           —
  122 |      P |     198 | \n               |       — | —                |           — |           — |           —
  123 |  **M** |  126349 | <BOOLEAN>        |  126081 | <|endoftext|>    |     16.9310 |     27.6042 |    5.520847
  124 |  **M** |   10158 |  False           |  126081 | <|endoftext|>    |     10.1565 |     16.5591 |    3.311816
  125 |  **M** |     220 |                  |  126081 | <|endoftext|>    |      4.7803 |      7.7937 |    1.558746
  126 |  **M** |  126350 | </BOOLEAN>       |  126081 | <|endoftext|>    |     16.9661 |     27.6615 |    5.532306
  127 |      A |  126348 | <|eot_id|>       |  126081 | <|endoftext|>    |           — |           — |           —
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      |        |         | TOTAL            |         |                  |             |             |   15.923716
```

- Padding (384 tokens, all EOS id=128001) 생략
- **Sample 0 기여도 합계**: 15.923716
- **최종 loss** = (Σ all samples) / batch_size = 172.008636

> 학습 전이므로 예측 Token이 정답과 무관한 것이 정상. 학습이 진행되면 정답Token과 예측Token이 일치하기 시작.

## 6. Backward + Weight Update

### Gradient 통계 (backward 후)

| Layer | Grad Norm | Grad Mean | Grad Max |
|-------|-----------|-----------|----------|
| Embedding (orig vocab) | 1.404604e+05 | -2.102385e-05 | 2.611200e+04 |
| Embedding (new vocab) | 5.690806e+04 | -1.900609e-04 | 8.000000e+03 |
| Head (tied to wte) (orig vocab) | 1.404604e+05 | -2.102385e-05 | 2.611200e+04 |
| Head (tied to wte) (new vocab) | 5.690806e+04 | -1.900609e-04 | 8.000000e+03 |
| LoRA_A (`.transformer.blocks.0.q_proj.lora_A.default.weight`) | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| LoRA_B (`.transformer.blocks.0.q_proj.lora_B.default.weight`) | 5.521293e+02 | 4.037167e-03 | 2.490699e+01 |

### Weight 변화량 (optimizer.step() 후)

```
Layer                                    |    Before Norm |     After Norm |     Delta Norm |   Δ/Before
────────────────────────────────────────────────────────────────────────────────────────────────────
Embedding (orig, idx < 126349)           |     274.079102 |     274.079102 |     0.01039666 |   0.0038%
Embedding (new,  idx >= 126349)          |      45.490242 |      45.490246 |     0.00576400 |   0.0127%
Head/wte (tied) (orig, idx < 126349)     |     274.079102 |     274.079102 |     0.01039666 |   0.0038%
Head/wte (tied) (new,  idx >= 126349)    |      45.490242 |      45.490246 |     0.00576400 |   0.0127%
LoRA_A (first layer)                     |       4.618896 |       4.617700 |     0.00115462 |   0.0250%
LoRA_B (first layer)                     |       0.000000 |       1.248414 |     1.24841440 | 124841439723968.5000%
```

## 7. Embedding & Head 상세 — Original vs New Vocab

> Original vocab (idx 0 ~ 126348): LLaDA 기본 토큰
> New vocab (idx 126349 ~): 프로젝트 추가 토큰 (BOOL, FLOAT, SELFIES, ...)

### Input Embedding (wte)

```
구분                        |         Mean |          Std |           Norm |         Δ Norm
─────────────────────────────────────────────────────────────────────────────────────
Orig (before)             | -2.118877e-05 | 1.322869e-02 |     274.079102 | —
Orig (after)              | -2.118875e-05 | 1.322869e-02 |     274.079102 | 0.01039666
New  (before)             | -1.434004e-05 | 1.303166e-02 |      45.490242 | —
New  (after)              | -1.434005e-05 | 1.303166e-02 |      45.490246 | 0.00576400
```

### 특정 토큰별 Embedding 변화

```
Token                     |      ID |  Vocab |     Emb Δ Norm |    Head Δ Norm
───────────────────────────────────────────────────────────────────────────
orig — 'the'              |    1614 |   orig |     0.00001917 |     0.00001917
orig — 'molecule'         |      76 |   orig |     0.00003898 |     0.00003898
new  — '<BOOLEAN>'        |  126349 |    new |     0.00149579 |     0.00149579
new  — '<SELFIES>'        |  126355 |    new |     0.00149543 |     0.00149543
new  — '<FLOAT>'          |  126351 |    new |     0.00000000 |     0.00000000
new  — '<mol>'            |  126361 |    new |     0.00000000 |     0.00000000
```

## 8. LoRA Weight 변화

```
Layer (last 60 chars)                                          |    Grad Norm |  Δ Weight Norm
───────────────────────────────────────────────────────────────────────────────────────────────
odel.model.transformer.blocks.0.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.0.q_proj.lora_B.default.weight   | 3.639884e-03 | —
odel.model.transformer.blocks.0.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.0.k_proj.lora_B.default.weight   | 2.736303e-03 | —
odel.model.transformer.blocks.0.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.0.v_proj.lora_B.default.weight   | 9.651880e-03 | —
del.model.transformer.blocks.0.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.0.up_proj.lora_B.default.weight   | 5.607792e-03 | —
odel.model.transformer.blocks.1.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.1.q_proj.lora_B.default.weight   | 4.299622e-04 | —
odel.model.transformer.blocks.1.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.1.k_proj.lora_B.default.weight   | 4.330823e-04 | —
odel.model.transformer.blocks.1.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.1.v_proj.lora_B.default.weight   | 3.290577e-02 | —
del.model.transformer.blocks.1.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.1.up_proj.lora_B.default.weight   | 1.002469e-02 | —
odel.model.transformer.blocks.2.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.2.q_proj.lora_B.default.weight   | 4.465265e-04 | —
odel.model.transformer.blocks.2.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.2.k_proj.lora_B.default.weight   | 5.198809e-04 | —
odel.model.transformer.blocks.2.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.2.v_proj.lora_B.default.weight   | 8.283716e-03 | —
del.model.transformer.blocks.2.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.2.up_proj.lora_B.default.weight   | 1.451977e-02 | —
odel.model.transformer.blocks.3.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.3.q_proj.lora_B.default.weight   | 2.379733e-04 | —
odel.model.transformer.blocks.3.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.3.k_proj.lora_B.default.weight   | 2.340151e-04 | —
odel.model.transformer.blocks.3.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.3.v_proj.lora_B.default.weight   | 4.342079e-03 | —
del.model.transformer.blocks.3.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.3.up_proj.lora_B.default.weight   | 5.742252e-03 | —
odel.model.transformer.blocks.4.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.4.q_proj.lora_B.default.weight   | 2.918365e-04 | —
odel.model.transformer.blocks.4.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.4.k_proj.lora_B.default.weight   | 3.674092e-04 | —
odel.model.transformer.blocks.4.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.4.v_proj.lora_B.default.weight   | 6.642297e-03 | —
del.model.transformer.blocks.4.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.4.up_proj.lora_B.default.weight   | 3.952279e-03 | —
odel.model.transformer.blocks.5.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.5.q_proj.lora_B.default.weight   | 1.582789e-04 | —
odel.model.transformer.blocks.5.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.5.k_proj.lora_B.default.weight   | 2.425286e-04 | —
odel.model.transformer.blocks.5.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.5.v_proj.lora_B.default.weight   | 3.004526e-03 | —
del.model.transformer.blocks.5.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.5.up_proj.lora_B.default.weight   | 3.343521e-03 | —
odel.model.transformer.blocks.6.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.6.q_proj.lora_B.default.weight   | 1.697714e-04 | —
odel.model.transformer.blocks.6.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.6.k_proj.lora_B.default.weight   | 1.953617e-04 | —
odel.model.transformer.blocks.6.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.6.v_proj.lora_B.default.weight   | 4.065507e-03 | —
del.model.transformer.blocks.6.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.6.up_proj.lora_B.default.weight   | 3.007776e-03 | —
odel.model.transformer.blocks.7.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.7.q_proj.lora_B.default.weight   | 2.157885e-04 | —
odel.model.transformer.blocks.7.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.7.k_proj.lora_B.default.weight   | 1.918850e-04 | —
odel.model.transformer.blocks.7.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.7.v_proj.lora_B.default.weight   | 2.625596e-03 | —
del.model.transformer.blocks.7.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.7.up_proj.lora_B.default.weight   | 2.971064e-03 | —
odel.model.transformer.blocks.8.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.8.q_proj.lora_B.default.weight   | 4.641325e-04 | —
odel.model.transformer.blocks.8.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.8.k_proj.lora_B.default.weight   | 3.953750e-04 | —
odel.model.transformer.blocks.8.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.8.v_proj.lora_B.default.weight   | 2.053538e-03 | —
del.model.transformer.blocks.8.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.8.up_proj.lora_B.default.weight   | 1.478517e-03 | —
odel.model.transformer.blocks.9.q_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.9.q_proj.lora_B.default.weight   | 2.259605e-04 | —
odel.model.transformer.blocks.9.k_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.9.k_proj.lora_B.default.weight   | 2.304856e-04 | —
odel.model.transformer.blocks.9.v_proj.lora_A.default.weight   | 0.000000e+00 | —
odel.model.transformer.blocks.9.v_proj.lora_B.default.weight   | 1.458472e-03 | —
del.model.transformer.blocks.9.up_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.9.up_proj.lora_B.default.weight   | 1.365004e-03 | —
del.model.transformer.blocks.10.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.10.q_proj.lora_B.default.weight   | 1.566202e-04 | —
del.model.transformer.blocks.10.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.10.k_proj.lora_B.default.weight   | 1.730001e-04 | —
del.model.transformer.blocks.10.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.10.v_proj.lora_B.default.weight   | 9.898987e-04 | —
el.model.transformer.blocks.10.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.10.up_proj.lora_B.default.weight   | 5.376899e-04 | —
del.model.transformer.blocks.11.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.11.q_proj.lora_B.default.weight   | 2.169797e-04 | —
del.model.transformer.blocks.11.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.11.k_proj.lora_B.default.weight   | 1.786947e-04 | —
del.model.transformer.blocks.11.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.11.v_proj.lora_B.default.weight   | 1.154944e-03 | —
el.model.transformer.blocks.11.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.11.up_proj.lora_B.default.weight   | 7.029275e-04 | —
del.model.transformer.blocks.12.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.12.q_proj.lora_B.default.weight   | 3.559437e-04 | —
del.model.transformer.blocks.12.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.12.k_proj.lora_B.default.weight   | 5.211054e-04 | —
del.model.transformer.blocks.12.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.12.v_proj.lora_B.default.weight   | 1.398626e-03 | —
el.model.transformer.blocks.12.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.12.up_proj.lora_B.default.weight   | 7.191537e-04 | —
del.model.transformer.blocks.13.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.13.q_proj.lora_B.default.weight   | 1.671763e-04 | —
del.model.transformer.blocks.13.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.13.k_proj.lora_B.default.weight   | 2.227595e-04 | —
del.model.transformer.blocks.13.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.13.v_proj.lora_B.default.weight   | 1.594147e-03 | —
el.model.transformer.blocks.13.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.13.up_proj.lora_B.default.weight   | 5.982257e-04 | —
del.model.transformer.blocks.14.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.14.q_proj.lora_B.default.weight   | 4.911767e-04 | —
del.model.transformer.blocks.14.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.14.k_proj.lora_B.default.weight   | 5.532581e-04 | —
del.model.transformer.blocks.14.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.14.v_proj.lora_B.default.weight   | 1.402943e-03 | —
el.model.transformer.blocks.14.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.14.up_proj.lora_B.default.weight   | 4.903200e-04 | —
del.model.transformer.blocks.15.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.15.q_proj.lora_B.default.weight   | 4.725311e-04 | —
del.model.transformer.blocks.15.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.15.k_proj.lora_B.default.weight   | 2.545151e-04 | —
del.model.transformer.blocks.15.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.15.v_proj.lora_B.default.weight   | 1.319498e-03 | —
el.model.transformer.blocks.15.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.15.up_proj.lora_B.default.weight   | 4.271232e-04 | —
del.model.transformer.blocks.16.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.16.q_proj.lora_B.default.weight   | 2.920088e-04 | —
del.model.transformer.blocks.16.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.16.k_proj.lora_B.default.weight   | 1.955309e-04 | —
del.model.transformer.blocks.16.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.16.v_proj.lora_B.default.weight   | 1.078228e-03 | —
el.model.transformer.blocks.16.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.16.up_proj.lora_B.default.weight   | 4.071230e-04 | —
del.model.transformer.blocks.17.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.17.q_proj.lora_B.default.weight   | 2.986888e-04 | —
del.model.transformer.blocks.17.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.17.k_proj.lora_B.default.weight   | 1.828847e-04 | —
del.model.transformer.blocks.17.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.17.v_proj.lora_B.default.weight   | 9.462207e-04 | —
el.model.transformer.blocks.17.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.17.up_proj.lora_B.default.weight   | 4.300833e-04 | —
del.model.transformer.blocks.18.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.18.q_proj.lora_B.default.weight   | 2.174407e-04 | —
del.model.transformer.blocks.18.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.18.k_proj.lora_B.default.weight   | 1.810105e-04 | —
del.model.transformer.blocks.18.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.18.v_proj.lora_B.default.weight   | 8.408388e-04 | —
el.model.transformer.blocks.18.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.18.up_proj.lora_B.default.weight   | 4.065329e-04 | —
del.model.transformer.blocks.19.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.19.q_proj.lora_B.default.weight   | 2.074749e-04 | —
del.model.transformer.blocks.19.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.19.k_proj.lora_B.default.weight   | 1.378586e-04 | —
del.model.transformer.blocks.19.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.19.v_proj.lora_B.default.weight   | 7.785988e-04 | —
el.model.transformer.blocks.19.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.19.up_proj.lora_B.default.weight   | 3.334869e-04 | —
del.model.transformer.blocks.20.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.20.q_proj.lora_B.default.weight   | 2.622051e-04 | —
del.model.transformer.blocks.20.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.20.k_proj.lora_B.default.weight   | 2.099645e-04 | —
del.model.transformer.blocks.20.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.20.v_proj.lora_B.default.weight   | 8.046824e-04 | —
el.model.transformer.blocks.20.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.20.up_proj.lora_B.default.weight   | 3.238274e-04 | —
del.model.transformer.blocks.21.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.21.q_proj.lora_B.default.weight   | 2.127793e-04 | —
del.model.transformer.blocks.21.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.21.k_proj.lora_B.default.weight   | 1.867172e-04 | —
del.model.transformer.blocks.21.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.21.v_proj.lora_B.default.weight   | 7.717567e-04 | —
el.model.transformer.blocks.21.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.21.up_proj.lora_B.default.weight   | 4.005802e-04 | —
del.model.transformer.blocks.22.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.22.q_proj.lora_B.default.weight   | 2.427278e-04 | —
del.model.transformer.blocks.22.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.22.k_proj.lora_B.default.weight   | 1.722446e-04 | —
del.model.transformer.blocks.22.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.22.v_proj.lora_B.default.weight   | 6.531782e-04 | —
el.model.transformer.blocks.22.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.22.up_proj.lora_B.default.weight   | 3.843315e-04 | —
del.model.transformer.blocks.23.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.23.q_proj.lora_B.default.weight   | 2.102186e-04 | —
del.model.transformer.blocks.23.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.23.k_proj.lora_B.default.weight   | 1.565651e-04 | —
del.model.transformer.blocks.23.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.23.v_proj.lora_B.default.weight   | 6.530629e-04 | —
el.model.transformer.blocks.23.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.23.up_proj.lora_B.default.weight   | 5.947739e-04 | —
del.model.transformer.blocks.24.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.24.q_proj.lora_B.default.weight   | 2.189351e-04 | —
del.model.transformer.blocks.24.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.24.k_proj.lora_B.default.weight   | 1.826265e-04 | —
del.model.transformer.blocks.24.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.24.v_proj.lora_B.default.weight   | 5.369986e-04 | —
el.model.transformer.blocks.24.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.24.up_proj.lora_B.default.weight   | 2.547734e-04 | —
del.model.transformer.blocks.25.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.25.q_proj.lora_B.default.weight   | 1.550151e-04 | —
del.model.transformer.blocks.25.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.25.k_proj.lora_B.default.weight   | 1.398900e-04 | —
del.model.transformer.blocks.25.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.25.v_proj.lora_B.default.weight   | 4.104064e-04 | —
el.model.transformer.blocks.25.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.25.up_proj.lora_B.default.weight   | 1.845205e-04 | —
del.model.transformer.blocks.26.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.26.q_proj.lora_B.default.weight   | 1.008680e-04 | —
del.model.transformer.blocks.26.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.26.k_proj.lora_B.default.weight   | 8.415298e-05 | —
del.model.transformer.blocks.26.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.26.v_proj.lora_B.default.weight   | 2.773979e-04 | —
el.model.transformer.blocks.26.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.26.up_proj.lora_B.default.weight   | 1.723732e-04 | —
del.model.transformer.blocks.27.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.27.q_proj.lora_B.default.weight   | 9.551735e-05 | —
del.model.transformer.blocks.27.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.27.k_proj.lora_B.default.weight   | 8.821952e-05 | —
del.model.transformer.blocks.27.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.27.v_proj.lora_B.default.weight   | 2.653656e-04 | —
el.model.transformer.blocks.27.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.27.up_proj.lora_B.default.weight   | 1.405731e-04 | —
del.model.transformer.blocks.28.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.28.q_proj.lora_B.default.weight   | 1.197771e-04 | —
del.model.transformer.blocks.28.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.28.k_proj.lora_B.default.weight   | 8.367583e-05 | —
del.model.transformer.blocks.28.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.28.v_proj.lora_B.default.weight   | 2.079416e-04 | —
el.model.transformer.blocks.28.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.28.up_proj.lora_B.default.weight   | 1.404310e-04 | —
del.model.transformer.blocks.29.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.29.q_proj.lora_B.default.weight   | 1.079424e-04 | —
del.model.transformer.blocks.29.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.29.k_proj.lora_B.default.weight   | 1.136277e-04 | —
del.model.transformer.blocks.29.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.29.v_proj.lora_B.default.weight   | 1.636248e-04 | —
el.model.transformer.blocks.29.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.29.up_proj.lora_B.default.weight   | 1.839700e-04 | —
del.model.transformer.blocks.30.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.30.q_proj.lora_B.default.weight   | 7.910704e-05 | —
del.model.transformer.blocks.30.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.30.k_proj.lora_B.default.weight   | 7.990140e-05 | —
del.model.transformer.blocks.30.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.30.v_proj.lora_B.default.weight   | 1.279796e-04 | —
el.model.transformer.blocks.30.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.30.up_proj.lora_B.default.weight   | 1.793434e-04 | —
del.model.transformer.blocks.31.q_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.31.q_proj.lora_B.default.weight   | 9.069666e-05 | —
del.model.transformer.blocks.31.k_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.31.k_proj.lora_B.default.weight   | 7.748113e-05 | —
del.model.transformer.blocks.31.v_proj.lora_A.default.weight   | 0.000000e+00 | —
del.model.transformer.blocks.31.v_proj.lora_B.default.weight   | 9.831491e-05 | —
el.model.transformer.blocks.31.up_proj.lora_A.default.weight   | 0.000000e+00 | —
el.model.transformer.blocks.31.up_proj.lora_B.default.weight   | 2.443125e-04 | —
```

### LoRA_A (첫 번째 layer) 변화 상세
- Layer: `llada._model.base_model.model.model.transformer.blocks.0.q_proj.lora_A.default.weight`
- Shape: [64, 4096]
- Before norm: 4.618896
- After norm:  4.617700
- Delta norm:  0.00115462
- Delta max:   0.00000391

### LoRA_B (첫 번째 layer) 변화 상세
- Layer: `llada._model.base_model.model.model.transformer.blocks.0.q_proj.lora_B.default.weight`
- Shape: [4096, 64]
- Before norm: 0.000000
- After norm:  1.248414
- Delta norm:  1.24841440
- Delta max:   0.00249985
- LoRA_B는 초기값이 0 → 첫 step에서 0이 아닌 값으로 변화 (정상)

## 9. GPU 메모리 사용량

| 항목 | GB |
|------|-----|
| Allocated | 20.90 |
| Reserved | 30.39 |
| Total GPU | 47.59 |
| Free | 17.20 |
