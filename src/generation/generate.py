"""
Wrapper around official LLaDA generate function.
src/official_LLaDA/generate.py is NOT modified — import only.

Supports all sampling strategies:
- Sampling: standard (block_length=gen_length) vs semi-AR (block_length < gen_length)
- Remasking: 'low_confidence' (Algorithm 5) vs 'random' (Algorithm 4)

generate_with_logging(): 공식 generate 로직 재구현 + step별 snapshot 수집.
    루프 내에서는 tensor clone만 수행 (file I/O, decode 없음 — 병목 방지).
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure official_LLaDA is importable
_official_dir = str(Path(__file__).resolve().parent.parent / "official_LLaDA")
if _official_dir not in sys.path:
    sys.path.insert(0, _official_dir)

from generate import generate as _llada_generate
from generate import add_gumbel_noise, get_num_transfer_tokens


def generate(
    model,
    prompt,
    attention_mask=None,
    gen_length: int = 256,
    steps: int = 32,
    remasking: str = "low_confidence",
    # Semi-AR control
    semi_ar: bool = False,
    block_length: int = 32,
    # Additional LLaDA params
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    mask_id: int = 126336,
    **kwargs,
):
    """Wrapper for LLaDA official generate().

    Args:
        model: LLaDA model (or PEFT-wrapped)
        prompt: [B, P] prompt token ids
        attention_mask: [B, P] attention mask
        gen_length: number of tokens to generate
        steps: total diffusion sampling steps
        remasking: 'low_confidence' or 'random'
        semi_ar: if True, use semi-autoregressive generation
        block_length: tokens per semi-AR block (only used when semi_ar=True)
            When semi_ar=False, block_length is set to gen_length (standard diffusion)
        temperature: Gumbel noise temperature (0 = greedy argmax)
        cfg_scale: classifier-free guidance scale (0 = disabled)
        mask_id: mask token id (LLaDA default: 126336)
        **kwargs: passed to official generate()

    Returns:
        output_ids: [B, P + gen_length] full sequence including prompt
    """
    # Determine effective block_length
    if semi_ar:
        effective_block_length = block_length
        # Ensure gen_length is divisible by block_length
        if gen_length % effective_block_length != 0:
            # Round up gen_length to nearest multiple
            gen_length = ((gen_length // effective_block_length) + 1) * effective_block_length
        # Ensure steps is divisible by num_blocks
        num_blocks = gen_length // effective_block_length
        if steps % num_blocks != 0:
            steps = ((steps // num_blocks) + 1) * num_blocks
    else:
        effective_block_length = gen_length

    return _llada_generate(
        model,
        prompt,
        attention_mask=attention_mask,
        gen_length=gen_length,
        steps=steps,
        block_length=effective_block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
        **kwargs,
    )


@torch.no_grad()
def generate_with_logging(
    model,
    prompt,
    attention_mask=None,
    gen_length: int = 256,
    steps: int = 32,
    remasking: str = "low_confidence",
    semi_ar: bool = False,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    mask_id: int = 126336,
    collect_confidence: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """공식 LLaDA generate 로직 재구현 + step별 snapshot 수집.

    루프 내에서는 tensor.clone().cpu() 만 수행 (file I/O, decode 없음).
    호출측에서 snapshot 리스트를 받아 deferred write 수행.

    Args:
        (generate() 와 동일)
        collect_confidence: True일 때만 각 step의 predicted token softmax prob를 수집.
            production(trainer.py)에서는 False(기본값) — 추가 softmax 연산 없음.

    Returns:
        output_ids: [B, P + gen_length] full sequence including prompt
        snapshots: list[Tensor] — 각 step의 gen_tokens [B, gen_length] (CPU)
        confidence_snapshots: list[Tensor] — 각 step의 predicted token prob [B, gen_length] (CPU).
            collect_confidence=False이면 빈 리스트.
            확정(non-mask) 위치는 1.0, mask 위치는 0.0.
    """
    # Determine effective block_length (generate()와 동일 로직)
    if semi_ar:
        effective_block_length = block_length
        if gen_length % effective_block_length != 0:
            gen_length = ((gen_length // effective_block_length) + 1) * effective_block_length
        num_blocks = gen_length // effective_block_length
        if steps % num_blocks != 0:
            steps = ((steps // num_blocks) + 1) * num_blocks
    else:
        effective_block_length = gen_length

    prompt_len = prompt.shape[1]

    # --- 공식 LLaDA generate.py 로직 재구현 (수정 금지 파일 건드리지 않음) ---
    x = torch.full(
        (prompt.shape[0], prompt_len + gen_length),
        mask_id, dtype=torch.long,
    ).to(model.device)
    x[:, :prompt_len] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(
                (prompt.shape[0], gen_length),
                dtype=attention_mask.dtype, device=model.device,
            ),
        ], dim=-1)

    prompt_index = (x != mask_id)

    num_blocks = gen_length // effective_block_length
    steps_per_block = steps // num_blocks

    snapshots: List[torch.Tensor] = []
    confidence_snapshots: List[torch.Tensor] = []

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * effective_block_length
        block_end = prompt_len + (num_block + 1) * effective_block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    logits = model(x_, attention_mask=attention_mask_).logits
                else:
                    logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1,
                )
            elif remasking == "random":
                x0_p = torch.rand(x0.shape[0], x0.shape[1], device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            # Snapshot: tensor clone만 (병목 없음, ~μs)
            snapshots.append(x[:, prompt_len:].clone().cpu())

            # Confidence snapshot (opt-in: collect_confidence=True일 때만)
            if collect_confidence:
                # 현재 x의 gen 영역에 대해: 확정된 토큰의 softmax prob 수집
                # remasking=="low_confidence"면 이미 p가 있으나, "random"이면 별도 계산 필요
                if remasking == "low_confidence":
                    # p는 이미 위에서 계산됨 — x0_p가 predicted token의 softmax prob
                    actual_prob = x0_p
                else:
                    # random remasking: softmax를 별도 계산 (collect_confidence=True일 때만)
                    p = F.softmax(logits, dim=-1)
                    actual_prob = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1,
                    )
                # 확정 위치(non-mask)는 1.0, 아직 mask인 위치는 0.0
                conf = torch.zeros_like(x, dtype=torch.float32)
                settled = (x != mask_id)
                # prompt 영역은 관심 없음 — gen 영역만
                conf[settled] = 1.0
                # 이번 step에서 새로 확정된 토큰에는 실제 확률 기록
                conf[transfer_index] = actual_prob[transfer_index].float()
                confidence_snapshots.append(conf[:, prompt_len:].clone().cpu())

    return x, snapshots, confidence_snapshots
