import argparse
import os
import importlib.util
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Base"
MASK_ID = 126336

def load_official_generate_fn():
    local_generate = Path("/opt/11-MolDA/New_MolDA/src/LLaDA/generate.py")
    if not local_generate.exists():
        raise FileNotFoundError(f"Official generate.py not found: {local_generate}")
    spec = importlib.util.spec_from_file_location("llada_generate_module", str(local_generate))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[INFO] HF_HOME={os.environ.get('HF_HOME')}")
    print(f"[INFO] HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
    print(f"[INFO] Loading tokenizer from {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        revision=args.revision,
    )

    print(f"[INFO] Loading model from {args.model} with dtype={dtype}")
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        revision=args.revision,
    )
    model.eval()

    generate_fn = load_official_generate_fn()

    prompt_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = prompt_ids.shape[1]

    print(f"[INFO] Prompt token length: {prompt_len}")
    print(f"[INFO] steps={args.steps}, gen_length={args.gen_length}, block_length={args.block_length}")
    print("[INFO] Running official LLaDA generate()...")

    with torch.no_grad():
        output_ids = generate_fn(
            model=model,
            prompt=prompt_ids,
            attention_mask=None,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
            mask_id=MASK_ID,
        )

    if isinstance(output_ids, torch.Tensor):
        output_ids = output_ids[0].tolist()

    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("\n" + "=" * 80)
    print("[PROMPT]")
    print(args.prompt)
    print("\n" + "=" * 80)
    print("[GENERATED]")
    print(decoded)
    print("=" * 80)

if __name__ == "__main__":
    main()
