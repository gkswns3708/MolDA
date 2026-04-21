import os
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"

print("HF_HOME =", os.environ.get("HF_HOME"))
print("HF_HUB_CACHE =", os.environ.get("HF_HUB_CACHE"))
print("CUDA available =", torch.cuda.is_available())
print("CUDA device count =", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Model loaded successfully.")

text = "Hello, this is a smoke test."
inputs = tokenizer(text, return_tensors="pt")
print("Tokenizer ok.")
print("Input keys:", list(inputs.keys()))
