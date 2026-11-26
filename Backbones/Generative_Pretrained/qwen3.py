"""Pinned Qwen3 loader. Downloading weights happens only on explicit load()."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def load(device="cuda", dtype="float16", model_id=MODEL_ID, revision=REVISION):
    if device == "cpu" and dtype != "float32":
        raise ValueError("use float32 for the CPU reference")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=getattr(torch, dtype),
        attn_implementation="sdpa",
    ).to(device).eval()
    return model, tokenizer
