"""Pinned loader and metadata for the base Qwen2.5-0.5B checkpoint."""

from __future__ import annotations

from typing import Any

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"

MODEL_METADATA = {
    "id": "qwen.qwen2_5_0_5b.base",
    "name": "Qwen2.5-0.5B",
    "modality": "text",
    "architecture": "Qwen2ForCausalLM",
    "pretraining_regime": "self_supervised",
    "objective": "causal next-token prediction",
    "weights_source": MODEL_ID,
    "revision": MODEL_REVISION,
    "license": "Apache-2.0",
    "context_length": 32768,
    "parameter_count": "0.49B (model card)",
}


def load_qwen25_05b(
    *,
    device: str | None = None,
    dtype: Any | None = None,
    local_files_only: bool = False,
):
    """Load the pinned base checkpoint and its tokenizer in evaluation mode.

    CUDA is used when available; CPU is the fallback. Pass ``device`` and
    ``dtype`` explicitly when a benchmark requires a fixed runtime setup.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        if target_device.startswith("cuda"):
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
    )
    # dtype is applied here rather than via from_pretrained(): that kwarg was renamed
    # (torch_dtype -> dtype) between transformers releases, .to() works on all of them.
    model.to(device=target_device, dtype=dtype)
    model.eval()
    return model, tokenizer
