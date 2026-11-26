"""Adapters turning dLLM checkpoints into the ``ModelFn`` used by the sampler.

The paper evaluates two open dLLMs (Sec. 4.1):

* LLaDA-8B-Instruct (Nie et al., 2025b) — ``GSAI-ML/LLaDA-8B-Instruct``
* Dream-v0-Instruct-7B (Ye et al., 2025) — ``Dream-org/Dream-v0-Instruct-7B``

Backbone registration (licence, revision, parameter count) belongs in
``Backbones/Generative_Pretrained``; this file only provides the inference
contract ``ids [B, N] -> logits [B, N, V]`` with ``logits[:, k]`` predicting
position ``k``. Both checkpoints need ``trust_remote_code=True``; pin a
``revision`` for reproducible runs.

Family-specific details (matching the upstream reference generation code):

* LLaDA: plain bidirectional mask predictor, ``model(x).logits`` predicts the
  token at each position. The reference ``generate.py`` uses mask id 126336.
* Dream: initialised from an AR model, its logits at position ``k`` predict
  position ``k + 1``; the reference sampler shifts them right by one
  (``cat([logits[:, :1], logits[:, :-1]])``) and calls the model with
  ``attention_mask="full"`` when no padding is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

__all__ = ["DLLMHandle", "LLaDAAdapter", "DreamAdapter", "load_hf_dllm", "MODEL_IDS"]

MODEL_IDS = {
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
    "dream": "Dream-org/Dream-v0-Instruct-7B",
}

# Fallback ids used by the official reference code if the tokenizer does not expose them.
_LLADA_MASK_ID = 126336


class LLaDAAdapter(torch.nn.Module):
    """``ModelFn`` wrapper for LLaDA-style mask predictors (trainable as a module)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).logits


class DreamAdapter(torch.nn.Module):
    """``ModelFn`` wrapper for Dream (applies the one-position logit shift)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            logits = self.model(x, attention_mask="full").logits
        except TypeError:  # older/newer remote code without the string mask
            logits = self.model(x).logits
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


@dataclass
class DLLMHandle:
    """A loaded dLLM: callable model, tokenizer and special token ids."""

    family: str
    model_id: str
    model_fn: object
    tokenizer: object
    mask_token_id: int
    eos_token_ids: List[int] = field(default_factory=list)
    revision: Optional[str] = None

    def encode_chat(self, user_message: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """Apply the instruct chat template and return ``[P]`` prompt ids."""
        messages = [{"role": "user", "content": user_message}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        return ids.to(device) if device is not None else ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        ids = [t for t in token_ids if t != self.mask_token_id]
        if self.eos_token_ids:
            for i, t in enumerate(ids):  # truncate at the first EOS
                if t in self.eos_token_ids:
                    ids = ids[:i]
                    break
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def load_hf_dllm(
    family: str,
    model_id: Optional[str] = None,
    revision: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> DLLMHandle:
    """Load LLaDA or Dream with Hugging Face ``transformers``.

    Requires ``transformers`` and enough accelerator memory for an 8B model
    (~16 GB in bf16). Weights are not committed to this repository.
    """
    from transformers import AutoModel, AutoTokenizer  # optional dependency

    family = family.lower()
    if family not in MODEL_IDS:
        raise ValueError(f"family must be one of {sorted(MODEL_IDS)}")
    model_id = model_id or MODEL_IDS[family]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=True, torch_dtype=dtype)
    model = model.to(device).eval()

    mask_id = getattr(tok, "mask_token_id", None)
    if mask_id is None:
        mask_id = getattr(getattr(model, "generation_config", None), "mask_token_id", None)
    if mask_id is None and family == "llada":
        mask_id = _LLADA_MASK_ID
    if mask_id is None:
        raise ValueError("could not determine the mask token id; pass it explicitly")

    eos = set()
    for attr in ("eos_token_id",):
        v = getattr(tok, attr, None)
        if v is not None:
            eos.add(int(v))
    for special in ("<|eot_id|>", "<|endoftext|>", "<|im_end|>"):
        tid = tok.convert_tokens_to_ids(special)
        if isinstance(tid, int) and tid != getattr(tok, "unk_token_id", None) and tid >= 0:
            eos.add(int(tid))

    fn = LLaDAAdapter(model) if family == "llada" else DreamAdapter(model)
    return DLLMHandle(family, model_id, fn, tok, int(mask_id), sorted(eos), revision)
