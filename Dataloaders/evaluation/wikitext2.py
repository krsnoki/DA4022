"""WikiText-2 (raw) evaluation loader for language-model perplexity.

Follows the common GPTQ-style protocol: join the split's lines with ``"\\n\\n"``, tokenize once,
and cut the token stream into non-overlapping chunks of ``seq_len`` tokens (the remainder is
dropped). Role: ``evaluation`` only -- never use these chunks for calibration.

See ``Dataloaders/evaluation/README_wikitext2.md`` for the dataset card.
"""
from __future__ import annotations

from typing import Optional

import torch

DATASET_ID = "text.wikitext2.raw"
HF_PATH = "Salesforce/wikitext"
HF_CONFIG = "wikitext-2-raw-v1"

__all__ = ["DATASET_ID", "load_wikitext2_text", "token_chunks", "load_wikitext2_chunks"]


def load_wikitext2_text(split: str = "test", revision: Optional[str] = None) -> str:
    """Download (or read from the HF cache) one split and return it as a single string."""
    from datasets import load_dataset  # optional dependency

    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown WikiText-2 split {split!r}")
    ds = load_dataset(HF_PATH, HF_CONFIG, split=split, revision=revision)
    return "\n\n".join(ds["text"])


def token_chunks(token_ids: torch.Tensor, seq_len: int, max_chunks: Optional[int] = None) -> torch.Tensor:
    """Cut a 1-D token tensor into ``(n_chunks, seq_len)`` non-overlapping chunks."""
    if token_ids.dim() != 1:
        raise ValueError("token_ids must be 1-D")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    n = token_ids.numel() // seq_len
    if max_chunks is not None:
        n = min(n, max_chunks)
    if n == 0:
        raise ValueError(f"not enough tokens ({token_ids.numel()}) for one chunk of {seq_len}")
    return token_ids[: n * seq_len].reshape(n, seq_len).clone()


def load_wikitext2_chunks(
    tokenizer,
    split: str = "test",
    seq_len: int = 2048,
    max_chunks: Optional[int] = None,
    revision: Optional[str] = None,
) -> torch.Tensor:
    text = load_wikitext2_text(split, revision)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return token_chunks(ids, seq_len, max_chunks)
