from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset


@dataclass
class EvalWindows:
    input_ids: torch.Tensor       # int64 [num_windows, seq_len]
    total_available_windows: int
    seq_len: int
    is_partial: bool
    dataset_name: str
    dataset_config: str
    split: str


def load_wikitext2_eval_windows(
    tokenizer,
    seq_len: int = 2048,
    max_windows: Optional[int] = None,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
) -> EvalWindows:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(ds["text"])
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    total_available = encoded.numel() // seq_len
    if total_available == 0:
        raise ValueError(f"Not enough tokens ({encoded.numel()}) for seq_len={seq_len}")

    used_windows = total_available if max_windows is None else min(max_windows, total_available)
    is_partial = used_windows < total_available

    trimmed = encoded[: used_windows * seq_len].reshape(used_windows, seq_len).contiguous()
    return EvalWindows(
        input_ids=trimmed,
        total_available_windows=total_available,
        seq_len=seq_len,
        is_partial=is_partial,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
    )