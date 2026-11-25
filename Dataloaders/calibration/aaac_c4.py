"""Deterministic C4 train windows for the AAAC calibration pass."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

C4_DATASET_ID = "allenai/c4"
C4_CONFIG = "en"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"


def iter_c4_calibration_batches(
    tokenizer: Any,
    *,
    num_sequences: int = 4,
    sequence_length: int = 2048,
    seed: int = 42,
    shuffle_buffer_size: int = 10_000,
) -> Iterator[dict[str, Any]]:
    """Yield fixed-length, deterministically sampled batches from C4 train.

    The pinned English C4 training split is streamed and shuffled with the
    given seed. One contiguous token window is selected from each sufficiently
    long document. Windows contain no padding and have shape [1, sequence_length].
    """
    if num_sequences < 1 or sequence_length < 1 or shuffle_buffer_size < 1:
        raise ValueError("sequence counts, lengths, and shuffle buffer must be positive")

    import torch
    from datasets import load_dataset

    records = load_dataset(
        C4_DATASET_ID,
        C4_CONFIG,
        split="train",
        revision=C4_REVISION,
        streaming=True,
    ).shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    offsets = random.Random(seed)
    found = 0

    for record in records:
        text = record.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        if len(token_ids) < sequence_length:
            continue

        start = offsets.randrange(len(token_ids) - sequence_length + 1)
        input_ids = torch.tensor(
            [token_ids[start : start + sequence_length]], dtype=torch.long
        )
        yield {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        found += 1
        if found == num_sequences:
            return

    raise RuntimeError(
        f"C4 train ended after {found} usable windows; requested {num_sequences}"
    )
