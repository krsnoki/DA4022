import random

import torch

from Dataloaders.causal_lm import TokenBatch


def make_synthetic_batches(
    *,
    split: str = "calibration",
    seed: int = 0,
    count: int = 8,
    seq_len: int = 32,
    vocab_size: int = 128,
    batch_size: int = 1,
) -> tuple[TokenBatch, ...]:
    if split not in {"calibration", "evaluation"}:
        raise ValueError(f"unknown synthetic split: {split}")
    if min(count, seq_len, vocab_size, batch_size) < 1:
        raise ValueError("count, seq_len, vocab_size and batch_size must be positive")
    gen = torch.Generator(device="cpu").manual_seed(random.Random(f"{split}:{seed}").getrandbits(64))
    ids = torch.randint(vocab_size, (count, seq_len), generator=gen, device="cpu")
    return tuple(TokenBatch.unpadded(x) for x in ids.split(batch_size))
