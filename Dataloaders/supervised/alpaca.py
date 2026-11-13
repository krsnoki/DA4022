"""Stanford Alpaca instruction-tuning data (dataset id ``text.alpaca.train``).

Used as the fine-tuning set of Pandey et al. (App. B.1).  Loss is computed on
the response tokens only (prompt tokens get label -100) unless
``train_on_prompt=True``.  A fixed, seeded hold-out of ``num_val`` examples is
split off for validation loss and never used for training.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

HF_ID = "tatsu-lab/alpaca"
REVISION = "dce01c9b08f87459cf36a430d809084718273017"

PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def format_prompt(ex: dict) -> str:
    tpl = PROMPT_INPUT if ex.get("input") else PROMPT_NO_INPUT
    return tpl.format(instruction=ex["instruction"], input=ex.get("input", ""))


def tokenize_example(ex: dict, tokenizer, max_length: int, train_on_prompt: bool = False) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(format_prompt(ex), add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(ex["output"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    ids = (prompt_ids + resp_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) if not train_on_prompt else list(prompt_ids)) + resp_ids
    return {"input_ids": ids, "labels": labels[:max_length]}


def load_alpaca(
    tokenizer,
    max_length: int = 512,
    num_train: Optional[int] = None,
    num_val: int = 500,
    seed: int = 42,
    train_on_prompt: bool = False,
    revision: str = REVISION,
):
    """Return ``(train_examples, val_examples)`` as lists of token-id dicts."""
    from datasets import load_dataset

    ds = load_dataset(HF_ID, split="train", revision=revision).shuffle(seed=seed)
    val = ds.select(range(num_val))
    train = ds.select(range(num_val, len(ds)))
    if num_train is not None:
        train = train.select(range(min(num_train, len(train))))

    def tok(split):
        out = [tokenize_example(ex, tokenizer, max_length, train_on_prompt) for ex in split]
        return [e for e in out if any(l != -100 for l in e["labels"])]

    return tok(train), tok(val)


def synthetic_lm_examples(n: int, seq_len: int, vocab_size: int, seed: int = 0):
    """Random token sequences with the same format (smoke tests only)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        L = int(torch.randint(seq_len // 2, seq_len + 1, (1,), generator=g))
        ids = torch.randint(3, vocab_size, (L,), generator=g).tolist()
        out.append({"input_ids": ids, "labels": [-100] * (L // 3) + ids[L // 3 :]})
    return out


class CausalLMCollator:
    """Right-pads ``input_ids``/``labels`` and builds ``attention_mask``."""

    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), self.pad, dtype=torch.long)
        labels = torch.full((len(batch), L), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            labels[i, :n] = torch.tensor(b["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}
