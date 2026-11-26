"""GSM8K test-set loader, batching helpers and answer-matching helpers.

Loader contract
---------------
- Dataset ID: ``text.gsm8k.test`` (card: Dataloaders/evaluation/gsm8k_test.md).
- Source: Hugging Face ``openai/gsm8k``, config ``main``; licence MIT.
- Revision: pinned to ``HF_REVISION`` (a fixed commit hash).
- Task: grade-school maths word problems, free-form generation, numeric answer.
- Fields: ``question`` and ``answer`` (a worked solution ending in
  ``#### <number>``). The loader returns ``index``, ``question``, ``answer``
  (normalized gold number) and ``solution`` (raw answer field).
- Split: ``test`` by default (1319 examples).
- Subsets: the split is shuffled with ``random.Random(seed)`` and subset ``k``
  of size ``n`` is the slice ``[k*n, (k+1)*n)`` of that permutation, returned
  sorted. Subsets with the same seed and size are disjoint.
- Preprocessing: ``build_prompt(question)`` gives the user message, which is
  then wrapped by the model's own chat template.
- Batching: ``make_batches`` splits items into consecutive chunks and
  ``collate_prompts`` tokenizes a chunk with left padding, no truncation.
- Seed: 42 by default.
- Role: evaluation only. No calibration data is drawn from this loader.

Only the ``datasets`` package is needed, and it is imported lazily inside
``load_gsm8k_subset`` so the pure helpers can be used (and tested) without it.
``torch`` is imported lazily inside ``collate_prompts``.
"""

from __future__ import annotations

import random
import re
from decimal import Decimal

DATASET_ID = "text.gsm8k.test"
HF_PATH = "openai/gsm8k"
HF_CONFIG = "main"
HF_REVISION = "740312add88f781978c0658806c59bc2815b9866"

_PLAIN_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
# Optional sign, optional $, digits with optional thousands commas, optional decimals.
_NUMBER_IN_TEXT = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
_FINAL_PHRASE = "the final answer is"


def normalize_number(text: str) -> str | None:
    """Return a canonical string for a number, or None if ``text`` is not one.

    Strips ``$``, ``%``, commas, whitespace and one trailing period. Integers
    are rendered without a decimal point ("18", "-3"); other numbers have
    trailing zeros trimmed ("2.5").
    """
    s = text.replace("$", "").replace("%", "").replace(",", "")
    s = s.replace(" ", "").strip()
    if s.endswith("."):
        s = s[:-1]
    if not _PLAIN_NUMBER.fullmatch(s):
        return None
    value = Decimal(s)
    if value == value.to_integral_value():
        return str(int(value))
    return format(value, "f").rstrip("0").rstrip(".")


def gold_answer(raw_answer: str) -> str:
    """Extract the normalized gold number from a raw GSM8K ``answer`` field.

    GSM8K answers end with a line ``#### <number>``.
    """
    marker = "####"
    if marker not in raw_answer:
        raise ValueError("GSM8K answer has no '####' line")
    tail = raw_answer.rsplit(marker, 1)[1].strip()
    result = normalize_number(tail)
    if result is None:
        raise ValueError(f"Could not parse gold answer from: {tail!r}")
    return result


def extract_answer(generation: str) -> str | None:
    """Flexibly extract the final numeric answer from a model generation.

    If "The final answer is" (case-insensitive) occurs, use the first number
    after its last occurrence (falling back to the last number in the whole
    text if nothing numeric follows it). Otherwise use the last number in the
    text. Returns the normalized number string, or None if there is no number.
    """
    idx = generation.lower().rfind(_FINAL_PHRASE)
    if idx != -1:
        match = _NUMBER_IN_TEXT.search(generation, idx + len(_FINAL_PHRASE))
        if match is not None:
            return normalize_number(match.group(0))
    matches = _NUMBER_IN_TEXT.findall(generation)
    if not matches:
        return None
    return normalize_number(matches[-1])


def is_correct(generation: str, gold: str) -> bool:
    """True if the extracted answer equals ``gold`` numerically (tol 1e-6)."""
    predicted = extract_answer(generation)
    gold_norm = normalize_number(gold)
    if predicted is None or gold_norm is None:
        return False
    return abs(float(predicted) - float(gold_norm)) <= 1e-6


def build_prompt(question: str) -> str:
    """Build the user message: the question, a blank line, then instructions."""
    return (
        f"{question}\n\n"
        'Solve this step by step. End your reply with "The final answer is <number>."'
    )


def select_indices(
    n_total: int, num_samples: int | None, seed: int, subset: int = 0
) -> list[int]:
    """Pick sorted indices for one disjoint subset of ``range(n_total)``.

    The indices are shuffled with ``random.Random(seed)``; subset ``k`` is the
    slice ``[k*num_samples, (k+1)*num_samples)`` of that permutation. With
    ``num_samples=None`` the whole range is returned (only subset 0 is valid).
    """
    if num_samples is None:
        if subset != 0:
            raise ValueError("subset must be 0 when num_samples is None")
        return list(range(n_total))
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if subset < 0:
        raise ValueError("subset must be >= 0")
    perm = list(range(n_total))
    random.Random(seed).shuffle(perm)
    chosen = perm[subset * num_samples : (subset + 1) * num_samples]
    if len(chosen) < num_samples:
        raise ValueError(
            f"Not enough data for subset {subset}: need {num_samples} items "
            f"from {n_total} total"
        )
    return sorted(chosen)


def make_batches(items: list, batch_size: int) -> list[list]:
    """Split ``items`` into consecutive chunks; the last one may be shorter."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def collate_prompts(prompts: list[str], tokenizer, max_length: int | None = None) -> dict:
    """Tokenize already-formatted prompt strings into one left-padded batch.

    Returns ``{"input_ids": LongTensor[B, L], "attention_mask": LongTensor[B, L]}``.

    Padding is on the left because decoder-only generation appends new tokens
    on the right, so every row must end with its real last prompt token.
    ``add_special_tokens=False`` is used because the chat template has already
    inserted the special tokens; adding them again would duplicate them.
    Nothing is truncated; if the longest prompt exceeds ``max_length`` a
    ValueError is raised instead.
    """
    import torch  # lazy: keeps the pure helpers dependency-free

    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer has no pad token; set one before collating")
    if not prompts:
        raise ValueError("prompts is empty")
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = original_side
    input_ids = enc["input_ids"].to(torch.long)
    attention_mask = enc["attention_mask"].to(torch.long)
    length = input_ids.shape[1]
    if max_length is not None and length > max_length:
        lengths = attention_mask.sum(dim=1)
        longest = int(lengths.argmax())
        raise ValueError(
            f"prompt {longest} has {int(lengths[longest])} tokens, "
            f"more than max_length={max_length}"
        )
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_gsm8k_subset(
    num_samples: int | None = 200,
    seed: int = 42,
    split: str = "test",
    subset: int = 0,
) -> list[dict]:
    """Load a deterministic, disjoint-by-``subset`` slice of GSM8K.

    Indices come from ``select_indices`` and are sorted ascending.
    ``num_samples=None`` returns the whole split in order.

    Each item has keys: ``index`` (position in the split), ``question``,
    ``answer`` (normalized gold final answer) and ``solution`` (raw answer field).
    """
    from datasets import load_dataset  # lazy: keeps the pure helpers dependency-free

    ds = load_dataset(HF_PATH, HF_CONFIG, split=split, revision=HF_REVISION)
    indices = select_indices(len(ds), num_samples, seed, subset)
    items = []
    for i in indices:
        row = ds[i]
        items.append(
            {
                "index": i,
                "question": row["question"],
                "answer": gold_answer(row["answer"]),
                "solution": row["answer"],
            }
        )
    return items
