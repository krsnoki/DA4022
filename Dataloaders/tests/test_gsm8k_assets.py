"""Tests for gsm8k.py that need locally cached assets (dataset, tokenizer).

Every test skips cleanly if the asset is not available offline.
"""

import copy
import os

# Must be set before importing datasets so nothing is ever downloaded.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

from Dataloaders.evaluation.gsm8k import (
    HF_CONFIG,
    HF_PATH,
    HF_REVISION,
    build_prompt,
    collate_prompts,
    gold_answer,
    load_gsm8k_subset,
    make_batches,
)

datasets = pytest.importorskip("datasets")

TOKENIZER_ID = "allenai/OLMoE-1B-7B-0125-Instruct"
TOKENIZER_REVISION = "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"


# ---------------------------------------------------------------- dataset


@pytest.fixture(scope="module")
def subsets():
    try:
        return [load_gsm8k_subset(200, 42, "test", s) for s in range(3)]
    except Exception:
        pytest.skip("GSM8K not available offline")


def test_subsets_shape_and_keys(subsets):
    """Catches wrong subset size or missing item fields."""
    for items in subsets:
        assert len(items) == 200
        for item in items:
            assert set(item) == {"index", "question", "answer", "solution"}


def test_subsets_disjoint(subsets):
    """Catches overlapping subsets (data leakage between runs)."""
    sets = [{it["index"] for it in items} for items in subsets]
    assert not (sets[0] & sets[1])
    assert not (sets[0] & sets[2])
    assert not (sets[1] & sets[2])


def test_loading_twice_identical(subsets):
    """Catches non-deterministic loading."""
    assert load_gsm8k_subset(200, 42, "test", 0) == subsets[0]


def test_answer_matches_gold_of_solution(subsets):
    """Catches answer and solution fields getting out of sync."""
    for items in subsets:
        for item in items:
            assert item["answer"] == gold_answer(item["solution"])


def test_no_test_question_in_train(subsets):
    """Train/test separation: catches test questions leaking into train."""
    try:
        train = datasets.load_dataset(
            HF_PATH, HF_CONFIG, revision=HF_REVISION, split="train"
        )
    except Exception:
        pytest.skip("GSM8K train split not available offline")
    train_questions = set(train["question"])
    for items in subsets:
        for item in items:
            assert item["question"] not in train_questions


# -------------------------------------------------------------- tokenizer

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tokenizer():
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            TOKENIZER_ID, revision=TOKENIZER_REVISION, local_files_only=True
        )
    except Exception:
        pytest.skip("OLMoE tokenizer not available offline")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


PROMPTS = [
    "Hi.",
    "What is two plus two? Please explain in a few words.",
    "A farmer has 12 cows and buys 7 more, then sells 5 of them at the market. "
    "How many cows does the farmer have now? Show every step of the work.",
]


def test_collate_shapes_longest(tokenizer):
    """Catches wrong batch shape or dtype."""
    out = collate_prompts(PROMPTS, tokenizer)
    lengths = [
        len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in PROMPTS
    ]
    assert out["input_ids"].shape == (3, max(lengths))
    assert out["attention_mask"].shape == out["input_ids"].shape
    assert out["input_ids"].dtype == torch.long
    assert out["attention_mask"].dtype == torch.long


def test_collate_left_padding(tokenizer):
    """Catches right padding, which breaks batched generation."""
    out = collate_prompts(PROMPTS, tokenizer)
    mask = out["attention_mask"]
    assert bool((mask[:, -1] == 1).all())
    for row in mask.tolist():
        first_one = row.index(1)
        assert all(v == 0 for v in row[:first_one])
        assert all(v == 1 for v in row[first_one:])
    assert int(mask.sum()) < mask.numel()  # the prompts differ, so some padding


def test_collate_tail_equals_solo_tokenization(tokenizer):
    """Catches batch tokenization differing from tokenizing a prompt alone."""
    out = collate_prompts(PROMPTS, tokenizer)
    for i, p in enumerate(PROMPTS):
        solo = tokenizer(p, add_special_tokens=False)["input_ids"]
        real = out["input_ids"][i][out["attention_mask"][i] == 1].tolist()
        assert real == solo


def test_collate_restores_padding_side(tokenizer):
    """Catches the tokenizer's padding_side leaking out of the call."""
    original = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        collate_prompts(PROMPTS, tokenizer)
        assert tokenizer.padding_side == "right"
    finally:
        tokenizer.padding_side = original


def test_collate_single_prompt_no_padding(tokenizer):
    """Catches padding being added to a batch of one."""
    out = collate_prompts(PROMPTS[:1], tokenizer)
    assert out["input_ids"].shape[0] == 1
    assert bool((out["attention_mask"] == 1).all())


def test_collate_max_length_exceeded(tokenizer):
    """Catches silent over-length prompts (or silent truncation)."""
    with pytest.raises(ValueError):
        collate_prompts(PROMPTS, tokenizer, max_length=3)


def test_collate_max_length_ok(tokenizer):
    """Catches max_length rejecting prompts that fit."""
    out = collate_prompts(PROMPTS, tokenizer, max_length=10_000)
    assert out["input_ids"].shape[0] == 3


def test_collate_empty_raises(tokenizer):
    """Catches an empty batch producing a confusing tokenizer error."""
    with pytest.raises(ValueError):
        collate_prompts([], tokenizer)


def test_collate_no_pad_token_raises(tokenizer):
    """Catches collating without a pad token."""
    tok = copy.deepcopy(tokenizer)
    tok.pad_token = None
    if tok.pad_token_id is not None:
        pytest.skip("could not clear pad token on tokenizer copy")
    with pytest.raises(ValueError):
        collate_prompts(PROMPTS, tok)


def test_batches_then_collate_real_prompts(tokenizer):
    """Catches batch/collate disagreement: 9 prompts, size 4 -> 4, 4, 1."""
    questions = [f"Tom has {n} apples and eats {n - 1}. How many are left?" for n in range(2, 11)]
    prompts = [build_prompt(q) for q in questions]
    batches = make_batches(prompts, 4)
    shapes = [collate_prompts(b, tokenizer)["input_ids"].shape for b in batches]
    assert [s[0] for s in shapes] == [4, 4, 1]
