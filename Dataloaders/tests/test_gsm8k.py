"""Tests for the pure helpers in Dataloaders.evaluation.gsm8k (no network)."""

import pytest

from Dataloaders.evaluation.gsm8k import (
    build_prompt,
    extract_answer,
    gold_answer,
    is_correct,
    make_batches,
    normalize_number,
    select_indices,
)


def test_select_indices_deterministic():
    """Catches unseeded randomness."""
    assert select_indices(1319, 200, 42) == select_indices(1319, 200, 42)


def test_select_indices_sorted_and_length():
    """Catches unsorted output or wrong subset size."""
    idx = select_indices(1319, 200, 42, subset=1)
    assert idx == sorted(idx)
    assert len(idx) == 200
    assert len(set(idx)) == 200


def test_select_indices_subsets_disjoint_in_range():
    """Catches overlapping subsets or out-of-range indices."""
    subsets = [set(select_indices(1319, 200, 42, subset=s)) for s in range(3)]
    for s in subsets:
        assert all(0 <= i < 1319 for i in s)
    assert not (subsets[0] & subsets[1])
    assert not (subsets[0] & subsets[2])
    assert not (subsets[1] & subsets[2])


def test_select_indices_seed_changes_set():
    """Catches the seed being ignored."""
    assert select_indices(1319, 200, 42) != select_indices(1319, 200, 43)


def test_select_indices_none_returns_all():
    """Catches num_samples=None not returning the full range."""
    assert select_indices(10, None, 42) == list(range(10))


@pytest.mark.parametrize(
    "args",
    [
        (10, 4, 42, 2),  # third subset needs 12 items
        (10, 4, 42, -1),
        (10, 0, 42, 0),
        (10, None, 42, 1),
    ],
)
def test_select_indices_errors(args):
    """Catches silent acceptance of invalid subset or sample sizes."""
    with pytest.raises(ValueError):
        select_indices(*args)


@pytest.mark.parametrize("n, batch", [(0, 4), (3, 4), (4, 4), (8, 4), (9, 4)])
def test_make_batches_boundaries(n, batch):
    """Catches dropped, duplicated or reordered items at boundary sizes."""
    items = list(range(n))
    batches = make_batches(items, batch)
    assert [x for b in batches for x in b] == items
    assert all(1 <= len(b) <= batch for b in batches)
    assert all(len(b) == batch for b in batches[:-1])
    assert len(batches) == (n + batch - 1) // batch


def test_make_batches_last_chunk_short():
    """Catches the trailing partial batch being lost or padded."""
    assert make_batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_make_batches_bad_size():
    """Catches batch_size=0 causing an infinite loop or silent result."""
    with pytest.raises(ValueError):
        make_batches([1, 2], 0)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1,234", "1234"),
        ("$18.00", "18"),
        ("2.50", "2.5"),
        ("-3", "-3"),
        ("abc", None),
        ("18.", "18"),
        ("50%", "50"),
        ("", None),
    ],
)
def test_normalize_number(text, expected):
    assert normalize_number(text) == expected


def test_gold_answer_multiline():
    raw = (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
        "#### 72"
    )
    assert gold_answer(raw) == "72"


def test_gold_answer_with_commas():
    assert gold_answer("Total = 1,250 dollars.\n#### 1,250") == "1250"


def test_extract_with_phrase():
    assert extract_answer("We get 5 then 7. The final answer is 12.") == "12"


def test_extract_phrase_twice_uses_last():
    text = "The final answer is 10. Wait, recheck. The final answer is 14."
    assert extract_answer(text) == "14"


def test_extract_phrase_case_insensitive():
    assert extract_answer("3 + 4 = 7. the FINAL answer is 7") == "7"


def test_extract_no_phrase_last_number():
    assert extract_answer("First 3, then 8, so 21 in total") == "21"


def test_extract_no_numbers():
    assert extract_answer("I do not know.") is None


def test_extract_dollar_with_commas():
    assert extract_answer("The final answer is $1,200.50.") == "1200.5"


def test_extract_negative():
    assert extract_answer("The final answer is -3.") == "-3"


def test_extract_sentence_end_period():
    assert extract_answer("She has 18 apples.") == "18"


def test_is_correct_true():
    assert is_correct("... The final answer is 18.", "18")
    assert is_correct("The final answer is 18.0", "18")
    assert is_correct("so 2.50", "2.5")


def test_is_correct_false():
    assert not is_correct("The final answer is 19.", "18")
    assert not is_correct("no digits here", "18")


def test_build_prompt_contains_question():
    q = "What is 2 + 2?"
    prompt = build_prompt(q)
    assert prompt.startswith(q + "\n\n")
    assert 'The final answer is <number>."' in prompt
