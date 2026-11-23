"""score_math.py normalizer / extraction cases."""
from __future__ import annotations

import pytest

from score_math import answers_match, gold_from_row, last_boxed, normalize_answer, score


@pytest.mark.parametrize("text,expected", [
    ("so the answer is \\boxed{42}.", "42"),
    ("\\boxed{1} then finally \\boxed{\\frac{3}{4}}", "\\frac{3}{4}"),
    ("\\boxed{\\sqrt{2}+\\frac{1}{2}}", "\\sqrt{2}+\\frac{1}{2}"),
    ("\\fbox{7}", "7"),
    ("no box here, 12", None),
    ("unbalanced \\boxed{3", None),
])
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


@pytest.mark.parametrize("pred,gold", [
    ("42", "42"), ("1,234", "1234"), ("$5$", "5"), ("\\text{(C)}", "(C)"), ("x = 3", "3"),
    ("90^\\circ", "90"), ("10\\%", "10"), ("\\dfrac{1}{2}", "\\frac{1}{2}"), ("\\frac{1}{2}", "0.5"),
    ("1/2", "\\frac12".replace("12", "{1}{2}")), ("3.0", "3"), ("18.", "18"), ("\\left(1,2\\right)", "(1,2)"),
    ("-\\frac{3}{4}", "-0.75"), ("\\textbf{204}", "204"), (" 7 ", "7"),
    ("\\$18", "18"), ("\\$18.00", "18"), ("\\$1,250", "1250"), ("\\$\\frac{1}{2}", "\\frac{1}{2}"),
])
def test_answers_match(pred, gold):
    assert answers_match(pred, gold), (normalize_answer(pred), normalize_answer(gold))


@pytest.mark.parametrize("pred,gold", [
    ("41", "42"), (None, "3"), ("", "0"), ("\\frac{1}{3}", "0.33"), ("(1,2)", "(2,1)"), ("1,2", "12"),
])
def test_answers_do_not_match(pred, gold):
    assert not answers_match(pred, gold)


def test_gold_extraction():
    assert gold_from_row("gsm8k", {"answer": "work...\n#### 1,000"}) == "1,000"
    assert answers_match("1000", gold_from_row("gsm8k", {"answer": "#### 1,000"}))
    assert gold_from_row("math500", {"answer": "\\frac{\\pi}{2}"}) == "\\frac{\\pi}{2}"
    assert gold_from_row("aime25", {"answer": 70}) == "70"


def test_score_groups():
    recs = [
        {"model": "m", "dataset": "gsm8k", "sample_idx": 0, "orig_idx": 5, "turn": 0, "mode": "ar", "fixed_budget": None,
         "output_ids": [1], "num_output_tokens": 10, "max_new_tokens": 2048},
        {"model": "m", "dataset": "gsm8k", "sample_idx": 0, "orig_idx": 5, "turn": 0, "mode": "bastion", "fixed_budget": None,
         "output_ids": [2], "num_output_tokens": 2048, "max_new_tokens": 2048},
    ]
    golds = {"gsm8k": ["x"] * 5 + ["18"]}
    decode = lambda model, ids: "\\boxed{18}" if ids == [1] else "I think 18"  # noqa: E731
    summary, per = score(recs, golds, decode)
    by = {s["mode"]: s for s in summary}
    assert by["ar"]["accuracy"] == 1.0 and by["bastion"]["accuracy"] == 0.0
    assert by["bastion"]["n_no_boxed"] == 1 and by["bastion"]["n_hit_max_new_tokens"] == 1
