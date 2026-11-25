"""Tests for the pure helpers in Dataloaders.evaluation.math_reasoning.

Only the extraction/normalization/equivalence helpers are tested here without
network access. ``load_math500`` and ``load_math_train`` need the HF Hub, so
their tests are skipped unless ``RUN_NETWORK_TESTS=1`` is set.
"""

from __future__ import annotations

import builtins
import os

import pytest

from Dataloaders.evaluation.math_reasoning import (
    MATH_TRAIN_CONFIGS,
    HF_PATH_MATH_TRAIN,
    extract_boxed,
    is_equiv,
    load_math500,
    load_math_train,
    normalize_answer,
)

NETWORK = pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1",
    reason="downloads data from the Hugging Face Hub; set RUN_NETWORK_TESTS=1 to run",
)


# --------------------------------------------------------------- extract_boxed

def test_extract_boxed_simple():
    assert extract_boxed("so the answer is \\boxed{42}.") == "42"


def test_extract_boxed_nested_braces():
    assert extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_boxed_multiple_uses_last():
    text = "first \\boxed{1} then \\boxed{2} finally \\boxed{3}"
    assert extract_boxed(text) == "3"


def test_extract_boxed_fbox():
    assert extract_boxed("\\fbox{7}") == "7"


def test_extract_boxed_last_marker_wins_across_kinds():
    assert extract_boxed("\\boxed{1} and \\fbox{2}") == "2"
    assert extract_boxed("\\fbox{1} and \\boxed{2}") == "2"


def test_extract_boxed_missing_marker():
    assert extract_boxed("no boxed answer here") is None


def test_extract_boxed_unbalanced():
    assert extract_boxed("\\boxed{1") is None


def test_extract_boxed_no_brace_form():
    assert extract_boxed("\\boxed 5") == "5"


def test_extract_boxed_no_brace_form_stops_at_dollar():
    assert extract_boxed("\\boxed 5$ is the answer") == "5"


def test_extract_boxed_trailing_marker_no_content():
    assert extract_boxed("the answer is \\boxed") is None


def test_extract_boxed_none_and_empty():
    assert extract_boxed(None) is None
    assert extract_boxed("") is None


def test_extract_boxed_strips_whitespace():
    assert extract_boxed("\\boxed{ 14k }") == "14k"


# ------------------------------------------------------------- normalize_answer

@pytest.mark.parametrize(
    "raw, expected",
    [
        (" 42 ", "42"),
        ("$42$", "42"),
        ("\\left(1,2\\right)", "(1,2)"),
        ("1\\!\\,\\;2", "12"),
        ("\\text{odd}", "odd"),
        ("\\dfrac{1}{2}", "\\frac{1}{2}"),
        ("\\tfrac{1}{2}", "\\frac{1}{2}"),
        ("45^\\circ", "45"),
        ("45^{\\circ}", "45"),
        ("3.", "3"),
        ("0.5", ".5"),
        (".5", ".5"),
        ("0.25", ".25"),
        ("x=5", "5"),
        ("1,000", "1000"),
        ("1{,}000", "1000"),
        ("(1,2)", "(1,2)"),
        ("\\frac12", "\\frac{1}{2}"),
        ("5.0", "5"),
        ("50\\%", "50"),
        ("\\$7", "7"),
        ("5\\text{ cm}", "5cm"),
        (None, ""),
    ],
)
def test_normalize_answer_cases(raw, expected):
    assert normalize_answer(raw) == expected


def test_normalize_answer_idempotent():
    """Normalizing twice should not change an already-normalized answer."""
    once = normalize_answer("$ \\left( 0.5 \\right) $")
    assert normalize_answer(once) == once


# ------------------------------------------------------------------- is_equiv

def test_is_equiv_none_prediction_is_false():
    assert is_equiv(None, "42") is False


def test_is_equiv_none_gold_is_false():
    assert is_equiv("42", None) is False


def test_is_equiv_normalized_match_without_math_verify(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "math_verify":
            raise ImportError("math_verify not installed (test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert is_equiv("x = 5", "5") is True
    assert is_equiv("\\dfrac12", "\\frac{1}{2}") is True


def test_is_equiv_fallback_equal(monkeypatch):
    """Force the normalize_answer fallback path by hiding math_verify."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "math_verify":
            raise ImportError("math_verify not installed (test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert is_equiv("0.5", ".5") is True
    assert is_equiv("$42$", "42") is True


def test_is_equiv_fallback_not_equal(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "math_verify":
            raise ImportError("math_verify not installed (test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert is_equiv("41", "42") is False


# --------------------------------------------------------------- network tests

@NETWORK
def test_load_math500_full_count():
    examples = load_math500()
    assert len(examples) == 500


@NETWORK
def test_load_math500_subset_deterministic_and_sized():
    a = load_math500(n=10, seed=1)
    b = load_math500(n=10, seed=1)
    assert len(a) == 10
    assert [e.uid for e in a] == [e.uid for e in b]


@NETWORK
def test_load_math_train_raw_row_count():
    from datasets import load_dataset

    total = sum(
        len(load_dataset(HF_PATH_MATH_TRAIN, config, split="train"))
        for config in MATH_TRAIN_CONFIGS
    )
    assert total == 7500


@NETWORK
def test_load_math_train_subset_sized_and_tagged():
    examples = load_math_train(n=20, seed=2)
    assert len(examples) == 20
    assert all(e.subject in MATH_TRAIN_CONFIGS for e in examples)
    assert all(e.uid.startswith("train/") for e in examples)


@NETWORK
def test_load_math_train_skips_braceless_boxed_rows():
    pool = load_math_train(n=7498, seed=0)
    assert len(pool) == 7498
    assert "train/algebra/888" not in {e.uid for e in pool}
