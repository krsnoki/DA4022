"""Tests for mathematical answer extraction, normalization, and verification."""

from __future__ import annotations

import pytest

from Adaptive_Stopping.methods.dipa.verifier import (
    extract_boxed_answer,
    normalize_math_answer,
    verify_math_answer,
)


class TestExtractBoxedAnswer:
    """Tests for ``extract_boxed_answer``."""

    def test_simple_boxed(self) -> None:
        assert extract_boxed_answer(r"Therefore the answer is \boxed{42}.") == "42"

    def test_multiple_boxed_returns_last(self) -> None:
        text = r"First try \boxed{10}, wait actually \boxed{42}."
        assert extract_boxed_answer(text) == "42"

    def test_nested_braces(self) -> None:
        assert extract_boxed_answer(r"Result: \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
        assert (
            extract_boxed_answer(r"Result: \boxed{\frac{\sqrt{2}}{2}}")
            == r"\frac{\sqrt{2}}{2}"
        )

    def test_escaped_braces(self) -> None:
        assert extract_boxed_answer(r"The set is \boxed{\{1, 2\}}.") == r"\{1, 2\}"

    def test_unclosed_boxed_returns_none(self) -> None:
        assert extract_boxed_answer(r"Incomplete \boxed{123") is None

    def test_missing_boxed_returns_none(self) -> None:
        assert extract_boxed_answer("The answer is 42.") is None

    def test_empty_boxed_returns_empty_string(self) -> None:
        assert extract_boxed_answer(r"\boxed{}") == ""


class TestNormalizeMathAnswer:
    """Tests for ``normalize_math_answer``."""

    def test_strips_surrounding_dollar_signs(self) -> None:
        assert normalize_math_answer("  $42$  ") == "42"
        assert normalize_math_answer("$$x + 1$$") == "x + 1"

    def test_strips_left_and_right_without_corrupting_macros(self) -> None:
        assert normalize_math_answer(r"\left(\frac{1}{2}\right)") == r"(\frac{1}{2})"
        assert normalize_math_answer(r"\leftarrow") == r"\leftarrow"

    def test_converts_dfrac_and_tfrac_to_frac(self) -> None:
        assert (
            normalize_math_answer(r"\dfrac{1}{2} + \tfrac{3}{4}")
            == r"\frac{1}{2} + \frac{3}{4}"
        )

    def test_strips_thousands_commas_without_corrupting_tuples(self) -> None:
        assert normalize_math_answer("1,000") == "1000"
        assert normalize_math_answer("1,234,567") == "1234567"
        assert normalize_math_answer("(1, 2)") == "(1, 2)"
        assert normalize_math_answer("(1,2)") == "(1,2)"

    def test_strips_trailing_period(self) -> None:
        assert normalize_math_answer("42.") == "42"
        assert normalize_math_answer("3.5..") == "3.5"


class TestVerifyMathAnswer:
    """Tests for ``verify_math_answer``."""

    @pytest.mark.parametrize(
        ("generation", "ground_truth", "expected_ok", "expected_extracted"),
        [
            (r"Answer: \boxed{42}", "42", True, "42"),
            (r"Answer: \boxed{42}", r"GT: \boxed{42}", True, "42"),
            (
                r"\boxed{$\left(\dfrac{1}{2}\right)$}",
                r"(\frac{1}{2})",
                True,
                r"$\left(\dfrac{1}{2}\right)$",
            ),
            (
                r"\boxed{\left(\dfrac{1}{2}\right).}",
                r"(\frac{1}{2})",
                True,
                r"\left(\dfrac{1}{2}\right).",
            ),
            (r"\boxed{\frac{1}{2}}", "0.5", True, r"\frac{1}{2}"),
            (r"\boxed{0.5}", r"\frac{1}{2}", True, "0.5"),
            (r"\boxed{-0.25}", r"-\frac{1}{4}", True, "-0.25"),
            (r"\boxed{-\frac{1}{4}}", "-0.25", True, r"-\frac{1}{4}"),
            (r"\boxed{1,000}", "1000", True, "1,000"),
            (r"\boxed{1000}", "1,000", True, "1000"),
            (r"\boxed{1/3}", "0.3333333", True, "1/3"),
            (r"\boxed{0.3333333}", "1/3", True, "0.3333333"),
            (r"\boxed{\dfrac{6}{8}}", "0.75", True, r"\dfrac{6}{8}"),
            (r"\boxed{41}", "42", False, "41"),
            ("No box here 42", "42", False, None),
            (r"\boxed{\frac{1}{0}}", "inf", False, r"\frac{1}{0}"),
            (r"\boxed{1/0}", "0", False, "1/0"),
            (r"\boxed{}", "", False, ""),
            (r"\boxed{   }", "", False, ""),
        ],
    )
    def test_verify_math_answer_cases(
        self,
        generation: str,
        ground_truth: str,
        expected_ok: bool,
        expected_extracted: str | None,
    ) -> None:
        ok, extracted = verify_math_answer(generation, ground_truth)
        assert ok is expected_ok
        assert extracted == expected_extracted
