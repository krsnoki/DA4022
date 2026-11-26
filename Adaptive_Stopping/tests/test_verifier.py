"""Tests for mathematical answer extraction, normalization, and verification."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

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

    @pytest.mark.parametrize(
        ("generation_answer", "ground_truth"),
        [
            (r"\left( 3, \frac{\pi}{2} \right)", r"(3, \frac{\pi}{2})"),
            ("2x + 1", "2x+1"),
            (r"3\, \pi", r"3\pi"),
            (r"\text{42}", "42"),
            (r"45^\circ", "45"),
            (r"\frac12", r"\frac{1}{2}"),
            (r"\sqrt2", r"\sqrt{2}"),
        ],
    )
    def test_spacing_and_macro_canonicalization(
        self, generation_answer: str, ground_truth: str
    ) -> None:
        ok, extracted = verify_math_answer(
            rf"\boxed{{{generation_answer}}}", ground_truth
        )
        assert ok is True
        assert extracted == generation_answer

        ok_rev, extracted_rev = verify_math_answer(
            rf"\boxed{{{ground_truth}}}", generation_answer
        )
        assert ok_rev is True
        assert extracted_rev == ground_truth

    @pytest.mark.parametrize(
        ("generation_answer", "ground_truth"),
        [
            (r"\frac{\sqrt{2}}{2}", r"\frac{1}{\sqrt{2}}"),
            ("2x + 4", "4 + 2x"),
            (r"\frac{\pi}{2}", r"\pi/2"),
            (r"(3, \frac{\pi}{2})", r"(3, \pi/2)"),
        ],
    )
    def test_tuple_and_symbolic_equivalence(
        self, generation_answer: str, ground_truth: str
    ) -> None:
        pytest.importorskip("sympy")
        ok, extracted = verify_math_answer(
            rf"\boxed{{{generation_answer}}}", ground_truth
        )
        assert ok is True
        assert extracted == generation_answer

    @pytest.mark.parametrize(
        ("generation_answer", "ground_truth"),
        [
            ("x + 1", "x + 2"),
            (r"\sqrt{2}", r"\sqrt{3}"),
            ("(1, 2)", "(2, 1)"),
            ("(1,2)+(3,4)", "(1,2)+(3,5)"),
            ("(1,2)+(3,4)", "(1.0,2)+(3,4)"),
        ],
    )
    def test_negative_symbolic_and_tuple_cases(
        self, generation_answer: str, ground_truth: str
    ) -> None:
        ok, extracted = verify_math_answer(
            rf"\boxed{{{generation_answer}}}", ground_truth
        )
        assert ok is False
        assert extracted == generation_answer

    def test_non_tuple_sum_spacing_and_long_malformed_safety(self) -> None:
        ok_sum, ext_sum = verify_math_answer(
            r"\boxed{(1,2)+(3,4)}", "(1, 2) + (3, 4)"
        )
        assert ok_sum is True
        assert ext_sum == "(1,2)+(3,4)"

        malformed = "(" * 110 + r"\frac{1}" + "x" * 100
        assert len(malformed) > 200
        ok_bad, ext_bad = verify_math_answer(rf"\boxed{{{malformed}}}", "42")
        assert ok_bad is False
        assert ext_bad == malformed

        unclosed = r"\boxed{" + "(" * 210
        ok_unclosed, ext_unclosed = verify_math_answer(unclosed, "42")
        assert ok_unclosed is False
        assert ext_unclosed is None

    def test_verifier_import_does_not_import_sympy(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{existing_pp}"
            if existing_pp
            else str(repo_root)
        )
        script = (
            "import sys\n"
            "import Adaptive_Stopping.methods.dipa.verifier as v\n"
            "assert 'sympy' not in sys.modules, 'sympy imported at module level'\n"
            "ok, _ = v.verify_math_answer(r'\\boxed{42}', '42')\n"
            "assert ok is True\n"
            "assert 'sympy' not in sys.modules, 'sympy imported on fast path'\n"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(repo_root),
            env=env,
            check=True,
        )
