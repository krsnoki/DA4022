"""Verifiable mathematical answer extraction, normalization, and equivalence checks."""

from __future__ import annotations

from fractions import Fraction
import math
import re

_LEFT_RIGHT_RE = re.compile(r"\\(?:left|right)(?![A-Za-z])")
_ALT_FRAC_RE = re.compile(r"\\(?:d|t)frac(?![A-Za-z])")
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_FRAC_EXPR_RE = re.compile(
    r"([+-]?)\s*\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}"
)


def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the last balanced ``\\boxed{...}`` expression."""
    marker = r"\boxed{"
    start_idx = text.rfind(marker)
    if start_idx == -1:
        return None

    content_start = start_idx + len(marker)
    depth = 1
    i = content_start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            # Skip escaped characters so \{ or \} do not alter brace depth.
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:i].strip()
        i += 1
    return None


def normalize_math_answer(answer: str) -> str:
    """Normalize whitespace, math delimiters, fraction macros, and trailing periods."""
    s = answer.strip()
    s = s.strip("$").strip()
    s = _LEFT_RIGHT_RE.sub("", s)
    s = _ALT_FRAC_RE.sub(lambda _: r"\frac", s)
    s = _THOUSANDS_COMMA_RE.sub("", s)
    s = s.rstrip(".").strip()
    return s


def _parse_fraction_pair(num_str: str, den_str: str) -> float | None:
    """Parse numerator/denominator strings into a finite float via Fraction or float."""
    num_clean = num_str.strip()
    den_clean = den_str.strip()
    try:
        val = float(Fraction(num_clean) / Fraction(den_clean))
    except (ValueError, ZeroDivisionError, OverflowError):
        try:
            val = float(num_clean) / float(den_clean)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
    return val if math.isfinite(val) else None


def _parse_numeric(expr: str) -> float | None:
    """Parse integers, decimals, ``a/b``, and ``\\frac{a}{b}`` into a finite float."""
    s = expr.strip()
    if not s:
        return None

    frac_match = _FRAC_EXPR_RE.fullmatch(s)
    if frac_match is not None:
        sign = -1.0 if frac_match.group(1) == "-" else 1.0
        val = _parse_fraction_pair(frac_match.group(2), frac_match.group(3))
        return sign * val if val is not None else None

    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            return _parse_fraction_pair(parts[0], parts[1])
        return None

    try:
        val = float(s)
    except (ValueError, OverflowError):
        return None
    return val if math.isfinite(val) else None


def verify_math_answer(
    generation_text: str,
    ground_truth: str,
) -> tuple[bool, str | None]:
    """Check whether the boxed answer in ``generation_text`` matches ``ground_truth``."""
    extracted = extract_boxed_answer(generation_text)
    if extracted is None:
        return False, None

    gt_boxed = extract_boxed_answer(ground_truth)
    gt_raw = gt_boxed if gt_boxed is not None else ground_truth

    norm_gen = normalize_math_answer(extracted)
    norm_gt = normalize_math_answer(gt_raw)
    if norm_gen != "" and norm_gen == norm_gt:
        return True, extracted

    gen_num = _parse_numeric(norm_gen)
    gt_num = _parse_numeric(norm_gt)
    if gen_num is not None and gt_num is not None:
        return math.isclose(gen_num, gt_num, rel_tol=1e-6, abs_tol=1e-9), extracted

    return False, extracted
