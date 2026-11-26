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
_SPACING_MACRO_RE = re.compile(
    r"(?<!\\)\\(?:qquad|quad)(?![A-Za-z])|(?<!\\)\\[,;!: ]"
)
_TEXT_MACRO_RE = re.compile(r"\\(?:textbf|mathbf|mathrm|text|mbox)(?![A-Za-z])\s*\{")
_DEGREE_RE = re.compile(r"\^\s*\{\s*\\circ\s*\}|\^\s*\\circ(?![A-Za-z])|°")
_FRAC_MACRO_RE = re.compile(r"\\frac(?![A-Za-z])")
_SQRT_MACRO_RE = re.compile(r"\\sqrt(?![A-Za-z])")
_PI_MACRO_RE = re.compile(r"\\pi")
_MUL_MACRO_RE = re.compile(r"\\(?:cdot(?!s)|times)")
_UNSUPPORTED_SYMPY_TOKENS = (r"\begin", r"\end", "=", "<", ">", ";")
_MAX_SYMPY_LEN = 128
_REL_TOL = 1e-6
_ABS_TOL = 1e-9


def _find_matching_brace(text: str, open_idx: int) -> int | None:
    """Return index of the matching ``}`` for ``text[open_idx] == '{'``."""
    depth = 1
    i = open_idx + 1
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
                return i
        i += 1
    return None


def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the last balanced ``\\boxed{...}`` expression."""
    marker = r"\boxed{"
    start_idx = text.rfind(marker)
    if start_idx == -1:
        return None

    open_idx = start_idx + len(marker) - 1
    close_idx = _find_matching_brace(text, open_idx)
    if close_idx is None:
        return None
    return text[open_idx + 1 : close_idx].strip()


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


def _unwrap_text_macros(s: str) -> str:
    """Iteratively unwrap LaTeX text-formatting macros to their inner text."""
    pos = 0
    while True:
        match = _TEXT_MACRO_RE.search(s, pos)
        if match is None:
            return s
        open_idx = match.end() - 1
        close_idx = _find_matching_brace(s, open_idx)
        if close_idx is None:
            pos = match.end()
            continue
        inner = s[open_idx + 1 : close_idx]
        s = s[: match.start()] + inner + s[close_idx + 1 :]
        pos = match.start()


def _skip_whitespace(s: str, idx: int) -> int:
    while idx < len(s) and s[idx].isspace():
        idx += 1
    return idx


def _read_braced_group(s: str, start: int) -> tuple[str, int] | None:
    """Read a balanced ``{...}`` group starting at or after ``start``."""
    idx = _skip_whitespace(s, start)
    if idx >= len(s) or s[idx] != "{":
        return None
    close_idx = _find_matching_brace(s, idx)
    if close_idx is None:
        return None
    return s[idx + 1 : close_idx], close_idx + 1


def _read_latex_arg(s: str, start: int) -> tuple[str, int] | None:
    """Read one LaTeX argument: braced group, single macro, or single alnum char."""
    idx = _skip_whitespace(s, start)
    n = len(s)
    if idx >= n:
        return None
    if s[idx] == "{":
        return _read_braced_group(s, idx)
    if s[idx] == "\\" and idx + 1 < n:
        if s[idx + 1].isalpha():
            end = idx + 2
            while end < n and s[end].isalpha():
                end += 1
            return s[idx:end], end
        return s[idx : idx + 2], idx + 2
    if s[idx].isalnum():
        return s[idx], idx + 1
    return None


def _normalize_shorthand_macros(s: str) -> str:
    """Normalize shorthand ``\\frac`` and ``\\sqrt`` arguments to braced form."""
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s.startswith(r"\frac", i) and (i + 5 == n or not s[i + 5].isalpha()):
            arg1 = _read_latex_arg(s, i + 5)
            if arg1 is not None:
                arg2 = _read_latex_arg(s, arg1[1])
                if arg2 is not None:
                    norm_num = _normalize_shorthand_macros(arg1[0])
                    norm_den = _normalize_shorthand_macros(arg2[0])
                    out.append(f"\\frac{{{norm_num}}}{{{norm_den}}}")
                    i = arg2[1]
                    continue
            out.append(r"\frac")
            i += 5
            continue

        if s.startswith(r"\sqrt", i) and (i + 5 == n or not s[i + 5].isalpha()):
            j = _skip_whitespace(s, i + 5)
            opt = ""
            if j < n and s[j] == "[":
                close_bracket = s.find("]", j + 1)
                if close_bracket != -1:
                    opt = s[j : close_bracket + 1]
                    j = close_bracket + 1
            arg = _read_latex_arg(s, j)
            if arg is not None:
                norm_arg = _normalize_shorthand_macros(arg[0])
                out.append(f"\\sqrt{opt}{{{norm_arg}}}")
                i = arg[1]
                continue
            out.append(r"\sqrt")
            i += 5
            continue

        out.append(s[i])
        i += 1
    return "".join(out)


def _canonicalize_latex(expr: str) -> str:
    """Canonicalize LaTeX spacing, text wrappers, shorthand macros, and degrees."""
    s = normalize_math_answer(expr)
    s = _unwrap_text_macros(s)
    s = _SPACING_MACRO_RE.sub("", s)
    s = _DEGREE_RE.sub("", s)
    s = _normalize_shorthand_macros(s)
    return re.sub(r"\s+", "", s)


def _split_top_level_sequence(canon: str) -> list[str] | None:
    """Split a top-level ``(...)`` or ``[...]`` tuple/interval on top-level commas."""
    n = len(canon)
    if n < 2 or (canon[0], canon[-1]) not in {("(", ")"), ("[", "]")}:
        return None

    depth = 0
    parts: list[str] = []
    part_start = 1
    for i, ch in enumerate(canon):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return None
            if depth == 0 and i < n - 1:
                return None
        elif ch == "," and depth == 1 and 0 < i < n - 1:
            parts.append(canon[part_start:i])
            part_start = i + 1

    if depth != 0:
        return None
    parts.append(canon[part_start : n - 1])
    if len(parts) < 2 or any(part == "" for part in parts):
        return None
    return parts


def _replace_pi_macro(match: re.Match[str]) -> str:
    text = match.string
    prefix = " " if match.start() > 0 and text[match.start() - 1].isalnum() else ""
    suffix = (
        " "
        if match.end() < len(text)
        and (text[match.end()].isalnum() or text[match.end()] == "\\")
        else ""
    )
    return f"{prefix}pi{suffix}"


def _latex_to_sympy_str(canon: str) -> str | None:
    """Convert canonicalized LaTeX into a SymPy-parseable expression string."""
    if not canon or len(canon) > _MAX_SYMPY_LEN:
        return None
    if any(tok in canon for tok in _UNSUPPORTED_SYMPY_TOKENS):
        return None

    s = canon
    while True:
        match = _FRAC_MACRO_RE.search(s)
        if match is None:
            break
        num_pair = _read_braced_group(s, match.end())
        if num_pair is None:
            return None
        num, after_num = num_pair
        den_pair = _read_braced_group(s, after_num)
        if den_pair is None:
            return None
        den, after_den = den_pair
        s = s[: match.start()] + f"(({num})/({den}))" + s[after_den:]

    s = _PI_MACRO_RE.sub(_replace_pi_macro, s)
    s = re.sub(r"(?<=[A-Za-z0-9])\\sqrt(?![A-Za-z])", " sqrt", s)
    s = _SQRT_MACRO_RE.sub("sqrt", s)
    s = _MUL_MACRO_RE.sub("*", s)
    s = s.replace("{", "(").replace("}", ")")
    if "\\" in s:
        return None
    return s


def _sympy_equivalent(canon_a: str, canon_b: str) -> bool:
    """Check algebraic or numeric equivalence via lazy-imported SymPy."""
    str_a = _latex_to_sympy_str(canon_a)
    str_b = _latex_to_sympy_str(canon_b)
    if str_a is None or str_b is None:
        return False

    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except ImportError:
        return False

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    local_dict = {"pi": sympy.pi, "sqrt": sympy.sqrt}
    try:
        expr_a = parse_expr(str_a, local_dict=local_dict, transformations=transformations)
        expr_b = parse_expr(str_b, local_dict=local_dict, transformations=transformations)
        if not isinstance(expr_a, sympy.Basic) or not isinstance(expr_b, sympy.Basic):
            return False
        if sympy.simplify(expr_a - expr_b) == 0:
            return True
        if not expr_a.free_symbols and not expr_b.free_symbols:
            val_a = complex(expr_a.evalf())
            val_b = complex(expr_b.evalf())
            if (
                math.isfinite(val_a.real)
                and math.isfinite(val_a.imag)
                and math.isfinite(val_b.real)
                and math.isfinite(val_b.imag)
            ):
                return math.isclose(
                    val_a.real, val_b.real, rel_tol=_REL_TOL, abs_tol=_ABS_TOL
                ) and math.isclose(
                    val_a.imag, val_b.imag, rel_tol=_REL_TOL, abs_tol=_ABS_TOL
                )
    except Exception:
        return False
    return False


def _expressions_equivalent(norm_gen: str, norm_gt: str) -> bool:
    """Check equivalence via exact, canonical, numeric, sequence, and SymPy tiers."""
    if norm_gen != "" and norm_gen == norm_gt:
        return True

    canon_gen = _canonicalize_latex(norm_gen)
    canon_gt = _canonicalize_latex(norm_gt)
    if canon_gen != "" and canon_gen == canon_gt:
        return True

    gen_num = _parse_numeric(norm_gen)
    if gen_num is None:
        gen_num = _parse_numeric(canon_gen)
    gt_num = _parse_numeric(norm_gt)
    if gt_num is None:
        gt_num = _parse_numeric(canon_gt)
    if gen_num is not None and gt_num is not None:
        return math.isclose(gen_num, gt_num, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)

    seq_gen = _split_top_level_sequence(canon_gen)
    seq_gt = _split_top_level_sequence(canon_gt)
    if seq_gen is not None and seq_gt is not None:
        if canon_gen[0] != canon_gt[0] or len(seq_gen) != len(seq_gt):
            return False
        return all(
            _expressions_equivalent(part_gen, part_gt)
            for part_gen, part_gt in zip(seq_gen, seq_gt)
        )

    return _sympy_equivalent(canon_gen, canon_gt)


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
    return _expressions_equivalent(norm_gen, norm_gt), extracted
