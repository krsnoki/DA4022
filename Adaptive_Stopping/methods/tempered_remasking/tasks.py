"""Prompt formatting and answer grading for the paper's four benchmarks.

GSM8K (Cobbe et al. 2021), MATH-500 (Hendrycks et al. 2021), HumanEval
(Chen et al. 2021) and MBPP (Austin et al. 2021b).

The paper does not publish its prompts or answer-matching code, so these are
documented, conventional choices (``\\boxed{}`` answers for math, a fenced
Python block for code). Dataset *loading* is handled by
``experiments/tempered_remasking/common.py``; this module formats prompts and
grades completions. MATH answers are compared after string normalisation
(LaTeX fractions, units, degrees, ``x =`` prefixes and decimal/fraction
equivalence).

SECURITY: :func:`run_python_tests` executes model-generated code. Run it only
inside a sandbox (container/VM without network or secrets).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import Optional

__all__ = [
    "MATH_SUFFIX",
    "format_math_prompt",
    "format_code_prompt",
    "extract_boxed",
    "extract_last_number",
    "extract_gsm8k_reference",
    "extract_math_answer",
    "normalize_math_answer",
    "grade_gsm8k",
    "grade_math",
    "extract_code",
    "run_python_tests",
    "grade_humaneval",
    "grade_mbpp",
]

MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def format_math_prompt(question: str) -> str:
    return question.strip() + MATH_SUFFIX


def format_code_prompt(task: str, tests: Optional[str] = None) -> str:
    msg = task.strip()
    if tests:
        msg += "\nYour code should pass these tests:\n" + tests.strip()
    return msg + "\nWrite the complete Python solution in a single ```python code block."


# --------------------------------------------------------------------------- #
# Math answers
# --------------------------------------------------------------------------- #
def extract_boxed(text: str) -> Optional[str]:
    """Content of the last ``\\boxed{...}`` (or ``\\fbox{...}``), brace-matched."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        # "\boxed 5" form
        m = re.match(r"\\boxed\s+([^\s$]+)", text[idx:])
        return m.group(1) if m else None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def extract_last_number(text: str) -> Optional[str]:
    nums = _NUM.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _canon_number(s: str) -> Optional[str]:
    s = s.replace(",", "").replace("$", "").strip()
    s = s.rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    if v == int(v):
        return str(int(v))
    return repr(round(v, 6))


def extract_gsm8k_reference(answer_field: str) -> str:
    """GSM8K ground truth: the text after ``####``."""
    ref = answer_field.split("####")[-1].strip()
    return _canon_number(ref) or ref


def grade_gsm8k(completion: str, reference: str) -> tuple:
    """Returns ``(extracted_answer, is_correct)``; prefers ``\\boxed{}``, else last number."""
    boxed = extract_boxed(completion)
    raw = boxed if boxed is not None else extract_last_number(completion)
    if raw is None:
        return None, False
    raw = re.sub(r"\\text\{[^}]*\}", "", raw)
    num = extract_last_number(raw) if _canon_number(raw) is None else raw
    pred = _canon_number(num) if num is not None else None
    ref = _canon_number(reference) or reference
    return pred, pred is not None and pred == ref


def normalize_math_answer(ans: str) -> str:
    """String normalisation in the spirit of the MATH/PRM800K graders."""
    s = ans.strip()
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{\s*([^}]*)\}", r"\1", s)
    for a, b in [
        ("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\:", ""),
        ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("^\\circ", ""), ("^{\\circ}", ""),
        ("\\$", ""), ("$", ""), ("\\%", ""), ("%", ""), (" ", ""), ("\n", ""),
    ]:
        s = s.replace(a, b)
    s = re.sub(r"(degrees?|units?|cm|inches|inch|meters?)$", "", s)
    s = s.rstrip(".")
    # \frac12 -> \frac{1}{2}
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]+)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\d+)", r"\\sqrt{\1}", s)
    if re.fullmatch(r"\.\d+", s):
        s = "0" + s
    # a = 5 -> 5 (single short lhs)
    m = re.fullmatch(r"[a-zA-Z]\s*=\s*(.+)", s)
    if m:
        s = m.group(1)
    c = _canon_number(s)
    if c is not None:
        return c
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m and int(m.group(2)) != 0:
        return _canon_number(str(int(m.group(1)) / int(m.group(2)))) or s
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m and int(m.group(2)) != 0:
        return _canon_number(str(int(m.group(1)) / int(m.group(2)))) or s
    return s


def extract_math_answer(completion: str) -> Optional[str]:
    boxed = extract_boxed(completion)
    return normalize_math_answer(boxed) if boxed is not None else None


def grade_math(completion: str, reference: str) -> tuple:
    """Returns ``(normalised_answer, is_correct)`` against a MATH-500 reference answer."""
    pred = extract_math_answer(completion)
    return pred, pred is not None and pred == normalize_math_answer(reference)


# --------------------------------------------------------------------------- #
# Code
# --------------------------------------------------------------------------- #
def extract_code(completion: str) -> str:
    """Last fenced code block (```python ... ```), else the raw completion."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", completion, flags=re.S)
    if blocks:
        return blocks[-1]
    m = re.search(r"```(?:python|py)?\s*\n(.*)", completion, flags=re.S)  # unterminated fence
    return m.group(1) if m else completion


def run_python_tests(program: str, timeout_s: float = 10.0) -> bool:
    """Run ``program`` in a fresh interpreter; pass iff it exits with status 0.

    Executes untrusted code: sandbox it. Uses ``python -I`` (isolated mode) in a
    temporary working directory and kills it after ``timeout_s``.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            r = subprocess.run(
                [sys.executable, "-I", path], cwd=d, capture_output=True, timeout=timeout_s,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
            )
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def grade_humaneval(completion: str, prompt: str, test: str, entry_point: str, timeout_s: float = 10.0) -> bool:
    """HumanEval: the completion may restate the full function or only its body."""
    code = extract_code(completion)
    if f"def {entry_point}" not in code:
        code = prompt + code
    program = f"{code}\n\n{test}\n\ncheck({entry_point})\n"
    return run_python_tests(program, timeout_s)


def grade_mbpp(completion: str, test_list, test_setup_code: str = "", timeout_s: float = 10.0) -> bool:
    code = extract_code(completion)
    program = "\n".join([test_setup_code or "", code, *test_list]) + "\n"
    return run_python_tests(program, timeout_s)
