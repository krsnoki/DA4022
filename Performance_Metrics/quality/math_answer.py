"""Final-answer correctness for math reasoning (GSM8K, MATH, MATH-500) with Math-Verify.

The ST-ES paper scores math answers with Math-Verify
(https://github.com/huggingface/Math-Verify, Apache-2.0). Gold answers are
parsed as a plain expression (GSM8K numbers) or as LaTeX (MATH); predictions
are parsed with LaTeX extraction (``\\boxed{}`` first) and then expression
extraction, as in the authors' released scoring helper.

Only the *answer section* of a reasoning trace is scored: the text after the
last ``</think>`` (or the whole text if the model never closed its thinking).

Windows note: Math-Verify enforces its timeouts with signals (Unix) or a
subprocess pool, and the latter fails in some Windows sandboxes. Timeouts are
therefore disabled by default (``timeouts=False``) on Windows.
"""

from __future__ import annotations

import sys
from functools import lru_cache

THINK_END = "</think>"
_USE_TIMEOUTS = not sys.platform.startswith("win")


def answer_section(text: str, think_end: str = THINK_END) -> str:
    return text.rsplit(think_end, 1)[-1] if think_end in text else text


@lru_cache(maxsize=4096)
def _parse_gold(gold: str, gold_format: str):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse

    timeout = 5 if _USE_TIMEOUTS else None
    if gold_format == "number":
        return parse(gold.replace(",", ""), extraction_config=[ExprExtractionConfig()], parsing_timeout=timeout)
    if gold_format == "latex":
        return parse("$" + gold + "$", extraction_config=[LatexExtractionConfig()], parsing_timeout=timeout)
    raise ValueError(f"unknown gold_format {gold_format!r}")


def is_correct(prediction_text: str, gold: str, gold_format: str) -> bool:
    """True if Math-Verify judges the prediction's final answer equal to ``gold``."""
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    timeout = 5 if _USE_TIMEOUTS else None
    g = _parse_gold(gold, gold_format)
    if not g:
        raise ValueError(f"could not parse gold answer {gold!r}")
    p = parse(answer_section(prediction_text), extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
              parsing_timeout=timeout)
    if not p:
        return False
    return bool(verify(g, p, timeout_seconds=timeout))


SCORER_ID = "math_verify.answer_section.v1"
