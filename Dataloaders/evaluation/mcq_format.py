"""Answer-only multiple-choice prompt format and answer canonicalizer (format ID ``mcq.answer_only.v1``).

Used for MMLU-Pro and the synthetic MCQ set. The canonicalizer maps every
completion to one option label or to ``None`` (null outcome); in SCPS it is
the evaluator that defines the answer-class targets.
"""

from __future__ import annotations

import re
import string
from typing import Optional, Sequence

LABELS = tuple(string.ascii_uppercase)

ANSWER_ONLY_INSTRUCTION = (
    "The following is a multiple choice question. "
    "Reply with only the letter of the correct option."
)


def option_labels(num_options: int) -> tuple:
    if not 1 <= num_options <= len(LABELS):
        raise ValueError(f"num_options must be in [1, {len(LABELS)}], got {num_options}")
    return LABELS[:num_options]


FORMAT_ID = "mcq.answer_only.v1"


def format_mcq(question: str, options: Sequence[str]) -> str:
    """Answer-only MCQ user message."""
    labels = option_labels(len(options))
    lines = [ANSWER_ONLY_INSTRUCTION, "", f"Question: {question.strip()}", "Options:"]
    lines += [f"{lab}. {str(opt).strip()}" for lab, opt in zip(labels, options)]
    return "\n".join(lines)


# Option letters must be upper case; only the keywords are case-insensitive,
# so "the answer is a bit unclear" does not parse as option A.
_PATTERNS = [
    # "the answer is (B)", "Answer: B", "answer is B."
    re.compile(r"(?i:answer)\s*(?i:is)?\s*:?\s*\(?\s*([A-Z])\b"),
    # leading "(B)" followed by anything
    re.compile(r"^\s*\(([A-Z])\)"),
    # leading "B", "B.", "B)", "B:" (a bare letter, not the start of a word)
    re.compile(r"^\s*([A-Z])\s*(?:[\.\):]|$)"),
    # "option B"
    re.compile(r"(?i:option)\s+\(?([A-Z])\b"),
]


def canonicalize_answer(text: str, labels: Sequence[str]) -> Optional[str]:
    """Map a completion to an option label, or ``None`` if none is found.

    Rules are applied in order; the first rule that yields a valid label
    wins. For the "answer is X" rule the last match is used, so a trace
    that reconsiders ends on its final choice.
    """
    valid = set(labels)
    if not text:
        return None
    first = _PATTERNS[0].findall(text)
    for cand in reversed(first):
        cand = cand.upper()
        if cand in valid:
            return cand
    for pat in _PATTERNS[1:]:
        m = pat.search(text)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
    return None


__all__ = ["LABELS", "FORMAT_ID", "ANSWER_ONLY_INSTRUCTION", "option_labels", "format_mcq", "canonicalize_answer"]
