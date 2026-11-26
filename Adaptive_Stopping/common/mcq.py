"""MCQ prompts for adaptive-compute methods.

The base prompt and the canonicalizer live in
``Dataloaders/evaluation/mcq_format.py``. This module only adds the
method-specific steering hint used by targeted (answer-diverse) actions.
"""

from __future__ import annotations

from typing import Optional, Sequence

from Dataloaders.evaluation.mcq_format import (  # noqa: F401  (re-exported)
    ANSWER_ONLY_INSTRUCTION,
    LABELS,
    canonicalize_answer,
    format_mcq,
    option_labels,
)

TARGET_HINT = "Before answering, carefully check whether option {target} is correct."


def build_mcq_prompt(question: str, options: Sequence[str], target: Optional[str] = None) -> str:
    """Base answer-only prompt, plus the targeted-action hint when ``target`` is given.

    The hint is our approximation of the paper's targeted action generator,
    whose exact prompt is not published.
    """
    base = format_mcq(question, options)
    if target is None:
        return base
    labels = option_labels(len(options))
    if target not in labels:
        raise ValueError(f"target {target!r} not in labels {labels}")
    return base + "\n\n" + TARGET_HINT.format(target=target)


__all__ = ["LABELS", "ANSWER_ONLY_INSTRUCTION", "TARGET_HINT", "option_labels", "build_mcq_prompt", "canonicalize_answer"]
