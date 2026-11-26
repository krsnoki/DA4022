"""Cheap quality check: GSM8K exact match of the final numeric answer.

The query sent to every agent is the question plus a fixed instruction asking for a
final ``#### <number>`` line. The prediction is the number after the last ``####``,
or failing that the last number in the text. Numbers are compared after removing
thousands separators and normalising "18.0" -> "18".
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

GSM8K_INSTRUCTION = (
    "Solve the following math problem step by step. At the end, write the final "
    "answer as a number on its own line in the form '#### <number>'.\n\n"
)

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def format_gsm8k(question: str) -> str:
    return GSM8K_INSTRUCTION + question


def normalize_number(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if value == value.to_integral_value():
        return str(value.to_integral_value())
    return format(value.normalize(), "f")


def extract_reference(answer: str) -> str | None:
    """Gold answer: the text after ``####`` in the GSM8K solution."""
    if "####" not in answer:
        return None
    return normalize_number(answer.split("####")[-1].strip().split()[0])


def extract_prediction(text: str) -> str | None:
    marked = re.findall(r"####\s*(" + _NUM + ")", text)
    if marked:
        return normalize_number(marked[-1])
    nums = re.findall(_NUM, text)
    return normalize_number(nums[-1]) if nums else None


def exact_match(prediction_text: str, reference_answer: str) -> bool:
    pred, ref = extract_prediction(prediction_text), extract_reference(reference_answer)
    return pred is not None and pred == ref
