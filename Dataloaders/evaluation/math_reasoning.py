"""MATH / MATH-500 evaluation loaders and answer-checking helpers.

Loader contract
----------------
- ``load_math500`` loads the Hugging Face ``HuggingFaceH4/MATH-500`` ``test``
  split (500 rows: ``problem``, ``solution``, ``answer``, ``subject``,
  ``level``, ``unique_id``). The gold answer is already given in ``answer``.
- ``load_math_train`` loads the 7 Hendrycks MATH configs from
  ``EleutherAI/hendrycks_math`` ``train`` split (7,500 rows total across
  configs: ``problem``, ``level``, ``type``, ``solution``). The gold answer
  is extracted from ``solution`` with ``extract_boxed``; rows with no braced
  boxed answer are skipped (7,498 rows kept).
- Both loaders return a list of ``MathExample``. With ``n`` given they return
  a deterministic subset chosen by ``random.Random(seed).sample``, kept in
  original row order; with ``n=None`` (MATH-500 only) the full split is
  returned in its original order.
- Role: MATH-500 for evaluation; a seeded MATH train subset for the
  traces the TERMINATOR probe is trained on. See
  ``Dataloaders/evaluation/README_math500.md``
  for the dataset card, including the train/test overlap note.
- ``extract_boxed``, ``normalize_answer`` and ``is_equiv`` are pure helpers
  used for scoring model generations against the gold answer; they do not
  need the ``datasets`` package and have no network dependency.

Only the ``datasets`` package is needed for the two loaders, and it is
imported lazily so the pure helpers can be used (and tested) without it.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

DATASET_ID_MATH500 = "text.math500.test"
DATASET_ID_MATH_TRAIN = "text.math.train"
HF_PATH_MATH500 = "HuggingFaceH4/MATH-500"
HF_PATH_MATH_TRAIN = "EleutherAI/hendrycks_math"
MATH_TRAIN_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

__all__ = [
    "DATASET_ID_MATH500",
    "DATASET_ID_MATH_TRAIN",
    "HF_PATH_MATH500",
    "HF_PATH_MATH_TRAIN",
    "MATH_TRAIN_CONFIGS",
    "MathExample",
    "load_math500",
    "load_math_train",
    "extract_boxed",
    "normalize_answer",
    "is_equiv",
]


@dataclass(frozen=True)
class MathExample:
    """One MATH problem with its gold answer.

    ``uid`` is the MATH-500 ``unique_id`` for test rows, or
    ``f"train/{config}/{index}"`` (config name and row position within that
    config's train split) for train rows. The TERMINATOR scripts derive the
    per-problem sampling seed from ``uid``, so keep this format stable.
    ``level`` is kept as given by the
    source field (as a string) rather than normalized across the two
    datasets, since MATH-500 and the Hendrycks configs do not format it the
    same way.
    """

    uid: str
    problem: str
    answer: str
    subject: str | None
    level: str | None


def _level_str(value) -> str | None:
    return None if value is None else str(value)


def load_math500(n: int | None = None, seed: int = 0) -> list[MathExample]:
    """Load the MATH-500 ``test`` split (500 rows) as ``MathExample`` records.

    With ``n`` given, returns a deterministic random subset of size ``n``
    (``random.Random(seed).sample`` over row indices, kept sorted so the
    result is in the dataset's original order). With ``n=None`` the full 500
    rows are returned in their original order.
    """
    from datasets import load_dataset  # lazy: optional dependency

    ds = load_dataset(HF_PATH_MATH500, split="test")
    total = len(ds)
    if n is None:
        indices = list(range(total))
    else:
        if not 0 <= n <= total:
            raise ValueError(f"n must be between 0 and {total}, got {n}")
        indices = sorted(random.Random(seed).sample(range(total), n))
    examples = []
    for i in indices:
        row = ds[i]
        examples.append(
            MathExample(
                uid=str(row["unique_id"]),
                problem=row["problem"],
                answer=row["answer"],
                subject=row.get("subject"),
                level=_level_str(row.get("level")),
            )
        )
    return examples


def load_math_train(n: int, seed: int = 0) -> list[MathExample]:
    """Load a deterministic subset of the Hendrycks MATH train split.

    Loads all 7 configs (7,500 rows total), extracts the gold answer from
    each row's ``solution`` with ``extract_boxed``, and skips rows where no
    braced ``\\boxed{...}``/``\\fbox{...}`` answer is found (7,498 rows are
    kept; the 2 rows that use the brace-less ``\\boxed 2`` form are left out
    so the pool, and so the seeded subset, matches the one used for the
    TERMINATOR traces). ``subject`` is set to the config name. The subset
    of size ``n`` is chosen with ``random.Random(seed).sample`` over the
    pooled, filtered rows and kept in the pool's original order (grouped by
    config in ``MATH_TRAIN_CONFIGS`` order, then by row position).
    """
    from datasets import load_dataset  # lazy: optional dependency

    pool: list[MathExample] = []
    for config in MATH_TRAIN_CONFIGS:
        ds = load_dataset(HF_PATH_MATH_TRAIN, config, split="train")
        for i, row in enumerate(ds):
            gold = extract_boxed(row["solution"])
            if gold is None or not _last_box_is_braced(row["solution"]):
                continue
            pool.append(
                MathExample(
                    uid=f"train/{config}/{i}",
                    problem=row["problem"],
                    answer=gold,
                    subject=config,
                    level=_level_str(row.get("level")),
                )
            )
    if not 0 <= n <= len(pool):
        raise ValueError(f"n must be between 0 and {len(pool)}, got {n}")
    chosen = sorted(random.Random(seed).sample(range(len(pool)), n))
    return [pool[i] for i in chosen]


_BOXED_MARKERS = ("\\boxed", "\\fbox")


def _last_box_is_braced(text: str) -> bool:
    """True if the last ``\\boxed``/``\\fbox`` marker is followed by ``{``."""
    pos, length = max((text.rfind(m), len(m)) for m in _BOXED_MARKERS)
    if pos == -1:
        return False
    rest = text[pos + length :].lstrip(" \t")
    return rest.startswith("{")


def extract_boxed(text: str | None) -> str | None:
    """Return the content of the last ``\\boxed{...}`` or ``\\fbox{...}``.

    Braces are brace-matched so a nested ``\\boxed{\\frac{1}{2}}`` returns
    ``\\frac{1}{2}`` whole, with surrounding whitespace stripped. If the
    text is empty or ``None``, has neither marker, or the final one has no
    matching closing brace, returns ``None``. The rare ``\\boxed 5`` form
    (no braces) returns the following run of non-space, non-``$``
    characters instead.
    """
    if not text:
        return None
    best_pos = -1
    best_len = 0
    for marker in _BOXED_MARKERS:
        pos = text.rfind(marker)
        if pos > best_pos:
            best_pos = pos
            best_len = len(marker)
    if best_pos == -1:
        return None

    i = best_pos + best_len
    n = len(text)
    while i < n and text[i] in " \t":
        i += 1
    if i >= n:
        return None
    if text[i] != "{":
        j = i
        while j < n and not text[j].isspace() and text[j] != "$":
            j += 1
        token = text[i:j]
        return token or None

    depth = 0
    content_start = i + 1
    j = i
    while j < n:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:j].strip()
        j += 1
    return None  # unbalanced braces


_SPACING_COMMANDS = ("\\!", "\\,", "\\;", "\\:", "\\ ")
_TEXT_WRAPPER = re.compile(r"^\\text\{(.*)\}$")
_INLINE_TEXT = re.compile(r"\\(?:text|textbf|mathrm|mbox)\{([^{}]*)\}")
_SHORT_LHS = re.compile(r"[a-zA-Z]\w{0,2}=[^=]+")
_THOUSANDS = re.compile(r"-?\d{1,3}(,\d{3})+")


def _fix_fracs(s: str) -> str:
    # \frac12 -> \frac{1}{2}, \frac{1}2 -> \frac{1}{2}, \frac1{2} -> \frac{1}{2}
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)
    return s


def normalize_answer(s: str | None) -> str:
    """Normalize a MATH-style answer string for plain string comparison.

    This is a simple, best-effort transform; it is not a computer-algebra
    system. ``None`` becomes ``""``. Steps, in order:

    1. Strip leading/trailing whitespace.
    2. Drop ``\\left`` / ``\\right`` sizing commands.
    3. Drop LaTeX spacing commands (``\\!``, ``\\,``, ``\\;``, ``\\:``, ``\\ ``).
    4. Drop ``\\$``, ``$`` and all remaining whitespace.
    5. Unwrap a whole-string ``\\text{...}`` wrapper, then any brace-free
       ``\\text{}``, ``\\textbf{}``, ``\\mathrm{}`` or ``\\mbox{}``.
    6. Canonicalize ``\\dfrac`` / ``\\tfrac`` to ``\\frac``.
    7. Drop degree marks (``^\\circ``, ``^{\\circ}``, ``\\circ``) and
       percent signs (``\\%``, ``%``).
    8. Strip one trailing ``.``.
    9. Drop a short variable on the left of a single ``=`` (``x=5`` -> ``5``).
    10. Add braces to shorthand fractions (``\\frac12`` -> ``\\frac{1}{2}``).
    11. Drop ``{,}`` and thousands separators (``1,000`` -> ``1000``).
    12. Drop a ``.0`` suffix on integers (``5.0`` -> ``5``).
    13. Rewrite a leading ``0.`` as ``.`` so ``"0.5"`` and ``".5"`` match.
    """
    if s is None:
        return ""
    out = str(s).strip()
    out = out.replace("\\left", "").replace("\\right", "")
    for cmd in _SPACING_COMMANDS:
        out = out.replace(cmd, "")
    out = out.replace("\\$", "").replace("$", "")
    out = "".join(out.split())

    match = _TEXT_WRAPPER.match(out)
    if match:
        out = match.group(1)
    out = _INLINE_TEXT.sub(r"\1", out)

    out = out.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    out = out.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\circ", "")
    out = out.replace("\\%", "").replace("%", "")

    if out.endswith("."):
        out = out[:-1]
    if _SHORT_LHS.fullmatch(out):
        out = out.split("=", 1)[1]
    out = _fix_fracs(out)
    out = out.replace("{,}", "")
    if _THOUSANDS.fullmatch(out):
        out = out.replace(",", "")
    if out.endswith(".0") and out[:-2].lstrip("-").isdigit():
        out = out[:-2]
    if out.startswith("0.") and len(out) > 2:
        out = out[1:]

    return out


def is_equiv(pred: str | None, gold: str | None) -> bool:
    """True if ``pred`` is equivalent to ``gold``, else ``False``.

    ``pred=None`` (no extracted answer) or ``gold=None`` always returns
    ``False``. Returns ``True`` if the two answers are equal after
    ``normalize_answer``. Otherwise uses ``math_verify``
    (``pip install math-verify``) when it is importable, since it can confirm
    equivalences a string comparison would miss (e.g. ``1/2`` vs ``0.5``, or
    a simplified vs unsimplified fraction). Returns ``False`` when
    ``math_verify`` is not installed or raises on malformed input.
    """
    if pred is None or gold is None:
        return False
    if normalize_answer(pred) == normalize_answer(gold):
        return True
    try:
        from math_verify import parse, verify  # type: ignore
    except ImportError:
        return False

    try:
        gold_parsed = parse(f"${gold}$")
        pred_parsed = parse(f"${pred}$")
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return False