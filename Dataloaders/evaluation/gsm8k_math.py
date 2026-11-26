"""Loaders for math-reasoning benchmarks: GSM8K, MATH (train) and MATH-500.

The loader turns a pinned Hugging Face split into a deterministic list of
``MathExample`` records and builds the chat prompt for a reasoning model. It
knows nothing about the method consuming it: baseline and method runs call the
same function with the same ``(dataset_id, num_samples, seed, offset)``.

Registry IDs (see ``Dataloaders/registry.yaml``):
``text.gsm8k.train``, ``text.gsm8k.test``, ``text.math.train``, ``text.math500.test``.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Iterable, Sequence

from Dataloaders.registry import get_dataset_entry

# Prompt templates. ``math_cot_boxed.v1`` appends the directive recommended on
# the DeepSeek-R1 model cards to the raw question; no system prompt is used.
PROMPTS = {
    "math_cot_boxed.v1": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    "raw_question.v1": "{question}",
}


@dataclasses.dataclass(frozen=True)
class MathExample:
    uid: str  # "<dataset_id>:<row index in the pinned split>"
    dataset_id: str
    index: int
    question: str
    gold: str  # GSM8K: final number; MATH: LaTeX answer string
    gold_format: str  # "number" | "latex"
    meta: dict = dataclasses.field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Answer-field parsing (pure functions, unit-tested on canned strings)
# --------------------------------------------------------------------------- #

def gsm8k_final_answer(answer_field: str) -> str:
    """Return the number after ``####`` in a GSM8K ``answer`` field."""
    if "####" not in answer_field:
        raise ValueError("GSM8K answer field has no '####' marker")
    return answer_field.split("####")[-1].strip().replace(",", "")


def last_boxed_content(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (or ``\\fbox{...}``) with balanced braces.

    The brace-less form ``\\boxed 2`` (used by 2 of the 7,500 MATH train
    solutions) is accepted too: the answer is the following run of
    non-space characters, up to a closing ``$``.
    """
    import re

    m = list(re.finditer(r"\\boxed\s+([^\s${}]+)", text))
    braced = text.rfind("\\boxed{")
    if m and m[-1].start() > braced:
        return m[-1].group(1)
    for marker in ("\\boxed{", "\\fbox{"):
        start = text.rfind(marker)
        if start == -1:
            continue
        i = start + len(marker)
        depth = 1
        out = []
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(out)
            out.append(ch)
            i += 1
        return None  # unbalanced
    return None


def records_to_examples(entry: dict, records: Sequence[dict]) -> list[MathExample]:
    """Convert raw rows of a pinned split into ``MathExample`` objects (row order kept)."""
    dataset_id = entry["id"]
    out = []
    for idx, row in enumerate(records):
        if dataset_id.startswith("text.gsm8k"):
            question, gold = row["question"], gsm8k_final_answer(row["answer"])
            meta = {}
        elif dataset_id == "text.math.train":
            question = row["problem"]
            gold = last_boxed_content(row["solution"])
            if gold is None:
                raise ValueError(f"{dataset_id} row {idx}: no \\boxed answer in solution")
            meta = {"level": row.get("level"), "subject": row.get("type")}
        elif dataset_id == "text.math500.test":
            question, gold = row["problem"], row["answer"]
            meta = {"level": row.get("level"), "subject": row.get("subject"), "unique_id": row.get("unique_id")}
        else:
            raise KeyError(f"no converter for dataset id {dataset_id!r}")
        out.append(MathExample(
            uid=f"{dataset_id}:{idx}", dataset_id=dataset_id, index=idx, question=question,
            gold=str(gold).strip(), gold_format=entry["gold_format"], meta=meta,
        ))
    return out


def sample_indices(n_total: int, num_samples: int | None, seed: int, offset: int = 0) -> list[int]:
    """Seeded permutation of ``range(n_total)``, sliced to ``[offset, offset + num_samples)``."""
    if num_samples is not None and num_samples < 0:
        raise ValueError("num_samples must be >= 0")
    if offset < 0:
        raise ValueError("offset must be >= 0")
    perm = list(range(n_total))
    random.Random(seed).shuffle(perm)
    end = n_total if num_samples is None else offset + num_samples
    if end > n_total:
        raise ValueError(f"requested rows [{offset}, {end}) but the split has {n_total}")
    return perm[offset:end]


# --------------------------------------------------------------------------- #
# Loading from the Hub (pinned revisions)
# --------------------------------------------------------------------------- #

def _read_file(path: str) -> list[dict]:
    import json

    if path.endswith(".parquet"):
        import pandas as pd  # lazy: unit tests do not need it

        return pd.read_parquet(path).to_dict("records")
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    raise ValueError(f"unsupported data file {path}")


def _load_records(entry: dict) -> list[dict]:
    """Read the exact files listed in the registry entry at the pinned revision (cached, works offline)."""
    from huggingface_hub import hf_hub_download

    rows: list[dict] = []
    for fname in entry["files"]:
        local = hf_hub_download(entry["hf_path"], fname, repo_type="dataset", revision=entry["version"])
        rows.extend(_read_file(local))
    expected = entry.get("num_examples")
    if isinstance(expected, int) and len(rows) != expected:
        raise ValueError(f"{entry['id']}: registry says {expected} rows, loaded {len(rows)}")
    return rows


def load_math_reasoning(dataset_id: str, num_samples: int | None = None, seed: int | None = None,
                        offset: int = 0) -> list[MathExample]:
    """Load a deterministic subset of a registered math-reasoning split.

    ``seed`` defaults to the registry seed. Rows are a seeded permutation of the
    pinned split, so two calls with the same arguments return the same examples.
    """
    entry = get_dataset_entry(dataset_id)
    seed = entry["seed"] if seed is None else seed
    examples = records_to_examples(entry, _load_records(entry))
    return [examples[i] for i in sample_indices(len(examples), num_samples, seed, offset)]


# --------------------------------------------------------------------------- #
# Prompt construction (kept here, not in the method code)
# --------------------------------------------------------------------------- #

def build_user_prompt(example: MathExample, prompt_id: str = "math_cot_boxed.v1") -> str:
    return PROMPTS[prompt_id].format(question=example.question.strip())


def build_messages(example: MathExample, prompt_id: str = "math_cot_boxed.v1") -> list[dict]:
    """Chat messages for a reasoning model (single user turn, no system prompt)."""
    return [{"role": "user", "content": build_user_prompt(example, prompt_id)}]


def check_disjoint(a: Iterable[MathExample], b: Iterable[MathExample]) -> None:
    """Raise if two example lists share a split row or an identical question."""
    a, b = list(a), list(b)
    shared_uid = {x.uid for x in a} & {x.uid for x in b}
    shared_q = {x.question.strip() for x in a} & {x.question.strip() for x in b}
    if shared_uid or shared_q:
        raise ValueError(f"leakage: {len(shared_uid)} shared rows, {len(shared_q)} shared questions")
