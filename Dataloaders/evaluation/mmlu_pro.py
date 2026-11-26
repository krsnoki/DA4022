"""MMLU-Pro evaluation loader (dataset ID ``text.mmlu_pro.test``).

Source: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro (MIT licence).
Records: ``question_id, question, options (10, padded with "N/A"), answer
(letter), answer_index (0-9), cot_content, category, src``.

Preprocessing ID ``mmlu_pro.strip_na.v1``: trailing ``"N/A"`` padding
options are removed. Only trailing padding is removed, so option letters
and ``answer_index`` keep their original meaning; a record whose gold
option is padding raises an error.

Sampling: ``sample_questions`` sorts by ``question_id`` and draws without
replacement with ``random.Random(seed)``, after removing excluded IDs.
The SCPS paper's 256 held-out IDs (seed 47) are not public, so a sample
drawn here is a different slice of the same split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

HF_DATASET = "TIGER-Lab/MMLU-Pro"
PREPROCESSING_ID = "mmlu_pro.strip_na.v1"
_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def strip_na(record: Dict) -> Dict:
    opts = list(record["options"])
    while opts and str(opts[-1]).strip() == "N/A":
        opts.pop()
    idx = int(record["answer_index"])
    if idx >= len(opts):
        raise ValueError(f"question {record.get('question_id')}: gold option is padding")
    if record.get("answer") is not None and record["answer"] != _LABELS[idx]:
        raise ValueError(f"question {record.get('question_id')}: answer/answer_index mismatch")
    out = dict(record)
    out["options"] = opts
    out["answer"] = _LABELS[idx]
    return out


def load_mmlu_pro(
    split: str = "test",
    revision: Optional[str] = None,
    local_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """Load a split from the Hub (needs ``datasets``) or from a local JSONL/Parquet export."""
    if local_path:
        p = Path(local_path)
        if p.suffix == ".jsonl":
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        elif p.suffix == ".parquet":
            import pandas as pd

            rows = pd.read_parquet(p).to_dict("records")
        else:
            raise ValueError("local_path must be .jsonl or .parquet")
    else:
        from datasets import load_dataset

        rows = list(load_dataset(HF_DATASET, split=split, revision=revision, cache_dir=cache_dir))
    return [strip_na({**r, "options": list(r["options"])}) for r in rows]


def sample_questions(records: Sequence[Dict], n: int, seed: int, exclude_ids: Iterable = ()) -> List[Dict]:
    excl = set(exclude_ids)
    pool = sorted((r for r in records if r["question_id"] not in excl), key=lambda r: r["question_id"])
    if n > len(pool):
        raise ValueError(f"requested {n} questions but only {len(pool)} available")
    return random.Random(seed).sample(pool, n)


__all__ = ["HF_DATASET", "PREPROCESSING_ID", "strip_na", "load_mmlu_pro", "sample_questions"]
