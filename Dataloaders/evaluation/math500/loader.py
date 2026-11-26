"""Reproducible dataloader for the MATH-500 evaluation test split."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

_REQUIRED_FIELDS = (
    "unique_id",
    "problem",
    "answer",
    "solution",
    "subject",
    "level",
)


@dataclass(frozen=True)
class Math500LoaderConfig:
    """Configuration for loading and deterministically subsampling MATH-500."""

    dataset_id: str = "HuggingFaceH4/MATH-500"
    revision: str = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    split: str = "test"
    num_samples: int | None = None
    seed: int = 42
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    local_jsonl_path: str | Path | None = None

    def __post_init__(self) -> None:
        if self.split != "test":
            raise ValueError(
                f"MATH-500 only exposes the 'test' evaluation split, got {self.split!r}"
            )
        if self.num_samples is not None and (
            isinstance(self.num_samples, bool) or self.num_samples < 0
        ):
            raise ValueError(
                f"num_samples must be None or >= 0, got {self.num_samples!r}"
            )


@dataclass(frozen=True)
class Math500Record:
    """Raw MATH-500 dataset row."""

    unique_id: str
    problem: str
    answer: str
    solution: str
    subject: str
    level: int


def format_math500_messages(
    problem: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Format a problem string into Appendix D.2 system/user chat messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]


def get_dataset_metadata() -> dict[str, Any]:
    """Return registry-aligned metadata for the MATH-500 evaluation split."""
    return {
        "id": "math.math500.test",
        "name": "MATH-500 evaluation test split",
        "modality": "text",
        "task": "mathematical_reasoning",
        "source_url": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500",
        "license": "mit",
        "version": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "role": "evaluation",
        "split": "test",
        "preprocessing_id": "dipa.math_cot_boxed.v1",
        "seed": 42,
        "num_examples": 500,
        "labels_available": True,
    }


def _read_local_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"MATH-500 JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Expected JSON object at line {line_no} of {path}, "
                    f"got {type(parsed).__name__}"
                )
            rows.append(parsed)
    return rows


def _load_from_huggingface(config: Math500LoaderConfig) -> list[Any]:
    # Lazy import keeps local JSONL loading and tests usable without datasets installed.
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required when local_jsonl_path is None. "
            "Install 'datasets' or pass local_jsonl_path."
        ) from exc
    dataset = load_dataset(
        config.dataset_id,
        split=config.split,
        revision=config.revision,
    )
    return list(dataset)


def _parse_record(raw: Mapping[str, Any], index: int) -> Math500Record:
    missing = [key for key in _REQUIRED_FIELDS if key not in raw]
    if missing:
        raise ValueError(f"Record at index {index} missing required fields: {missing}")
    return Math500Record(
        unique_id=str(raw["unique_id"]),
        problem=str(raw["problem"]),
        answer=str(raw["answer"]),
        solution=str(raw["solution"]),
        subject=str(raw["subject"]),
        level=int(raw["level"]),
    )


def _subsample_records(
    records: list[Math500Record],
    num_samples: int | None,
    seed: int,
) -> list[Math500Record]:
    if num_samples == 0:
        return []
    if num_samples is None or num_samples >= len(records):
        return list(records)
    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), num_samples)
    return [records[idx] for idx in indices]


def _format_output_record(
    record: Math500Record,
    config: Math500LoaderConfig,
) -> dict[str, Any]:
    return {
        "instance_id": record.unique_id,
        "prompt": record.problem,
        "messages": format_math500_messages(
            record.problem,
            system_prompt=config.system_prompt,
        ),
        "ground_truth": record.answer,
        "metadata": {
            "subject": record.subject,
            "level": record.level,
            "solution": record.solution,
            "split": config.split,
            "dataset_id": config.dataset_id,
        },
    }


def load_math500(config: Math500LoaderConfig | None = None) -> list[dict[str, Any]]:
    """Load MATH-500 examples from a local JSONL file or Hugging Face Datasets."""
    cfg = config if config is not None else Math500LoaderConfig()
    if cfg.local_jsonl_path is not None:
        raw_rows = _read_local_jsonl(Path(cfg.local_jsonl_path))
    else:
        raw_rows = _load_from_huggingface(cfg)

    records = [_parse_record(row, idx) for idx, row in enumerate(raw_rows)]
    sampled = _subsample_records(records, cfg.num_samples, cfg.seed)
    return [_format_output_record(record, cfg) for record in sampled]
