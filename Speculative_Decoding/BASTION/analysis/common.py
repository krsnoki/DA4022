"""Shared loading / matching helpers for the analysis scripts (CPU only, no torch)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro_lib import PAPER_SHORT_DATASETS, load_jsonl  # noqa: E402

PAPER_ID = "bastion-arxiv-2605.29727"
PAPER_DATASET_ORDER = PAPER_SHORT_DATASETS
DATASET_LABELS = {
    "gsm8k": "GSM8K", "math500": "MATH500", "aime25": "AIME25", "humaneval": "HumanEval",
    "mbpp": "MBPP", "lcb": "LCB", "mt-bench": "MT-Bench", "alpaca": "Alpaca",
}
# HF sources used by upstream bastion/benchmark.py DATASETS (commit 1989b80).
DATASET_SOURCES = {
    "gsm8k": ("openai/gsm8k:main", "test"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
    "aime25": ("math-ai/aime25", "test"),
    "humaneval": ("openai/openai_humaneval", "test"),
    "mbpp": ("google-research-datasets/mbpp:sanitized", "test"),
    "lcb": ("livecodebench/code_generation_lite:release_v2", "test"),
    "mt-bench": ("HuggingFaceH4/mt_bench_prompts", "train"),
    "alpaca": ("tatsu-lab/alpaca", "train"),
}

# Paper Table 1, greedy decoding (T=0), NVIDIA A100: (speedup, tau). Paper-reported, NOT measured here.
PAPER_TABLE1_T0 = {
    ("Qwen/Qwen3-4B", "dflash"): {
        "gsm8k": (5.23, 6.50), "math500": (6.22, 8.12), "aime25": (6.16, 7.48), "humaneval": (5.34, 6.66),
        "mbpp": (5.09, 6.31), "lcb": (5.71, 7.23), "mt-bench": (2.74, 4.63), "alpaca": (2.12, 2.90),
        "Average": (4.83, 6.23),
    },
    ("Qwen/Qwen3-4B", "bastion"): {
        "gsm8k": (7.44, 9.07), "math500": (8.57, 10.60), "aime25": (8.23, 9.70), "humaneval": (7.91, 9.38),
        "mbpp": (7.41, 8.99), "lcb": (7.81, 9.48), "mt-bench": (4.53, 6.58), "alpaca": (3.53, 4.64),
        "Average": (6.93, 8.56),
    },
    ("Qwen/Qwen3-8B", "dflash"): {
        "gsm8k": (4.96, 6.51), "math500": (6.06, 8.25), "aime25": (5.91, 7.20), "humaneval": (5.25, 6.64),
        "mbpp": (4.91, 6.04), "lcb": (5.90, 7.64), "mt-bench": (2.81, 4.46), "alpaca": (2.28, 3.15),
        "Average": (4.76, 6.24),
    },
    ("Qwen/Qwen3-8B", "bastion"): {
        "gsm8k": (6.92, 8.85), "math500": (8.02, 10.53), "aime25": (7.69, 9.46), "humaneval": (7.64, 9.44),
        "mbpp": (7.10, 8.65), "lcb": (7.64, 9.91), "mt-bench": (4.31, 6.28), "alpaca": (3.54, 4.68),
        "Average": (6.61, 8.48),
    },
}


def load_records(paths: Iterable[str | Path]) -> list[dict]:
    out = []
    for p in paths:
        for rec in load_jsonl(p):
            rec = dict(rec)
            rec["_source"] = str(p)
            out.append(rec)
    return out


def load_meta(jsonl_path: str | Path) -> dict:
    """Merged view of ``<run>.meta.json`` (later sessions override earlier keys)."""
    path = Path(jsonl_path)
    meta_path = path.with_name(path.name[: -len(".jsonl")] + ".meta.json") if path.name.endswith(".jsonl") else None
    if meta_path is None or not meta_path.exists():
        return {}
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    merged: dict = {}
    for session in data.get("sessions", []):
        merged.update({k: v for k, v in session.items() if v is not None})
    merged["num_sessions"] = len(data.get("sessions", []))
    return merged


def method_label(mode: str, budget) -> str:
    if mode == "ar":
        return "AR"
    if mode == "dflash":
        return "DFlash"
    if mode == "bastion":
        return "BASTION" if budget is None else f"BASTION-Fixed-{int(budget)}"
    return mode


def turn_key(rec: dict) -> tuple:
    """(model, dataset, problem, turn): the unit upstream averages over.

    The problem is identified by ``orig_idx`` (index in upstream's dataset
    order), not ``sample_idx``: runs with different caps can order the same
    problems differently (e.g. aime25 is not shuffled at cap 30 but is at cap 16).
    """
    prob = rec.get("orig_idx")
    if prob is None:
        prob = ("sample", rec["sample_idx"])
    return (rec["model"], rec["dataset"], prob, int(rec["turn"]))


def dataset_sort_key(name: str) -> tuple:
    return (PAPER_DATASET_ORDER.index(name), name) if name in PAPER_DATASET_ORDER else (len(PAPER_DATASET_ORDER), name)


def model_short(model: str) -> str:
    return model.split("/")[-1]
