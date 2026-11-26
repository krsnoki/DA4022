"""Deterministic text-prompt loader for registry datasets (evaluation / benchmark roles).

Usage::

    from Dataloaders.evaluation.text_prompts import load_prompts
    prompts = load_prompts("text.alpaca_eval.eval", num_samples=32, seed=42)

Downloads the pinned file once into ``$DA4022_DATA_CACHE`` (default
``~/.cache/da4022/datasets``), verifies its sha256, and returns
:class:`PromptRecord` objects. Sampling is ``random.Random(seed).sample`` over record
indices, returned in ascending index order, so the same seed always gives the same
subset. Stdlib + PyYAML only.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "registry.yaml"


@dataclass(frozen=True)
class PromptRecord:
    """One prompt. ``reference`` is the gold answer text when the dataset has labels."""

    id: str
    prompt: str
    reference: str | None
    dataset_id: str


@lru_cache(maxsize=1)
def load_registry(path: str | None = None) -> dict[str, dict[str, Any]]:
    with open(path or REGISTRY_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["datasets"]


def get_entry(dataset_id: str) -> dict[str, Any]:
    reg = load_registry()
    if dataset_id not in reg:
        raise KeyError(f"dataset {dataset_id!r} not in registry ({sorted(reg)})")
    return reg[dataset_id]


def cache_dir() -> Path:
    return Path(os.environ.get("DA4022_DATA_CACHE", Path.home() / ".cache" / "da4022" / "datasets"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(dataset_id: str, entry: dict[str, Any]) -> Path:
    """Download (once) and verify the pinned file for ``dataset_id``."""
    dl = entry["download"]
    name = dl["url"].rsplit("/", 1)[-1]
    path = cache_dir() / dataset_id / f"{entry['version'][:12]}_{name}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        req = urllib.request.Request(dl["url"], headers={"User-Agent": "da4022-dataloader/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
            out.write(resp.read())
        tmp.replace(path)
    digest = sha256_of(path)
    if digest != dl["sha256"]:
        raise ValueError(f"{dataset_id}: sha256 mismatch for {path} (got {digest}, "
                         f"expected {dl['sha256']}); delete the file and retry")
    return path


def _read_records(path: Path, fmt: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if fmt == "json":
        return json.loads(text)
    if fmt == "jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    raise ValueError(f"unsupported format {fmt!r}")


# ---------------------------------------------------------------- synthetic sets
_TOPICS = ["photosynthesis", "binary search", "the water cycle", "TCP handshakes",
           "compound interest", "black holes", "garbage collection", "vaccines"]


def _synthetic(dataset_id: str, kind: str, n: int) -> list[PromptRecord]:
    rng = random.Random(0)  # fixed generator seed; subset sampling uses the caller's seed
    out = []
    for i in range(n):
        if kind == "prompts":
            topic = _TOPICS[i % len(_TOPICS)]
            out.append(PromptRecord(f"syn{i}", f"Explain {topic} to a student in a few paragraphs.",
                                    None, dataset_id))
        elif kind == "arithmetic":
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            out.append(PromptRecord(f"arith{i}", f"Sam has {a} apples and buys {b} more. "
                                    "How many apples does Sam have now?",
                                    f"{a} + {b} = {a + b}\n#### {a + b}", dataset_id))
        else:
            raise ValueError(f"unknown synthetic generator {kind!r}")
    return out


# --------------------------------------------------------------------- public API
def load_all(dataset_id: str) -> list[PromptRecord]:
    """All records of a registry dataset, in file order."""
    entry = get_entry(dataset_id)
    if "generator" in entry:
        return _synthetic(dataset_id, entry["generator"], int(entry["num_examples"]))
    dl = entry["download"]
    rows = _read_records(fetch(dataset_id, entry), dl["format"])
    records = []
    for i, row in enumerate(rows):
        rid = str(row[dl["id_field"]]) if dl.get("id_field") else str(i)
        ref = row.get(dl["reference_field"]) if dl.get("reference_field") else None
        records.append(PromptRecord(rid, row[dl["prompt_field"]], ref, dataset_id))
    if len(records) != int(entry["num_examples"]):
        raise ValueError(f"{dataset_id}: expected {entry['num_examples']} records, got {len(records)}")
    return records


def load_prompts(dataset_id: str, num_samples: int | None = None, seed: int = 42) -> list[PromptRecord]:
    """Deterministic subset of ``num_samples`` records (all records if None)."""
    records = load_all(dataset_id)
    if num_samples is None or num_samples >= len(records):
        return records
    idx = sorted(random.Random(seed).sample(range(len(records)), num_samples))
    return [records[i] for i in idx]
