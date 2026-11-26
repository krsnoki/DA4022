"""Small I/O helpers shared by the ST-ES experiment scripts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Iterable, Iterator

# Generation caches and run outputs live outside the repository by default.
DEFAULT_RUN_DIR = Path(os.environ.get("ST_ES_RUN_DIR", Path(__file__).resolve().parents[4] / "st_es_runs"))


def run_relative(path: str | Path, run_dir: str | Path) -> str:
    """A path as written into committed outputs: ``<run_dir>/...`` if it lies in the run
    directory, else only its file name. Absolute local paths are never recorded."""
    p = Path(path).resolve()
    try:
        return "<run_dir>/" + p.relative_to(Path(run_dir).resolve()).as_posix()
    except ValueError:
        return p.name


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str | Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment_metadata() -> dict:
    import importlib

    meta = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "datasets", "math_verify", "numpy"):
        try:
            meta[mod] = importlib.import_module(mod).__version__
        except Exception:  # pragma: no cover - optional dependency missing
            meta[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            meta.update({"gpu": props.name, "gpu_total_memory_mb": round(props.total_memory / 2**20),
                         "cuda_runtime": torch.version.cuda})
    except Exception:  # pragma: no cover
        pass
    return meta


METRICS_SCHEMA = ["experiment_id", "paper_id", "method", "commit", "backbone_id", "backbone_revision", "dataset_id",
                  "split", "seed", "device", "framework_version", "dtype", "batch_size", "input_length", "output_length",
                  "quality_metric", "quality_value", "latency_median_ms", "latency_p95_ms", "throughput",
                  "peak_memory_mb", "energy_j", "status", "notes"]


def upsert_metric_rows(path: str | Path, rows: list[dict], id_prefix: str) -> None:
    """Replace the rows whose ``experiment_id`` starts with ``id_prefix`` and keep every other row.

    Several scripts write to the same results CSV (``Performance_Metrics`` schema); each owns one prefix.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = []
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            keep = [r for r in csv.DictReader(fh) if not r["experiment_id"].startswith(id_prefix)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=METRICS_SCHEMA)
        w.writeheader()
        w.writerows(keep + rows)


def git_commit(repo: str | Path) -> str:
    """HEAD commit of ``repo``, with ``+dirty`` when tracked code/config files are modified.

    Result folders are excluded from the check: report scripts rewrite them, so a
    pending result file says nothing about the code that produced the numbers.
    """
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".",
                                ":(exclude)Adaptive_Stopping/experiments/st_es/results",
                                ":(exclude)Performance_Metrics/results"], cwd=repo,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def limit_gpu_memory(fraction: float = 0.85) -> None:
    """Cap PyTorch's CUDA caching allocator at ``fraction`` of device memory.

    On Windows (WDDM) an allocator that grows past dedicated memory does not
    fail: it spills into shared system memory, which is backed by host commit
    charge. In our runs this pushed one process to 16.5 GB of commit and grew
    the page file until the system drive was almost full. With a cap, PyTorch
    releases cached blocks and retries at the limit, and only raises OOM if the
    memory is really needed.
    """
    import torch

    if fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


def rows_by_uid(records: Iterable[dict]) -> dict[str, dict]:
    return {r["uid"]: r for r in records}
