"""Shared helpers for the cascade-escalation experiment scripts."""

from __future__ import annotations

import gzip
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(path: str | os.PathLike) -> Dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def environment_metadata() -> Dict:
    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "commit": git_commit(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            meta["cuda_runtime"] = torch.version.cuda
            meta["device"] = torch.cuda.get_device_name(0)
            meta["device_memory_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**20)
    except Exception:
        pass
    for mod in ("transformers", "vllm", "numpy", "pandas"):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        meta["nvidia_driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip()
    except Exception:
        pass
    return meta


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> List[Dict]:
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def generation_path(out_dir: Path, dataset: str, slug: str) -> Path:
    return Path(out_dir) / "generations" / dataset / f"{slug}.jsonl.gz"
