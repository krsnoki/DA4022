"""Machine-readable result rows in the schema of ``Performance_Metrics/README_metrics.md``.

Unknown or unmeasured values stay ``None`` (written as an empty CSV cell),
never ``0``.
"""

from __future__ import annotations

import csv
import math
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

COLUMNS = [
    "experiment_id", "paper_id", "method", "commit", "backbone_id", "backbone_revision", "dataset_id", "split",
    "seed", "device", "framework_version", "dtype", "batch_size", "input_length", "output_length",
    "quality_metric", "quality_value", "latency_median_ms", "latency_p95_ms", "throughput", "peak_memory_mb",
    "energy_j", "status", "notes",
]
STATUSES = {"reproduced", "partial", "approximate", "not reproduced"}


def make_row(**fields) -> Dict[str, object]:
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown result columns: {sorted(unknown)}")
    row = {c: fields.get(c) for c in COLUMNS}
    if row["status"] not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}, got {row['status']!r}")
    return row


def write_csv(rows: Iterable[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})


def percentile(xs: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolation percentile, q in [0, 100]."""
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def environment_metadata() -> Dict[str, object]:
    meta = {"platform": platform.platform(), "python": sys.version.split()[0], "processor": platform.processor()}
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["device_name"] = torch.cuda.get_device_name(0)
            meta["cuda_runtime"] = torch.version.cuda
    except ImportError:
        meta["torch"] = None
    return meta


def peak_memory_mb() -> Dict[str, Optional[float]]:
    """Peak process RSS (all platforms) and peak CUDA allocated memory (if available), labelled separately."""
    out: Dict[str, Optional[float]] = {"process_peak_rss_mb": None, "cuda_peak_allocated_mb": None}
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["process_peak_rss_mb"] = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            out["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    except ImportError:
        pass
    return out


__all__ = ["COLUMNS", "STATUSES", "make_row", "write_csv", "percentile", "environment_metadata", "peak_memory_mb"]
