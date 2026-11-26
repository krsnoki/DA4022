"""Shared CSV schema, summary statistics, and append-safe writers for metrics."""

from __future__ import annotations

from collections.abc import Sequence
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

METRICS_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "paper_id",
    "method",
    "commit",
    "backbone_id",
    "backbone_revision",
    "dataset_id",
    "split",
    "seed",
    "device",
    "framework_version",
    "dtype",
    "batch_size",
    "input_length",
    "output_length",
    "quality_metric",
    "quality_value",
    "latency_median_ms",
    "latency_p95_ms",
    "throughput",
    "peak_memory_mb",
    "energy_j",
    "status",
    "notes",
)

VALID_STATUSES: frozenset[str] = frozenset(
    {"reproduced", "partial", "approximate", "not_reproduced", "smoke_test"}
)


def _serialize_cell(value: Any) -> Any:
    """Serialize None or non-finite floats as empty strings, never zero."""
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


@dataclass(frozen=True)
class MetricsRow:
    """Single row conforming to ``Performance_Metrics/README_metrics.md``."""

    experiment_id: str
    paper_id: str
    method: str
    commit: str
    backbone_id: str
    backbone_revision: str
    dataset_id: str
    split: str
    seed: int
    device: str
    framework_version: str
    dtype: str
    batch_size: int
    input_length: int | float | str | None
    output_length: int | float | str | None
    quality_metric: str
    quality_value: float | None
    latency_median_ms: float | None = None
    latency_p95_ms: float | None = None
    throughput: float | None = None
    peak_memory_mb: float | None = None
    energy_j: float | str = "not_measured"
    status: str = "partial"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            allowed = ", ".join(sorted(VALID_STATUSES))
            raise ValueError(
                f"Invalid status {self.status!r}. Expected one of: {allowed}"
            )

    def to_csv_dict(self) -> dict[str, Any]:
        """Convert row to an ordered dict ready for ``csv.DictWriter``."""
        return {col: _serialize_cell(getattr(self, col)) for col in METRICS_COLUMNS}


def summarize_latencies_and_throughput(
    latencies_ms: Sequence[float],
    total_tokens: int,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(median_ms, p95_ms, tokens_per_sec)`` over per-attempt latencies."""
    if not latencies_ms:
        return (None, None, None)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    median_ms = float(np.percentile(arr, 50))
    p95_ms = float(np.percentile(arr, 95))
    total_ms = float(np.sum(arr))

    if not math.isfinite(total_ms) or total_ms <= 0.0:
        tokens_per_sec: float | None = None
    else:
        tokens_per_sec = float(total_tokens / (total_ms / 1000.0))

    return (median_ms, p95_ms, tokens_per_sec)


def append_metrics_rows(
    csv_path: str | Path,
    rows: Sequence[MetricsRow],
    overwrite: bool = False,
) -> Path:
    """Validate header and write or append ``MetricsRow`` records to CSV."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_has_content = path.exists() and path.stat().st_size > 0
    write_header = overwrite or not file_has_content

    if not overwrite and file_has_content:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, None)
        if existing_header != list(METRICS_COLUMNS):
            raise ValueError(
                f"CSV header mismatch in {path}: expected {list(METRICS_COLUMNS)}, "
                f"got {existing_header}"
            )

    mode = "w" if write_header else "a"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRICS_COLUMNS))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())

    return path
