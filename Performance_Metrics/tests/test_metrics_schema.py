"""Unit and contract tests for Performance_Metrics schema and CSV writers."""

from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from Performance_Metrics import (
    METRICS_COLUMNS,
    VALID_STATUSES,
    MetricsRow,
    append_metrics_rows,
    summarize_latencies_and_throughput,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "Performance_Metrics" / "README_metrics.md"


def _make_row(**overrides: object) -> MetricsRow:
    defaults: dict[str, object] = {
        "experiment_id": "exp_001",
        "paper_id": "DIPA",
        "method": "dipa_mgl",
        "commit": "abc1234",
        "backbone_id": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "backbone_revision": "main",
        "dataset_id": "HuggingFaceH4/MATH-500",
        "split": "test",
        "seed": 42,
        "device": "cpu",
        "framework_version": "python3.12",
        "dtype": "float32",
        "batch_size": 4,
        "input_length": None,
        "output_length": 512,
        "quality_metric": "coverage",
        "quality_value": 0.75,
    }
    defaults.update(overrides)
    return MetricsRow(**defaults)  # type: ignore[arg-type]


class TestMetricsSchemaAndSerialization:
    """Tests for column alignment with README_metrics.md and MetricsRow serialization."""

    def test_metrics_columns_match_readme_header_exactly(self) -> None:
        readme_lines = README_PATH.read_text(encoding="utf-8").splitlines()
        header_lines = [
            line.strip()
            for line in readme_lines
            if line.strip().startswith("experiment_id,paper_id,method,")
        ]
        assert len(header_lines) == 1
        assert tuple(header_lines[0].split(",")) == METRICS_COLUMNS
        assert len(METRICS_COLUMNS) == 24

    def test_status_validation_and_immutability(self) -> None:
        for valid_status in VALID_STATUSES:
            row = _make_row(status=valid_status)
            assert row.status == valid_status

        with pytest.raises(ValueError, match="Invalid status"):
            _make_row(status="invalid_status")

        row = _make_row()
        with pytest.raises(FrozenInstanceError):
            row.method = "other"  # type: ignore[misc]

    def test_none_serializes_as_empty_string_and_energy_defaults_not_measured(
        self,
    ) -> None:
        row = _make_row(
            input_length=None,
            quality_value=None,
            latency_median_ms=None,
            latency_p95_ms=None,
            throughput=None,
            peak_memory_mb=None,
            status="smoke_test",
        )
        as_dict = row.to_csv_dict()
        assert list(as_dict.keys()) == list(METRICS_COLUMNS)
        assert as_dict["input_length"] == ""
        assert as_dict["quality_value"] == ""
        assert as_dict["latency_median_ms"] == ""
        assert as_dict["latency_p95_ms"] == ""
        assert as_dict["throughput"] == ""
        assert as_dict["peak_memory_mb"] == ""
        assert as_dict["energy_j"] == "not_measured"

        measured = _make_row(
            latency_median_ms=12.5,
            latency_p95_ms=20.0,
            throughput=100.0,
            peak_memory_mb=2048.5,
            energy_j=15.25,
        )
        measured_dict = measured.to_csv_dict()
        assert measured_dict["energy_j"] == 15.25
        assert measured_dict["peak_memory_mb"] == 2048.5


class TestSummarizeLatenciesAndCsvWriter:
    """Tests for latency/throughput summaries and append-safe CSV writing."""

    def test_summarize_latencies_standard_empty_and_zero(self) -> None:
        assert summarize_latencies_and_throughput([], 100) == (None, None, None)

        med_z, p95_z, tps_z = summarize_latencies_and_throughput([0.0, 0.0], 50)
        assert med_z == 0.0
        assert p95_z == 0.0
        assert tps_z is None

        latencies = [100.0, 200.0, 300.0, 400.0]
        med, p95, tps = summarize_latencies_and_throughput(latencies, total_tokens=500)
        assert med == pytest.approx(float(np.percentile(latencies, 50)))
        assert p95 == pytest.approx(float(np.percentile(latencies, 95)))
        assert tps == pytest.approx(500.0)

    def test_append_metrics_rows_overwrite_append_and_header_mismatch(
        self, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "nested" / "metrics.csv"
        row1 = _make_row(experiment_id="e1", seed=42, quality_value=0.5)
        row2 = _make_row(experiment_id="e2", seed=43, quality_value=0.75)

        out = append_metrics_rows(csv_path, [row1], overwrite=True)
        assert out == csv_path
        assert csv_path.exists()

        append_metrics_rows(csv_path, [row2], overwrite=False)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == 3
        assert rows[0] == list(METRICS_COLUMNS)
        assert rows[1][0] == "e1"
        assert rows[2][0] == "e2"

        # Overwrite replaces existing content cleanly with a single header.
        append_metrics_rows(csv_path, [row2], overwrite=True)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            overwritten = list(csv.reader(handle))
        assert len(overwritten) == 2
        assert overwritten[1][0] == "e2"

        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col_a,col_b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="header mismatch"):
            append_metrics_rows(bad_csv, [row1], overwrite=False)


class TestPerformanceMetricsIsolation:
    """Ensure Performance_Metrics stays decoupled from peer packages and PyTorch."""

    def test_import_isolation_in_clean_subprocess(self) -> None:
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(REPO_ROOT)
        )
        script = (
            "import sys\n"
            "import Performance_Metrics\n"
            "from Performance_Metrics import METRICS_COLUMNS, MetricsRow\n"
            "assert len(METRICS_COLUMNS) == 24\n"
            "forbidden = ('torch', 'Adaptive_Stopping', 'Backbones', 'Dataloaders')\n"
            "leaked = [m for m in sys.modules if any(m == f or m.startswith(f + '.') for f in forbidden)]\n"
            "assert not leaked, f'Forbidden imports leaked: {leaked}'\n"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            env=env,
            check=True,
        )
