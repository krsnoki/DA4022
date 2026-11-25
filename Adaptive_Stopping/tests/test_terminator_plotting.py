"""Tests for the TERMINATOR results summary/plotting script.

Builds a tiny synthetic JSONL in ``tmp_path`` (two problems, a ``no_stop``
and a ``terminator`` config) and checks the summary numbers exactly, that
the plot files are created, and that the appended metrics CSV uses the
column list from ``Performance_Metrics/README_metrics.md``.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from Adaptive_Stopping.experiments.terminator.plot_results import (  # noqa: E402
    METRICS_COLUMNS,
    run,
)

ROWS = [
    # no_stop baseline: both problems run to completion.
    {
        "uid": "p1", "method": "no_stop", "threshold": None, "window": None,
        "budget_tokens": None, "seed": 0, "correct": True,
        "thinking_tokens": 100, "answer_tokens": 10, "stopped_early": False,
        "latency_s": None, "peak_mem_mb": None,
        "label_answer_pos": None, "exit_pos": None,
    },
    {
        "uid": "p2", "method": "no_stop", "threshold": None, "window": None,
        "budget_tokens": None, "seed": 0, "correct": True,
        "thinking_tokens": 200, "answer_tokens": 10, "stopped_early": False,
        "latency_s": None, "peak_mem_mb": None,
        "label_answer_pos": None, "exit_pos": None,
    },
    # terminator: stops early on both, gets p2 wrong.
    {
        "uid": "p1", "method": "terminator", "threshold": 0.5, "window": 5,
        "budget_tokens": None, "seed": 0, "correct": True,
        "thinking_tokens": 40, "answer_tokens": 10, "stopped_early": True,
        "latency_s": 1.0, "peak_mem_mb": 500,
        "label_answer_pos": 40, "exit_pos": 42,
    },
    {
        "uid": "p2", "method": "terminator", "threshold": 0.5, "window": 5,
        "budget_tokens": None, "seed": 0, "correct": False,
        "thinking_tokens": 100, "answer_tokens": 10, "stopped_early": True,
        "latency_s": 2.0, "peak_mem_mb": 600,
        "label_answer_pos": 100, "exit_pos": 105,
    },
]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def summary_by_method(summary: list[dict]) -> dict[str, dict]:
    return {row["method"]: row for row in summary}


def test_summary_numbers_exact(tmp_path):
    jsonl = write_jsonl(tmp_path / "results.jsonl", ROWS)
    out_dir = tmp_path / "out"
    metrics_csv = tmp_path / "metrics.csv"

    result = run([jsonl], out_dir, commit="abc123", device="Tesla T4", metrics_csv=metrics_csv)
    by_method = summary_by_method(result["summary"])

    no_stop = by_method["no_stop"]
    assert no_stop["n"] == 2
    assert no_stop["accuracy"] == pytest.approx(1.0)
    assert no_stop["mean_thinking_tokens"] == pytest.approx(150.0)
    assert no_stop["compression_rate"] == pytest.approx(1.0)
    assert no_stop["stop_rate"] == pytest.approx(0.0)
    assert no_stop["latency_median_s"] is None
    assert no_stop["latency_p95_s"] is None
    assert no_stop["peak_mem_mb"] is None

    terminator = by_method["terminator"]
    assert terminator["n"] == 2
    assert terminator["accuracy"] == pytest.approx(0.5)
    assert terminator["mean_thinking_tokens"] == pytest.approx(70.0)
    # compression_rate = mean thinking tokens / no_stop mean on the same uids
    assert terminator["compression_rate"] == pytest.approx(70.0 / 150.0)
    assert terminator["stop_rate"] == pytest.approx(1.0)
    assert terminator["latency_median_s"] == pytest.approx(1.5)
    assert terminator["latency_p95_s"] == pytest.approx(1.95)
    assert terminator["peak_mem_mb"] == pytest.approx(600.0)


def test_summary_csv_written(tmp_path):
    jsonl = write_jsonl(tmp_path / "results.jsonl", ROWS)
    out_dir = tmp_path / "out"
    run([jsonl], out_dir, commit="abc123", device="Tesla T4", metrics_csv=tmp_path / "metrics.csv")

    with (out_dir / "summary.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert {r["method"] for r in rows} == {"no_stop", "terminator"}


def test_plots_created(tmp_path):
    jsonl = write_jsonl(tmp_path / "results.jsonl", ROWS)
    out_dir = tmp_path / "out"
    result = run([jsonl], out_dir, commit="abc123", device="Tesla T4", metrics_csv=tmp_path / "metrics.csv")

    assert result["made_compression_plot"] is True
    assert result["made_latency_plot"] is True
    assert result["made_hist_plot"] is True
    assert (out_dir / "accuracy_vs_compression.png").exists()
    assert (out_dir / "accuracy_vs_latency.png").exists()
    assert (out_dir / "terminator_exit_offset_hist.png").exists()


def test_latency_plot_skipped_without_latency_data(tmp_path):
    no_latency_rows = [dict(r, latency_s=None) for r in ROWS]
    jsonl = write_jsonl(tmp_path / "results.jsonl", no_latency_rows)
    out_dir = tmp_path / "out"
    result = run([jsonl], out_dir, commit="abc123", device="Tesla T4", metrics_csv=tmp_path / "metrics.csv")

    assert result["made_latency_plot"] is False
    assert not (out_dir / "accuracy_vs_latency.png").exists()


def test_hist_skipped_without_exit_positions(tmp_path):
    no_pos_rows = [dict(r, exit_pos=None, label_answer_pos=None) for r in ROWS]
    jsonl = write_jsonl(tmp_path / "results.jsonl", no_pos_rows)
    out_dir = tmp_path / "out"
    result = run([jsonl], out_dir, commit="abc123", device="Tesla T4", metrics_csv=tmp_path / "metrics.csv")

    assert result["made_hist_plot"] is False
    assert not (out_dir / "terminator_exit_offset_hist.png").exists()


def test_metrics_csv_columns_match_readme(tmp_path):
    jsonl = write_jsonl(tmp_path / "results.jsonl", ROWS)
    out_dir = tmp_path / "out"
    metrics_csv = tmp_path / "metrics.csv"
    run([jsonl], out_dir, commit="abc123", device="Tesla T4", metrics_csv=metrics_csv)

    with metrics_csv.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == METRICS_COLUMNS

    readme = Path(__file__).resolve().parents[2] / "Performance_Metrics" / "README_metrics.md"
    text = readme.read_text(encoding="utf-8")
    match = re.search(r"```text\n(.+?)\n```", text, re.DOTALL)
    assert match is not None, "could not find the required report format code block"
    readme_columns = match.group(1).strip().split(",")
    assert header == readme_columns


def test_metrics_csv_appends_without_duplicate_header(tmp_path):
    jsonl = write_jsonl(tmp_path / "results.jsonl", ROWS)
    out_dir = tmp_path / "out"
    metrics_csv = tmp_path / "metrics.csv"
    metrics_csv.write_text(",".join(METRICS_COLUMNS) + "\n", encoding="utf-8")

    run([jsonl], out_dir, commit="abc123", device="Tesla T4", metrics_csv=metrics_csv)

    with metrics_csv.open(newline="", encoding="utf-8") as fh:
        lines = fh.readlines()
    assert lines[0].strip() == ",".join(METRICS_COLUMNS)
    assert len(lines) == 1 + 2  # header + one row per config, no duplicated header


def test_metrics_csv_energy_empty_with_note(tmp_path):
    jsonl = write_jsonl(tmp_path / "results.jsonl", ROWS)
    out_dir = tmp_path / "out"
    metrics_csv = tmp_path / "metrics.csv"
    run([jsonl], out_dir, commit="abc123", device="Tesla T4", metrics_csv=metrics_csv)

    with metrics_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        assert row["energy_j"] == ""
        assert "not_measured" in row["notes"]
