"""Summarize and plot TERMINATOR early-stopping results.

    python plot_results.py --inputs a.jsonl b.jsonl --out-dir DIR \
        --commit HASH --device "Tesla T4" [--metrics-csv PATH]

Reads one or more JSONL files, one row per (problem, method run), with the
fields written by ``evaluate.py`` (``per_problem.jsonl``) and ``latency.py``:
``uid``, ``method`` (``no_stop`` | ``fixed_budget`` |
``oracle`` | ``terminator``), ``threshold``, ``window``, ``budget_tokens``,
``seed``, ``correct``, ``thinking_tokens``, ``answer_tokens``,
``stopped_early``, ``latency_s``, ``peak_mem_mb``, ``label_answer_pos`` and
``exit_pos``. A **config** is the tuple
``(method, threshold, window, budget_tokens)``; rows are pooled over seeds
within a config.

Writes, under ``--out-dir``:

* ``summary.csv`` -- one row per config: ``n``, ``accuracy``,
  ``mean_thinking_tokens``, ``compression_rate`` (this config's mean
  thinking tokens divided by the ``no_stop`` config's mean thinking tokens,
  computed over the *same uids* this config has rows for), ``stop_rate``,
  ``latency_median_s`` / ``latency_p95_s`` (over rows with a latency value),
  and ``peak_mem_mb`` (max over rows with a value).
* ``accuracy_vs_compression.png`` -- accuracy vs. compression rate per
  config, with the Pareto frontier (low compression, high accuracy).
* ``accuracy_vs_latency.png`` -- accuracy vs. median latency per config.
  Skipped (with a message, no error) if no config has latency data.
* ``terminator_exit_offset_hist.png`` -- histogram of
  ``exit_pos - label_answer_pos`` over ``terminator`` rows where both are
  present. Skipped (with a message, no error) if there are none.

Also appends one row per config to the shared results CSV (default
``Performance_Metrics/results/terminator.csv``, override with
``--metrics-csv``), using the column list from
``Performance_Metrics/README_metrics.md``'s "Required report format".
Fields this script cannot derive from the JSONL rows or the CLI arguments
(e.g. ``backbone_id``, ``dataset_id``, ``framework_version``) are left
empty rather than guessed or set to 0. ``energy_j`` is always left empty,
with ``energy not_measured`` recorded in ``notes``. ``status`` is written as
``partial`` since this script only summarizes numbers; update it by hand
once a run has been reviewed against a baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NO_STOP_METHOD = "no_stop"
TERMINATOR_METHOD = "terminator"

SUMMARY_COLUMNS = [
    "method", "threshold", "window", "budget_tokens", "n", "accuracy",
    "mean_thinking_tokens", "compression_rate", "stop_rate",
    "latency_median_s", "latency_p95_s", "peak_mem_mb",
]

METRICS_COLUMNS = [
    "experiment_id", "paper_id", "method", "commit", "backbone_id",
    "backbone_revision", "dataset_id", "split", "seed", "device",
    "framework_version", "dtype", "batch_size", "input_length",
    "output_length", "quality_metric", "quality_value",
    "latency_median_ms", "latency_p95_ms", "throughput", "peak_memory_mb",
    "energy_j", "status", "notes",
]

ConfigKey = tuple  # (method, threshold | None, window | None, budget_tokens | None)


# --------------------------------------------------------------------- I/O

def load_rows(paths: list[Path]) -> list[dict]:
    """Read and concatenate JSONL rows from one or more files."""
    rows = []
    for path in paths:
        with path.open(encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def config_key(row: dict) -> ConfigKey:
    return (row["method"], row.get("threshold"), row.get("window"), row.get("budget_tokens"))


def group_by_config(rows: list[dict]) -> dict[ConfigKey, list[dict]]:
    groups: dict[ConfigKey, list[dict]] = {}
    for row in rows:
        groups.setdefault(config_key(row), []).append(row)
    return groups


# ---------------------------------------------------------------- statistics

def mean_or_none(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (same convention as numpy's default)."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def no_stop_mean_for_uids(no_stop_rows: list[dict], uids: set) -> float | None:
    """Mean ``thinking_tokens`` of the no_stop rows restricted to ``uids``."""
    matched = [r["thinking_tokens"] for r in no_stop_rows if r["uid"] in uids]
    return mean_or_none(matched)


def summarize_config(key: ConfigKey, rows: list[dict], no_stop_rows: list[dict]) -> dict:
    method, threshold, window, budget_tokens = key
    n = len(rows)
    accuracy = mean_or_none([1.0 if r["correct"] else 0.0 for r in rows])
    mean_thinking = mean_or_none([r["thinking_tokens"] for r in rows])
    uids = {r["uid"] for r in rows}
    baseline_mean = no_stop_mean_for_uids(no_stop_rows, uids)
    compression_rate = None
    if mean_thinking is not None and baseline_mean:
        compression_rate = mean_thinking / baseline_mean
    stop_rate = mean_or_none([1.0 if r["stopped_early"] else 0.0 for r in rows])
    latencies = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    mem = [r["peak_mem_mb"] for r in rows if r.get("peak_mem_mb") is not None]
    return {
        "method": method, "threshold": threshold, "window": window,
        "budget_tokens": budget_tokens, "n": n, "accuracy": accuracy,
        "mean_thinking_tokens": mean_thinking, "compression_rate": compression_rate,
        "stop_rate": stop_rate,
        "latency_median_s": statistics.median(latencies) if latencies else None,
        "latency_p95_s": percentile(latencies, 95) if latencies else None,
        "peak_mem_mb": max(mem) if mem else None,
    }


def _sort_rank(value) -> tuple:
    """Sort key placing ``None`` after any real value."""
    return (1, 0) if value is None else (0, value)


def build_summary(groups: dict[ConfigKey, list[dict]]) -> list[dict]:
    no_stop_rows = [r for key, rs in groups.items() if key[0] == NO_STOP_METHOD for r in rs]
    ordered_keys = sorted(
        groups,
        key=lambda k: (k[0], _sort_rank(k[1]), _sort_rank(k[2]), _sort_rank(k[3])),
    )
    return [summarize_config(key, groups[key], no_stop_rows) for key in ordered_keys]


def csv_value(x):
    return "" if x is None else x


def write_summary_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: csv_value(r.get(c)) for c in SUMMARY_COLUMNS})


# --------------------------------------------------------------- plotting

def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated (compression, accuracy) points: minimize x, maximize y."""
    def dominates(a, b) -> bool:
        weakly_better = a[0] <= b[0] and a[1] >= b[1]
        strictly_better = a[0] < b[0] or a[1] > b[1]
        return weakly_better and strictly_better

    front = [p for p in points if not any(dominates(q, p) for q in points if q != p)]
    return sorted(set(front))


def plot_accuracy_vs_compression(rows: list[dict], path: Path) -> bool:
    points = [
        (r["compression_rate"], 100 * r["accuracy"], r["method"])
        for r in rows
        if r["compression_rate"] is not None and r["accuracy"] is not None
    ]
    if not points:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (x, y, method) in enumerate(points):
        ax.scatter(x, y, color=f"C{i % 10}", s=60, zorder=2)
        ax.annotate(method, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    front = pareto_front([(x, y) for x, y, _ in points])
    if len(front) > 1:
        ax.plot(*zip(*front), linestyle="--", color="black", linewidth=1, zorder=1)
    ax.set_xlabel("Compression rate (mean thinking tokens / no_stop mean, same uids)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. thinking-token compression rate")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_accuracy_vs_latency(rows: list[dict], path: Path) -> bool:
    points = [
        (r["latency_median_s"], 100 * r["accuracy"], r["method"])
        for r in rows
        if r["latency_median_s"] is not None and r["accuracy"] is not None
    ]
    if not points:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (x, y, method) in enumerate(points):
        ax.scatter(x, y, color=f"C{i % 10}", s=60)
        ax.annotate(method, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Median latency (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. median latency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_exit_offset_histogram(rows: list[dict], path: Path) -> bool:
    diffs = [
        r["exit_pos"] - r["label_answer_pos"]
        for r in rows
        if r.get("method") == TERMINATOR_METHOD
        and r.get("exit_pos") is not None
        and r.get("label_answer_pos") is not None
    ]
    if not diffs:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(diffs, bins=min(30, max(5, len(diffs) // 2 or 1)), color="C0", edgecolor="black")
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("exit_pos - label_answer_pos (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("TERMINATOR exit offset from the labeled answer position")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ----------------------------------------------------------- metrics export

def metrics_rows(summary: list[dict], rows_by_config: dict[ConfigKey, list[dict]], commit: str, device: str) -> list[dict]:
    out = []
    for i, s in enumerate(summary):
        key = (s["method"], s["threshold"], s["window"], s["budget_tokens"])
        raw = rows_by_config[key]
        seeds = sorted({r["seed"] for r in raw})
        seed_value = seeds[0] if len(seeds) == 1 else "|".join(str(x) for x in seeds)
        notes = (
            f"threshold={s['threshold']}, window={s['window']}, "
            f"budget_tokens={s['budget_tokens']}, n={s['n']}, "
            f"stop_rate={csv_value(s['stop_rate'])}; energy not_measured"
        )
        out.append({
            "experiment_id": f"terminator-{i:03d}-{s['method']}",
            "paper_id": "terminator",
            "method": s["method"],
            "commit": commit,
            "backbone_id": "",
            "backbone_revision": "",
            "dataset_id": "",
            "split": "",
            "seed": seed_value,
            "device": device,
            "framework_version": "",
            "dtype": "",
            "batch_size": "",
            "input_length": "",
            "output_length": "",
            "quality_metric": "accuracy",
            "quality_value": csv_value(s["accuracy"]),
            "latency_median_ms": csv_value(None if s["latency_median_s"] is None else 1000 * s["latency_median_s"]),
            "latency_p95_ms": csv_value(None if s["latency_p95_s"] is None else 1000 * s["latency_p95_s"]),
            "throughput": "",
            "peak_memory_mb": csv_value(s["peak_mem_mb"]),
            "energy_j": "",
            "status": "partial",
            "notes": notes,
        })
    return out


def append_metrics_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRICS_COLUMNS)
        if needs_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in METRICS_COLUMNS})


# ------------------------------------------------------------------- main

def default_metrics_csv() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "Performance_Metrics" / "results" / "terminator.csv"


def run(inputs: list[Path], out_dir: Path, commit: str, device: str, metrics_csv: Path | None) -> dict:
    """Build the summary/plots/metrics-csv outputs; returns a small report dict (for tests)."""
    rows = load_rows(inputs)
    groups = group_by_config(rows)
    summary = build_summary(groups)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summary, out_dir / "summary.csv")

    made_compression = plot_accuracy_vs_compression(summary, out_dir / "accuracy_vs_compression.png")
    made_latency = plot_accuracy_vs_latency(summary, out_dir / "accuracy_vs_latency.png")
    made_hist = plot_exit_offset_histogram(rows, out_dir / "terminator_exit_offset_hist.png")

    metrics_path = metrics_csv or default_metrics_csv()
    exported = metrics_rows(summary, groups, commit, device)
    append_metrics_csv(exported, metrics_path)

    return {
        "summary": summary,
        "made_compression_plot": made_compression,
        "made_latency_plot": made_latency,
        "made_hist_plot": made_hist,
        "metrics_csv": metrics_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    args = parser.parse_args()

    result = run(args.inputs, args.out_dir, args.commit, args.device, args.metrics_csv)

    print(f"Wrote {args.out_dir / 'summary.csv'}")
    if result["made_compression_plot"]:
        print(f"Wrote {args.out_dir / 'accuracy_vs_compression.png'}")
    if result["made_latency_plot"]:
        print(f"Wrote {args.out_dir / 'accuracy_vs_latency.png'}")
    else:
        print("Skipped accuracy_vs_latency.png: no config has latency data")
    if result["made_hist_plot"]:
        print(f"Wrote {args.out_dir / 'terminator_exit_offset_hist.png'}")
    else:
        print("Skipped terminator_exit_offset_hist.png: no terminator rows with exit_pos and label_answer_pos")
    print(f"Appended {len(result['summary'])} row(s) to {result['metrics_csv']}")


if __name__ == "__main__":
    main()
