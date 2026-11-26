"""Plot the XShare GSM8K sweep from ``summary.csv`` (written by summarize.py).

    python -m Mixture_of_Experts.experiments.xshare.plot_results SUMMARY_CSV --out-dir DIR

Per batch size B it writes ``accuracy_vs_throughput_c{B}.png`` and
``accuracy_vs_activated_experts_c{B}.png``, plus ``overview.png`` (the
throughput plot for every batch size side by side). Error bars are the 95%
Wilson interval (vertical) and the standard deviation over subsets
(horizontal). Differences inside the bars are noise.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NUMERIC = [
    "accuracy", "acc_ci_low", "acc_ci_high", "tps_mean", "tps_sd", "activated_experts",
]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row = dict(raw)
            row["batch_size"] = int(raw["batch_size"])
            for key in NUMERIC:
                row[key] = float(raw[key]) if raw.get(key) not in (None, "") else None
            rows.append(row)
    return rows


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points (x, y) not dominated by another (both >=, at least one >), sorted by x."""
    front = [
        p for p in points
        if not any(q != p and q[0] >= p[0] and q[1] >= p[1] for q in points)
    ]
    return sorted(set(front))


def acc_errors(rows: list[dict]) -> list[list[float]]:
    """Asymmetric vertical error bars (percent) from the Wilson interval."""
    lower = [100 * (r["accuracy"] - r["acc_ci_low"]) for r in rows]
    upper = [100 * (r["acc_ci_high"] - r["accuracy"]) for r in rows]
    return [lower, upper]


def draw_points(ax, rows: list[dict], xkey: str, xerr_key: str | None) -> None:
    for i, r in enumerate(rows):
        is_base = r["method"] == "baseline"
        xerr = r[xerr_key] if xerr_key and r[xerr_key] is not None else 0
        yerr = [[100 * (r["accuracy"] - r["acc_ci_low"])], [100 * (r["acc_ci_high"] - r["accuracy"])]]
        ax.errorbar(
            r[xkey], 100 * r["accuracy"], xerr=xerr, yerr=yerr,
            marker="*" if is_base else "o", markersize=14 if is_base else 7,
            color=f"C{i % 10}", capsize=3, linestyle="none",
        )
        ax.annotate(
            r["method"], (r[xkey], 100 * r["accuracy"]), textcoords="offset points",
            xytext=(5, 5), fontsize=7,
        )


def throughput_panel(ax, rows: list[dict], batch: int) -> None:
    rows = [r for r in rows if r["accuracy"] is not None and r["tps_mean"] is not None]
    draw_points(ax, rows, "tps_mean", "tps_sd")
    base = next((r for r in rows if r["method"] == "baseline"), None)
    if base is not None:
        ax.axhline(100 * base["accuracy"], color="gray", linestyle=":", linewidth=1)
    front = pareto_front([(r["tps_mean"], 100 * r["accuracy"]) for r in rows])
    if len(front) > 1:
        ax.plot(*zip(*front), linestyle="--", color="black", linewidth=1, zorder=0)
    ax.set_title(f"GSM8K accuracy vs decode throughput, batch size {batch}")
    ax.set_xlabel("Decode throughput (tokens/s, mean ± sd over subsets)")
    ax.set_ylabel("Accuracy (%, 95% Wilson CI)")
    ax.grid(alpha=0.3)


def experts_panel(ax, rows: list[dict], batch: int) -> None:
    rows = [r for r in rows if r["accuracy"] is not None and r["activated_experts"] is not None]
    draw_points(ax, rows, "activated_experts", None)
    ax.set_title(f"GSM8K accuracy vs activated experts, batch size {batch}")
    ax.set_xlabel("Mean activated experts per MoE layer per decode step")
    ax.set_ylabel("Accuracy (%, 95% Wilson CI)")
    ax.grid(alpha=0.3)


def save_single(panel, rows: list[dict], batch: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    panel(ax, rows, batch)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_overview(by_batch: dict[int, list[dict]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(by_batch), figsize=(8 * len(by_batch), 6), squeeze=False)
    for ax, (batch, rows) in zip(axes[0], sorted(by_batch.items())):
        throughput_panel(ax, rows, batch)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    by_batch: dict[int, list[dict]] = {}
    for r in rows:
        by_batch.setdefault(r["batch_size"], []).append(r)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for batch, group in sorted(by_batch.items()):
        for name, panel in (("accuracy_vs_throughput", throughput_panel),
                            ("accuracy_vs_activated_experts", experts_panel)):
            path = args.out_dir / f"{name}_c{batch}.png"
            save_single(panel, group, batch, path)
            written.append(path)
    overview = args.out_dir / "overview.png"
    save_overview(by_batch, overview)
    written.append(overview)
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
