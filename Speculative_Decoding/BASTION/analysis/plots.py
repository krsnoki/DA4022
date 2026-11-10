#!/usr/bin/env python
"""Figures for the BASTION reproduction (PNG, matplotlib, colour-blind-safe palette).

(a) ``speedup_bars_<model>.png``  per-dataset speedup over AR, DFlash vs BASTION
    (measured, solid) with paper-reported A100 numbers (Table 1, T=0) as
    hatched, outlined reference bars when ``--paper`` is on.
(b) ``budget_sweep_<model>.png``  average speedup vs mean realized tree size:
    fixed-N line plus the adaptive controller as a star; and
    ``budget_sweep_<model>_per_dataset.png`` small multiples.
(c) ``latency_model_<gpu>_<model>.png``  measured verification latency vs
    verified sequence length at c in {64, 256, 1024}, uncalibrated roofline
    (dashed) and calibrated fit (solid), with RMSE in the legend.

Inputs are the JSON outputs of aggregate.py and latency_model.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASET_LABELS, PAPER_TABLE1_T0, dataset_sort_key, model_short  # noqa: E402

# Validated categorical palette (dataviz reference instance, light mode), fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METHOD_COLORS = {"dflash": BLUE, "bastion": ORANGE}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
})


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# (a) speedup bars
# ---------------------------------------------------------------------------

def plot_speedup_bars(rows: list[dict], out_dir: Path, paper: bool = True, device_label: str = "") -> list[Path]:
    out = []
    models = sorted({r["model"] for r in rows})
    for model in models:
        mrows = {(r["mode"], r["dataset"]): r for r in rows
                 if r["model"] == model and r.get("fixed_budget") is None and r["mode"] in ("dflash", "bastion")}
        if not mrows:
            continue
        datasets = sorted({d for (_, d) in mrows if d not in ("Average", "Pooled")}, key=dataset_sort_key)
        cats = datasets + (["Average"] if any(d == "Average" for (_, d) in mrows) else [])
        series = [("dflash", "measured"), ("bastion", "measured")]
        have_paper = paper and any((model, m) in PAPER_TABLE1_T0 for m in ("dflash", "bastion"))
        if have_paper:
            series = [("dflash", "measured"), ("dflash", "paper"), ("bastion", "measured"), ("bastion", "paper")]
        width = 0.8 / len(series)
        fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(cats) + 2), 3.8))
        x = np.arange(len(cats))
        for i, (mode, src) in enumerate(series):
            vals = []
            for d in cats:
                if src == "measured":
                    r = mrows.get((mode, d))
                    vals.append(np.nan if r is None or r.get("speedup") is None else r["speedup"])
                else:
                    vals.append(PAPER_TABLE1_T0.get((model, mode), {}).get(d, (np.nan,))[0])
            pos = x - 0.4 + width * (i + 0.5)
            color = METHOD_COLORS[mode]
            name = {"dflash": "DFlash", "bastion": "BASTION"}[mode]
            if src == "measured":
                bars = ax.bar(pos, vals, width * 0.92, color=color, label=f"{name} (measured{', ' + device_label if device_label else ''})")
                for b, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", va="bottom",
                                fontsize=6.5, color=INK2, rotation=90)
            else:
                ax.bar(pos, vals, width * 0.92, facecolor="white", edgecolor=color, hatch="////", linewidth=1.0,
                       label=f"{name} (paper-reported, A100)")
        ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle=":")
        labels = [DATASET_LABELS.get(d, d) for d in cats]
        if "Average" in cats and have_paper and len(datasets) != 8:
            labels[-1] = f"Average\n(measured {len(datasets)} ds,\npaper 8 ds)"
        ax.set_xticks(x, labels)
        ax.set_ylabel("Speedup over AR decoding (×)")
        ax.set_title(f"{model_short(model)}: decode speedup at T=0, batch 1", loc="left", fontsize=11, color=INK)
        ax.legend(ncol=2, fontsize=8, loc="upper right")
        ymax = np.nanmax([b.get_height() for b in ax.patches if np.isfinite(b.get_height())] + [1.0])
        ax.set_ylim(0, ymax * 1.25)
        fig.tight_layout()
        p = out_dir / f"speedup_bars_{model_short(model)}.png"
        fig.savefig(p)
        plt.close(fig)
        out.append(p)
    return out


def _int_ticks(ax):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ax.xaxis.set_major_locator(FixedLocator([2 ** k for k in range(3, 14)]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))


# ---------------------------------------------------------------------------
# (b) budget sweep
# ---------------------------------------------------------------------------

def plot_budget_sweep(rows: list[dict], out_dir: Path) -> list[Path]:
    out = []
    for model in sorted({r["model"] for r in rows}):
        b_rows = [r for r in rows if r["model"] == model and r["mode"] == "bastion"]
        avg = [r for r in b_rows if r["dataset"] == "Average"]
        fixed = sorted([r for r in avg if r.get("fixed_budget") is not None], key=lambda r: r["fixed_budget"])
        adaptive = [r for r in avg if r.get("fixed_budget") is None]
        if not fixed and not adaptive:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        if fixed:
            xs = [r["mean_tree_size"] or r["fixed_budget"] for r in fixed]
            ys = [r["speedup"] for r in fixed]
            ax.plot(xs, ys, "-o", color=BLUE, linewidth=2, markersize=6, label="BASTION-Fixed (N)")
            for r, xv, yv in zip(fixed, xs, ys):
                if yv is not None:
                    ax.annotate(f"N={r['fixed_budget']}", (xv, yv), textcoords="offset points", xytext=(0, 7),
                                ha="center", fontsize=7, color=INK2)
        for r in adaptive:
            if r.get("speedup") is not None and r.get("mean_tree_size") is not None:
                ax.plot([r["mean_tree_size"]], [r["speedup"]], marker="*", markersize=15, color=ORANGE,
                        markeredgecolor="white", linestyle="none",
                        label=f"BASTION adaptive (mean N̄={r['mean_tree_size']:.0f})")
        ax.set_xscale("log", base=2)
        _int_ticks(ax)
        ax.set_xlabel("Mean realized verification tree size (nodes, log scale)")
        ax.set_ylabel("Speedup over AR (×), mean over datasets")
        n_ds = max((len(r.get("datasets_included", [])) for r in avg), default=0)
        ax.set_title(f"{model_short(model)}: budget sweep ({n_ds} datasets)", loc="left", fontsize=11, color=INK)
        ax.legend(fontsize=8, loc="lower center")
        fig.tight_layout()
        p = out_dir / f"budget_sweep_{model_short(model)}.png"
        fig.savefig(p)
        plt.close(fig)
        out.append(p)

        per = [r for r in b_rows if r["dataset"] not in ("Average", "Pooled")]
        datasets = sorted({r["dataset"] for r in per}, key=dataset_sort_key)
        if not datasets:
            continue
        ncol = min(4, len(datasets))
        nrow = int(np.ceil(len(datasets) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.5 * nrow), squeeze=False, sharex=True)
        for ax, d in zip(axes.flat, datasets):
            fx = sorted([r for r in per if r["dataset"] == d and r.get("fixed_budget") is not None], key=lambda r: r["fixed_budget"])
            ad = [r for r in per if r["dataset"] == d and r.get("fixed_budget") is None]
            if fx:
                ax.plot([r["mean_tree_size"] or r["fixed_budget"] for r in fx], [r["speedup"] for r in fx], "-o",
                        color=BLUE, linewidth=2, markersize=4)
            for r in ad:
                if r.get("speedup") is not None and r.get("mean_tree_size"):
                    ax.plot([r["mean_tree_size"]], [r["speedup"]], marker="*", markersize=12, color=ORANGE,
                            markeredgecolor="white", linestyle="none")
            ax.set_xscale("log", base=2)
            _int_ticks(ax)
            ax.set_title(DATASET_LABELS.get(d, d), fontsize=9, loc="left")
        for ax in list(axes.flat)[len(datasets):]:
            ax.axis("off")
        fig.supxlabel("Mean realized tree size (nodes, log scale)", fontsize=9)
        fig.supylabel("Speedup over AR (×)", fontsize=9)
        fig.suptitle(f"{model_short(model)}: per-dataset budget sweep (line: fixed N, star: adaptive)", fontsize=10)
        fig.tight_layout()
        p = out_dir / f"budget_sweep_{model_short(model)}_per_dataset.png"
        fig.savefig(p)
        plt.close(fig)
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# (c) latency model
# ---------------------------------------------------------------------------

def plot_latency_model(latency_dir: Path, out_dir: Path, contexts=(64, 256, 1024)) -> list[Path]:
    meta_path = latency_dir / "latency_model.json"
    if not meta_path.exists():
        return []
    results = _load(meta_path)
    if not isinstance(results, list):
        return []
    out = []
    colors = [BLUE, ORANGE, AQUA]
    for res in results:
        pts_path = latency_dir / f"latency_points_{res['gpu_key']}_{res['model_key']}.csv"
        if not pts_path.exists():
            continue
        with open(pts_path) as fh:
            pts = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        for color, c in zip(colors, contexts):
            sel = sorted([p for p in pts if int(p["c"]) == c], key=lambda p: p["s"])
            if not sel:
                continue
            s = [p["s"] for p in sel]
            ax.plot(s, [p["measured_s"] * 1e3 for p in sel], "o", color=color, markersize=3.5, alpha=0.8, label=f"measured, c={c}")
            ax.plot(s, [p["uncal_s"] * 1e3 for p in sel], "--", color=color, linewidth=1.5)
            ax.plot(s, [p["cal_s"] * 1e3 for p in sel], "-", color=color, linewidth=2)
        a = res.get("rmse_paper_contexts") or res.get("rmse_all")
        scope = "c∈{64,256,1024}" if res.get("rmse_paper_contexts") else "all c"
        ax.plot([], [], "--", color=INK2, label=f"uncalibrated roofline (RMSE {a['rmse_uncal_ms']:.1f} ms, {scope})")
        ax.plot([], [], "-", color=INK2, label=f"calibrated fit (RMSE {a['rmse_cal_ms']:.1f} ms, {scope}, in-sample)")
        ax.set_xlabel("Verified sequence length s (tokens)")
        ax.set_ylabel("Verification latency (ms)")
        ax.set_title(f"{res['model_key']} on {res['gpu_key']}: latency model", loc="left", fontsize=11, color=INK)
        ax.legend(fontsize=7.5, loc="upper left")
        fig.tight_layout()
        p = out_dir / f"latency_model_{res['gpu_key']}_{res['model_key']}.png"
        fig.savefig(p)
        plt.close(fig)
        out.append(p)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", nargs="*", default=[], help="aggregate.py summary.json for E1 runs")
    ap.add_argument("--sweep-summary", nargs="*", default=[], help="aggregate.py summary.json for budget sweeps")
    ap.add_argument("--latency-dir", default=None, help="directory with latency_model.py outputs")
    ap.add_argument("--no-paper", action="store_true", help="omit paper-reported reference bars")
    ap.add_argument("--device-label", default="")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    rows = [r for p in args.summary for r in _load(p)]
    if rows:
        written += plot_speedup_bars(rows, out, paper=not args.no_paper, device_label=args.device_label)
    srows = [r for p in args.sweep_summary for r in _load(p)]
    if srows:
        written += plot_budget_sweep(srows, out)
    if args.latency_dir:
        written += plot_latency_model(Path(args.latency_dir), out)
    for p in written:
        print("wrote", p)
    if not written:
        print("no figures written (missing inputs)")


if __name__ == "__main__":
    main()
