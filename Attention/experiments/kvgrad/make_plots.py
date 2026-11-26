"""
Figures and per-task table for the KVgrad results, generated from the files written by run_ruler.py.

    python make_plots.py --results results_rtx5050_bf16

Reads <results>/summary.csv (and <results>/*/metrics.json for the per-task table, when present) and writes
<results>/plots/*.png and <results>/per_task_scores.csv.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical colours (validated for colour-vision deficiency; colour follows the method in every figure)
COLOURS = {
    "kvzip": "#257de0",
    "kvzip_plus": "#f47230",
    "kvgrad": "#00b894",
    "kvgrad_bound_max": "#00b894",  # same method as kvgrad (Eq. 8), our implementation
    "kvgrad_bound_sum": "#6a4ce2",
    "kvgrad_taylor_max": "#f2a600",
    "kvgrad_taylor_sum": "#ec4647",
}
MARKERS = {"kvzip": "s", "kvzip_plus": "^", "kvgrad": "o", "kvgrad_bound_max": "o", "kvgrad_bound_sum": "D",
           "kvgrad_taylor_max": "v", "kvgrad_taylor_sum": "X"}  # fmt: skip
LABELS = {
    "kvzip": "KVzip",
    "kvzip_plus": "KVzip+",
    "kvgrad": "KVgrad",
    "kvgrad_bound_max": "Eq. 8: max, magnitude (KVgrad)",
    "kvgrad_bound_sum": "Eq. 7: sum, magnitude",
    "kvgrad_taylor_max": "max, signed Taylor",
    "kvgrad_taylor_sum": "Eq. 6: sum, signed Taylor",
}
TEXT, TEXT_2, GRID = "#1f1f1f", "#5e5e5e", "#e6e6e6"
RATIO_LABEL = {0.5: "2:1", 0.75: "4:1", 0.875: "8:1", 0.938: "16:1", 0.969: "32:1"}  # run ids round to 3 decimals


def load_summary(results: Path) -> pd.DataFrame:
    s = pd.read_csv(results / "summary.csv")
    s["ratio"] = s["experiment_id"].str.extract(r"cr([0-9.]+)$", expand=False).astype(float).round(3)
    for col, pattern in [
        ("prefill_ms", r"prefill\+compress median ([0-9.]+)"),
        ("decode_ms", r"decode median ([0-9.]+)"),
    ]:
        s[col] = s["notes"].str.extract(pattern, expand=False).astype(float)
    s["cached"] = s["notes"].str.contains("scores reused without replay", regex=False)
    return s


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, fontweight="bold", pad=18)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbdbd")
    ax.tick_params(colors=TEXT_2, labelsize=8)


def _spread(values: list[float], min_gap: float) -> list[float]:
    """Nudge label positions apart so end labels never overlap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for a, b in zip(order, order[1:]):
        if placed[b] - placed[a] < min_gap:
            placed[b] = placed[a] + min_gap
    return placed


def quality_plot(
    s: pd.DataFrame, methods: list[str], ratios: list[float], title: str, subtitle: str, out: Path
) -> None:
    baseline = float(s.loc[s["method"] == "no_press", "quality_value"].iloc[0])
    x = {r: i for i, r in enumerate(ratios)}
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    ax.axhline(baseline, color="#9c9c9c", linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(-0.25, baseline + 1.5, f"no compression: {baseline:.1f}", va="bottom", fontsize=8, color=TEXT_2)
    ends = []
    for m in methods:
        d = s[(s["method"] == m) & (s["ratio"].isin(ratios))].sort_values("ratio")
        xs = [x[r] for r in d["ratio"]]
        ax.plot(xs, d["quality_value"], color=COLOURS[m], linewidth=2, marker=MARKERS[m], markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=LABELS[m], zorder=3)  # fmt: skip
        last = d.iloc[-1]
        ends.append((x[last["ratio"]], float(last["quality_value"])))
    for (xe, ye), yl in zip(ends, _spread([e[1] for e in ends], 4.0)):
        ax.text(xe + 0.1, yl, f"{ye:.1f}", va="center", fontsize=8, color=TEXT)
    ax.set_xticks(range(len(ratios)), [RATIO_LABEL[r] for r in ratios])
    ax.set_xlim(-0.3, len(ratios) - 1 + 0.95)
    ax.set_ylim(0, 100)
    fig.text(0.012, 0.905, subtitle, fontsize=8, color=TEXT_2)
    _style(ax, title, "Compression ratio (paper notation; 8:1 keeps 1 of every 8 KV pairs)",
           "RULER string match, macro over 13 tasks")  # fmt: skip
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def cost_plot(s: pd.DataFrame, out: Path) -> None:
    rows = [("No compression", s[s["method"] == "no_press"])]
    for m in ["kvzip", "kvzip_plus", "kvgrad"]:
        rows.append((LABELS[m] + " (8:1)", s[(s["method"] == m) & (s["ratio"] == 0.875)]))
    rows.append(("Eviction only, scores cached\n(our presses, 11 configs)", s[s["cached"]]))
    names = [r[0] for r in rows]
    prefill = [float(r[1]["prefill_ms"].median()) / 1000 for r in rows]
    decode = [float(r[1]["decode_ms"].median()) / 1000 for r in rows]
    memory = [float(r[1]["peak_memory_mb"].median()) / 1024 for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=160, gridspec_kw={"width_ratios": [1.6, 1]})
    y = range(len(names))[::-1]
    a1.barh(y, prefill, color="#1b64be", height=0.55, label="prefill + scoring/eviction")
    a1.barh(y, decode, left=prefill, color="#9dc8f7", height=0.55, label="decode", edgecolor="white", linewidth=2)
    for yi, p, d in zip(y, prefill, decode):
        a1.text(p + d + 0.1, yi, f"{p:.1f} + {d:.1f} s", va="center", fontsize=8, color=TEXT)
    a1.set_yticks(list(y), names, fontsize=8)
    a1.set_xlim(0, max(p + d for p, d in zip(prefill, decode)) * 1.35)
    _style(a1, "Median time per question", "seconds (median of each phase)", "")
    a1.grid(axis="x", color=GRID, linewidth=0.8)
    a1.grid(axis="y", visible=False)
    a1.legend(frameon=False, fontsize=8, loc="lower right")

    a2.barh(y, memory, color="#2370cd", height=0.55)
    for yi, mval in zip(y, memory):
        a2.text(mval + 0.08, yi, f"{mval:.1f} GiB", va="center", fontsize=8, color=TEXT)
    a2.set_yticks(list(y), [""] * len(names))
    a2.set_xlim(0, max(memory) * 1.35)
    _style(a2, "Peak GPU memory", "GiB allocated (torch.cuda.max_memory_allocated)", "")
    a2.grid(axis="x", color=GRID, linewidth=0.8)
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def per_task_table(results: Path) -> pd.DataFrame | None:
    rows = []
    for f in sorted(results.glob("*_cr*/metrics.json")):
        m = json.loads(f.read_text())
        rows.append({"configuration": f.parent.name, **m["string_match_per_task"], "macro": m["string_match_macro"]})
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    s = load_summary(args.results)
    plots = args.results / "plots"
    plots.mkdir(exist_ok=True)
    n = int(re.search(r"n=(\d+)", s["notes"].iloc[0]).group(1))
    tag = f"Qwen3-0.6B, RULER 4K ({n} questions, 13 tasks), {s['device'].iloc[0]}, {s['dtype'].iloc[0]}"
    quality_plot(s, ["kvzip", "kvzip_plus", "kvgrad"], [0.5, 0.75, 0.875], "Accuracy vs compression",
                 tag, plots / "quality_vs_compression.png")  # fmt: skip
    ablation = ["kvgrad_bound_max", "kvgrad_bound_sum", "kvgrad_taylor_max", "kvgrad_taylor_sum"]
    if s["method"].isin(ablation).any():
        quality_plot(s, ablation, [0.875, 0.938, 0.969], "Score-variant ablation (our implementation)",
                     tag, plots / "ablation_score_variants.png")  # fmt: skip
    cost_plot(s, plots / "latency_and_memory.png")
    table = per_task_table(args.results)
    if table is not None:
        table.to_csv(args.results / "per_task_scores.csv", index=False)
    print(
        f"Wrote {sorted(p.name for p in plots.glob('*.png'))}"
        + (" and per_task_scores.csv" if table is not None else "")
    )


if __name__ == "__main__":
    main()
