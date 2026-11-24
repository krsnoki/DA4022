"""Aggregate ``results.csv`` over seeds into report tables (and optional plots).

    python Quantization_and_Pruning/experiments/jacobian_noise/summarize.py \
        --results outputs/jacobian_noise/eval/results.csv --out outputs/jacobian_noise/summary [--plot]

Writes ``summary.csv`` (mean/std/n per backbone x train method x quantization x
metric) and ``summary.md`` (one table per backbone and metric: rows = training
method, columns = quantization setting, cells = mean +- std over seeds, plus the
difference to the no-noise fine-tuning baseline).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict

QUANT_ORDER = ["FP16", "W8A8", "W6A8", "W6A6", "W5A5", "W4A8", "W4A4", "W4A4QK4"]
METHOD_ORDER = ["pretrained(no-finetune)", "finetune(no-noise)", "finetune+uniform_matched_noise"]
BASELINE = "finetune(no-noise)"
LOWER_IS_BETTER = ("ppl",)


def quant_key(q: str):
    tag = q.split("-")[-1]
    return (QUANT_ORDER.index(tag) if tag in QUANT_ORDER else 99, q)


def method_key(m: str):
    return (METHOD_ORDER.index(m) if m in METHOD_ORDER else 50, m)


def load(path):
    # de-duplicate: the same (experiment_id, metric) can be written twice, e.g. the FP16
    # row is re-evaluated in both the rtn and the gptq pass; the last row wins.
    latest = {}
    for r in csv.DictReader(open(path)):
        if r["quality_value"]:
            latest[(r["experiment_id"], r["quality_metric"])] = r
    groups = defaultdict(list)
    for r in latest.values():
        train, _, quant = r["method"].partition(" | ")
        groups[(r["backbone_id"], r["quality_metric"], train, quant)].append(float(r["quality_value"]))
    return groups


def fmt(mean, std, n, metric):
    pct = not any(s in metric for s in LOWER_IS_BETTER) and mean <= 1.0
    if pct:
        return f"{100 * mean:.2f} ± {100 * std:.2f}" if n > 1 else f"{100 * mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}" if n > 1 else f"{mean:.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    groups = load(args.results)

    stats = {}
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone_id", "metric", "train_method", "quantization", "mean", "std", "n_seeds"])
        for k in sorted(groups):
            v = groups[k]
            mean = statistics.fmean(v)
            std = statistics.stdev(v) if len(v) > 1 else 0.0
            stats[k] = (mean, std, len(v))
            w.writerow([*k, f"{mean:.6g}", f"{std:.6g}", len(v)])

    lines = ["# Jacobian-guided noise injection: local results", "",
             "Accuracy-type metrics are shown in %, perplexity raw. Mean ± std over seeds.",
             "Δ = method − no-noise fine-tuning baseline (positive Δ is better for accuracy, negative for perplexity).",
             "All quantization is simulated (fake-quant).", ""]
    by_table = defaultdict(dict)
    for (bb, metric, train, quant), s in stats.items():
        by_table[(bb, metric)][(train, quant)] = s
    for (bb, metric), cells in sorted(by_table.items()):
        quants = sorted({q for _, q in cells}, key=quant_key)
        methods = sorted({m for m, _ in cells}, key=method_key)
        lines += [f"## {bb} — {metric}", "", "| train method | " + " | ".join(quants) + " |",
                  "|---|" + "---:|" * len(quants)]
        for m in methods:
            row = []
            for q in quants:
                if (m, q) not in cells:
                    row.append("")
                    continue
                mean, std, n = cells[(m, q)]
                txt = fmt(mean, std, n, metric)
                if m != BASELINE and (BASELINE, q) in cells:
                    d = mean - cells[(BASELINE, q)][0]
                    pct = not any(s in metric for s in LOWER_IS_BETTER) and mean <= 1.0
                    txt += f" (Δ {100 * d:+.2f})" if pct else f" (Δ {d:+.2f})"
                row.append(txt)
            lines.append(f"| {m} | " + " | ".join(row) + " |")
        lines.append("")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for (bb, metric), cells in sorted(by_table.items()):
            quants = sorted({q for _, q in cells}, key=quant_key)
            fig, ax = plt.subplots(figsize=(7, 4))
            for m in sorted({m for m, _ in cells}, key=method_key):
                xs = [i for i, q in enumerate(quants) if (m, q) in cells]
                ys = [cells[(m, quants[i])][0] for i in xs]
                es = [cells[(m, quants[i])][1] for i in xs]
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=m)
            ax.set_xticks(range(len(quants)), quants, rotation=30)
            ax.set_ylabel(metric)
            if any(s in metric for s in LOWER_IS_BETTER):
                ax.set_yscale("log")
            ax.set_title(bb)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"{bb}__{metric.replace('/', '_')}.png"), dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    main()
