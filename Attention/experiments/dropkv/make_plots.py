"""Draw the two figures used in the paper card from the result CSVs.

    python Attention/experiments/dropkv/make_plots.py

Needs matplotlib. Reads the CSVs next to this file and writes
accuracy_vs_budget.png and results_perturbation/perturbation_vs_budget.png.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
STYLE = {
    "dropkv": dict(color="tab:blue", marker="o", markersize=7, label="DropKV"),
    "snapkv": dict(color="tab:orange", marker="x", markersize=9, markeredgewidth=2, linestyle="--", label="SnapKV"),
    "random": dict(color="tab:gray", marker="s", markersize=5, label="Random"),
}


def read(path):
    with open(HERE / path) as f:
        return list(csv.DictReader(f))


def accuracy_figure():
    rows = read("results/results.csv") + read("results_low_budget/results.csv")
    rows = [r for r in rows if r["input_length"] == "4006"]
    full = float([r for r in rows if r["method"] == "full"][0]["quality_value"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(full, color="black", linewidth=1, linestyle=":", label="Full cache")
    for method, style in STYLE.items():
        pts = sorted(
            {float(r["budget"]): float(r["quality_value"]) for r in rows if r["method"] == method}.items()
        )
        ax.plot([100 * b for b, _ in pts], [a for _, a in pts], **style)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks([30, 10, 5, 2, 1])
    ax.set_xticklabels(["30", "10", "5", "2", "1"])
    ax.set_xlabel("KV cache kept (% of prompt tokens)")
    ax.set_ylabel("Passkey accuracy")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Qwen2.5-0.5B, 4000-token prompts, T4 (DropKV and SnapKV overlap)", fontsize=9)
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "accuracy_vs_budget.png", dpi=130)
    plt.close(fig)


def perturbation_figure():
    rows = [r for r in read("results_perturbation/perturbation.csv") if r["context_tokens"] == "4000"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panels = [("attn_shift_mean", "Change in attention output (relative)"),
              ("next_token_kl_mean", "KL of next-token distribution")]
    for ax, (key, ylabel) in zip(axes, panels):
        for method, style in STYLE.items():
            pts = sorted({float(r["budget"]): float(r[key]) for r in rows if r["method"] == method}.items())
            ax.plot([100 * b for b, _ in pts], [v for _, v in pts], **style)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xticks([30, 10, 5, 2, 1])
        ax.set_xticklabels(["30", "10", "5", "2", "1"])
        ax.set_xlabel("KV cache kept (% of prompt tokens)")
        ax.set_ylabel(ylabel + " (lower is better)", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Effect of eviction on the model output, 4000-token prompts, 20 prompts per point", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "results_perturbation" / "perturbation_vs_budget.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    accuracy_figure()
    perturbation_figure()
    print("wrote accuracy_vs_budget.png and results_perturbation/perturbation_vs_budget.png")
