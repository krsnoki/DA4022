"""Plots for the staircase benchmark (matplotlib, headless)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_VARIANT = re.compile(r"staircase_f(\d+)_r(\d+)$")


def make_plots(summaries: list[dict[str, Any]], rows: list[dict[str, Any]], out: Path) -> list[str]:
    """Write the plots and return their file names."""
    made = []
    lat = [r for r in rows if r["phase"] == "latency" and r["status"] == "ok"]
    if not lat:
        return made
    made.append(_ttft_box(lat, out))
    made.append(_ttft_vs_first_chunk(summaries, out))
    made.append(_prompt_tokens(summaries, out))
    return [m for m in made if m]


def _ttft_box(lat: list[dict[str, Any]], out: Path) -> str:
    variants = ["baseline"] + sorted({r["variant"] for r in lat if r["variant"] != "baseline"},
                                     key=lambda v: (not v.startswith("staircase_f8"), v))
    keep = [v for v in variants if v == "baseline" or v.startswith(("staircase_f8_", "staircase_pc"))]
    data = [[r["result"]["ttft_s"] * 1000 for r in lat if r["variant"] == v and r["result"]["ttft_s"]]
            for v in keep]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(keep) + 1), keep, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("TTFT (ms, log scale)")
    ax.set_title("TTFT per request: baseline MoA vs staircase")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "ttft_baseline_vs_staircase.png", dpi=150)
    plt.close(fig)
    return "ttft_baseline_vs_staircase.png"


def _ttft_vs_first_chunk(summaries: list[dict[str, Any]], out: Path) -> str | None:
    pts: dict[int, list[tuple[int, float]]] = {}
    base = None
    for s in summaries:
        if s["phase"] != "latency" or s["scope"] != "all" or s.get("ttft_median_ms") is None:
            continue
        if s["variant"] == "baseline":
            base = s["ttft_median_ms"]
        m = _VARIANT.match(s["variant"])
        if m:
            pts.setdefault(int(m.group(2)), []).append((int(m.group(1)), s["ttft_median_ms"]))
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for r, xy in sorted(pts.items()):
        xy.sort()
        ax.plot([x for x, _ in xy], [y for _, y in xy], marker="o", label=f"staircase, redundancy={r}")
    if base is not None:
        ax.axhline(base, color="gray", linestyle="--", label="baseline MoA")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({x for xy in pts.values() for x, _ in xy}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_xlabel("first_chunk_size (tokens)")
    ax.set_ylabel("median TTFT (ms, log scale)")
    ax.set_title("TTFT vs first chunk size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "ttft_vs_first_chunk.png", dpi=150)
    plt.close(fig)
    return "ttft_vs_first_chunk.png"


def _prompt_tokens(summaries: list[dict[str, Any]], out: Path) -> str | None:
    sel = [s for s in summaries if s["phase"] == "latency" and s["scope"] == "all"
           and s.get("agg_prompt_tokens_mean") is not None]
    if not sel:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(sel)), [s["agg_prompt_tokens_mean"] for s in sel])
    ax.set_xticks(range(len(sel)), [s["variant"] for s in sel], rotation=45, ha="right")
    ax.set_ylabel("aggregator prompt tokens / request (mean)")
    ax.set_title("Prompt-token cost of re-prefilling the aggregator")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "aggregator_prompt_tokens.png", dpi=150)
    plt.close(fig)
    return "aggregator_prompt_tokens.png"
