"""Experiment 1: how many experts does a batch activate?

The XShare paper (Sec. 1) gives the expected number of activated experts
under uniform, independent top-k routing:

    E[N_a] = N * (1 - (1 - k/N) ** B)

with N experts, top-k routing and B tokens in the batch. This script compares

* the formula,
* a simulation with uniform random routing (should match the formula),
* a simulation with skewed router logits (real routers are not uniform), and
* activated-expert counts measured on a real model, from router logits
  saved by ``run_sweep.py`` (unpruned decode steps).

Run from the repository root (CPU is enough):

    python -m Mixture_of_Experts.experiments.xshare.expected_activation \
        --n-experts 64 --top-k 8 --batches 1 2 3 4 6 8 12 16 24 32 \
        --capture Mixture_of_Experts/experiments/xshare/results/gsm8k/router_logits_c8.pt Mixture_of_Experts/experiments/xshare/results/gsm8k/router_logits_c32.pt \
        --out Mixture_of_Experts/experiments/xshare/results/expected_activation.csv

The router-logit files are written by ``run_sweep.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from Mixture_of_Experts.routing.xshare import XShareConfig, xshare_route


def expected_activated(n_experts: int, top_k: int, batch: int) -> float:
    """The paper's closed form."""
    return n_experts * (1.0 - (1.0 - top_k / n_experts) ** batch)


def simulated_activated(
    n_experts: int,
    top_k: int,
    batch: int,
    trials: int,
    skew: float,
    generator: torch.Generator,
) -> float:
    """Mean activated experts over random batches.

    ``skew`` is the standard deviation of a per-trial expert bias added to
    every token's logits. ``0`` gives uniform, independent routing. Larger
    values make all tokens of a batch prefer the same experts, which is what
    lowers the count in practice.
    """
    config = XShareConfig(top_k=top_k)  # no pruning: baseline routing
    total = 0
    for _ in range(trials):
        logits = torch.randn(batch, n_experts, generator=generator)
        if skew > 0:
            logits = logits + skew * torch.randn(1, n_experts, generator=generator)
        total += xshare_route(logits, config).diagnostics["activated_experts"]
    return total / trials


def measured_activated(paths: list[Path], top_k: int) -> tuple[dict, dict]:
    """Mean activated experts per MoE layer call, grouped by live tokens in the call.

    Each file holds ``router_logits [steps, layers, batch, experts]`` and
    ``valid [steps, batch]`` from unpruned decode steps. A layer call with
    ``B`` live (unfinished) sequences is one batch of ``B`` tokens.
    """
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for path in paths:
        data = torch.load(path)
        logits, valid = data["router_logits"].float(), data["valid"]
        for step in range(logits.shape[0]):
            rows = valid[step]
            live = int(rows.sum())
            if live == 0:
                continue
            for layer in range(logits.shape[1]):
                chosen = torch.topk(logits[step, layer][rows], top_k, dim=-1).indices
                totals[live] = totals.get(live, 0.0) + chosen.unique().numel()
                counts[live] = counts.get(live, 0) + 1
    return {b: totals[b] / counts[b] for b in totals}, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 96])
    parser.add_argument("--skews", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--capture",
        type=Path,
        nargs="*",
        default=[],
        help="router-logit files saved by run_sweep.py (baseline decode steps of a real model)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    measured, measured_n = measured_activated(args.capture, args.top_k)

    generator = torch.Generator().manual_seed(args.seed)
    rows = []
    for batch in args.batches:
        row = {
            "n_experts": args.n_experts,
            "top_k": args.top_k,
            "batch_tokens": batch,
            "formula": round(expected_activated(args.n_experts, args.top_k, batch), 3),
        }
        for skew in args.skews:
            row[f"simulated_skew_{skew:g}"] = round(
                simulated_activated(
                    args.n_experts, args.top_k, batch, args.trials, skew, generator
                ),
                3,
            )
        row["measured_real_model"] = round(measured[batch], 3) if batch in measured else ""
        row["measured_layer_calls"] = measured_n.get(batch, 0)
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
