"""Experiment: offline simulation of two XShare variants on captured router logits.

We cannot run speculative decoding or several GPUs here, so we replay router
logits captured from consecutive greedy decode steps (see ``save_capture`` in
``run_sweep.py``) and SIMULATE the paper's variants. Every number written by
this script is APPROXIMATE. Three CSVs are produced:

A) ``overlap.csv`` (analogue of the paper's Fig. 4): mean size of the
   intersection of two tokens' top-k expert sets for k in {8, 16, 24}, for
   (1) consecutive tokens of the same request and (2) tokens of different
   requests at the same step. Averaged over layers.

B) ``speculative.csv`` (Algorithm 4): non-overlapping windows of ``s + 1``
   consecutive steps are treated as one "effective batch" (all tokens of the
   window for one layer, flattened) with ``request_ids`` = batch row. Several
   configurations (baseline, OEA, global greedy, per-request budgets) are run
   through ``xshare_route`` and compared.

C) ``expert_parallel.csv`` (Algorithms 5-6): single decode steps, with the
   experts split over G virtual GPU groups. Compares baseline, global greedy
   and GPU-aware selection at matched set sizes.

Why this is approximate:

* There is no draft model. The tokens in a window are the true consecutive
  greedy tokens of each request, not draft tokens that might be rejected.
  Real draft tokens would be less similar to the target model's routing.
* Virtual GPU placement: ``expert_groups = arange(n_experts) % G`` is an
  ASSUMED round-robin placement; no real expert-parallel system is involved.
* Router logits came from an unpruned run, so downstream effects of pruning
  (changed hidden states in later layers) are not captured. Each layer is
  treated independently.

Run (CPU is enough):

    python -m Mixture_of_Experts.experiments.xshare.simulate_variants \
        --logits results/gsm8k/router_logits_c8.pt results/gsm8k/router_logits_c32.pt \
        --out-dir results/simulations
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import Tensor

from Mixture_of_Experts.routing.xshare import XShareConfig, xshare_route

OVERLAP_KS = (8, 16, 24)
MAX_PAIRS = 20000
APPROX = "approximate"

# G -> per-GPU budgets m_g compared in section C.
EP_BUDGETS = {2: (8, 12), 4: (4, 6), 8: (2, 3)}
EP_NOTE = (
    "approximate; virtual round-robin placement (expert e on group e % G); "
    "global budget B=m_g*G is additional to warm-up, GPU-aware cap m_g*G "
    "includes warm-up, so set sizes are not exactly equal (see mean_kept_experts)"
)


# --------------------------------------------------------------------------
# Loading and small helpers
# --------------------------------------------------------------------------


def load_capture(path: Path) -> tuple[Tensor, Tensor]:
    """Return float32 logits ``[steps, layers, batch, experts]`` and bool ``valid``."""
    data = torch.load(path, map_location="cpu")
    return data["router_logits"].float(), data["valid"].bool()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"--- {path}")
    for row in rows:
        print(row)


def even_subset(items: list, limit: int) -> list:
    """At most ``limit`` items, evenly spaced (deterministic)."""
    if len(items) <= limit:
        return items
    picks = torch.linspace(0, len(items) - 1, limit).round().long().tolist()
    return [items[i] for i in picks]


def subsample(values: Tensor, generator: torch.Generator) -> Tensor:
    """Random subset of at most MAX_PAIRS values, using the seeded generator."""
    if values.numel() <= MAX_PAIRS:
        return values
    keep = torch.randperm(values.numel(), generator=generator)[:MAX_PAIRS]
    return values[keep]


def topk_mask(logits: Tensor, k: int) -> Tensor:
    """Bool mask of each token's top-k experts. Softmax is monotone, so top-k
    of the logits equals top-k of the softmax scores."""
    idx = torch.topk(logits, k, dim=-1).indices
    return torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, idx, True)


# --------------------------------------------------------------------------
# A) Overlap of top-k sets
# --------------------------------------------------------------------------


def layer_overlap_values(mask: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Overlap sizes for one layer.

    ``mask`` is ``[steps, batch, experts]`` bool. Returns 1-D tensors of
    overlaps for (same request, steps t and t+1) and (different requests,
    same step).
    """
    same = (mask[:-1] & mask[1:]).sum(-1)  # [steps-1, batch]
    same = same[valid[:-1] & valid[1:]]

    batch = mask.shape[1]
    first, second = torch.triu_indices(batch, batch, offset=1)
    diff = (mask[:, first] & mask[:, second]).sum(-1)  # [steps, pairs]
    diff = diff[valid[:, first] & valid[:, second]]
    return same.float(), diff.float()


def overlap_rows(name: str, logits: Tensor, valid: Tensor, seed: int) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    n_layers = logits.shape[1]
    rows = []
    for k in OVERLAP_KS:
        masks = topk_mask(logits, k)  # [steps, layers, batch, experts]
        sums = {"same_request_consecutive": [], "different_requests": []}
        counts = {"same_request_consecutive": 0, "different_requests": 0}
        for layer in range(n_layers):
            same, diff = layer_overlap_values(masks[:, layer], valid)
            for pair_type, values in (
                ("same_request_consecutive", same),
                ("different_requests", diff),
            ):
                values = subsample(values, generator)
                if values.numel():
                    sums[pair_type].append(float(values.mean()))
                    counts[pair_type] += values.numel()
        for pair_type, layer_means in sums.items():
            mean = sum(layer_means) / len(layer_means) if layer_means else float("nan")
            rows.append(
                {
                    "source_file": name,
                    "k": k,
                    "pair_type": pair_type,
                    "mean_overlap": round(mean, 4),
                    "n_pairs": counts[pair_type],
                }
            )
    return rows


# --------------------------------------------------------------------------
# Shared: run one xshare_route call and collect metrics
# --------------------------------------------------------------------------


def topk_mass_retained(logits_2d: Tensor, result, valid: Tensor, top_k: int) -> float:
    """Mean over valid tokens of (softmax mass on routed experts) / (mass on
    the unpruned top-k experts)."""
    probs = torch.softmax(logits_2d, dim=-1)
    full = torch.topk(probs, top_k, dim=-1).values.sum(-1)
    routed = (torch.gather(probs, -1, result.indices) * (result.weights > 0)).sum(-1)
    ratio = routed / full.clamp_min(1e-20)
    return float(ratio[valid].mean())


def run_call(logits_2d: Tensor, config: XShareConfig, valid: Tensor, **kwargs) -> dict:
    result = xshare_route(logits_2d, config, valid_mask=valid, **kwargs)
    metrics = dict(result.diagnostics)
    metrics["topk_mass_retained"] = topk_mass_retained(logits_2d, result, valid, config.top_k)
    return metrics


def mean_of(calls: list[dict], key: str) -> float:
    return round(sum(c[key] for c in calls) / len(calls), 4)


# --------------------------------------------------------------------------
# B) Speculative-decoding-aware selection (Algorithm 4)
# --------------------------------------------------------------------------


def speculative_configs(top_k: int) -> dict[str, XShareConfig]:
    def cfg(**kw) -> XShareConfig:
        return XShareConfig(top_k=top_k, gate_norm="softmax_topk", **kw)

    return {
        "baseline": cfg(),
        "oea_k0-1": cfg(warmup_k0=1, budget=0),
        "oea_k0-2": cfg(warmup_k0=2, budget=0),
        "global_k0-1_b-8": cfg(warmup_k0=1, budget=8),
        "global_k0-1_b-16": cfg(warmup_k0=1, budget=16),
        "perreq_k0-1_g-2": cfg(warmup_k0=1, per_request_budget=2, budget=0),
        "perreq_k0-1_g-4": cfg(warmup_k0=1, per_request_budget=4, budget=0),
        "perreq_k0-1_g-6": cfg(warmup_k0=1, per_request_budget=6, budget=0),
        "perreq_k0-2_g-4": cfg(warmup_k0=2, per_request_budget=4, budget=0),
    }


def speculative_rows(
    name: str, logits: Tensor, valid: Tensor, top_k: int, draft_tokens: int, max_windows: int
) -> list[dict]:
    steps, n_layers, batch, _ = logits.shape
    width = draft_tokens + 1
    starts = list(range(0, steps - width + 1, width))
    starts = even_subset(starts, max_windows)
    request_ids = torch.arange(batch).repeat(width)
    configs = speculative_configs(top_k)
    calls = {config_name: [] for config_name in configs}

    for start in starts:
        window_valid = valid[start : start + width].reshape(-1)
        if not window_valid.any():
            continue
        for layer in range(n_layers):
            logits_2d = logits[start : start + width, layer].reshape(width * batch, -1)
            for config_name, config in configs.items():
                calls[config_name].append(
                    run_call(logits_2d, config, window_valid, request_ids=request_ids)
                )

    rows = []
    for config_name, items in calls.items():
        if not items:
            continue
        rows.append(
            {
                "source_file": name,
                "config": config_name,
                "effective_batch_tokens": mean_of(items, "n_tokens"),
                "mean_kept_experts": mean_of(items, "kept_experts"),
                "mean_activated_experts": mean_of(items, "activated_experts"),
                "mean_score_mass_kept": mean_of(items, "score_mass_kept"),
                "mean_topk_mass_retained": mean_of(items, "topk_mass_retained"),
                "n_calls": len(items),
            }
        )
    return rows


# --------------------------------------------------------------------------
# C) Expert-parallel-aware selection (Algorithms 5-6)
# --------------------------------------------------------------------------


def expert_parallel_configs(top_k: int, n_groups: int) -> dict[str, XShareConfig]:
    def cfg(**kw) -> XShareConfig:
        return XShareConfig(top_k=top_k, gate_norm="softmax_topk", **kw)

    configs = {"baseline": cfg()}
    for m_g in EP_BUDGETS[n_groups]:
        configs[f"global_k0-1_b-{m_g * n_groups}"] = cfg(warmup_k0=1, budget=m_g * n_groups)
        configs[f"gpuaware_k0-1_m-{m_g}"] = cfg(warmup_k0=1, per_gpu_budget=m_g)
    return configs


def expert_parallel_rows(
    name: str, logits: Tensor, valid: Tensor, top_k: int, max_steps: int
) -> list[dict]:
    steps, n_layers, _, n_experts = logits.shape
    step_ids = even_subset([t for t in range(steps) if bool(valid[t].any())], max_steps)
    rows = []
    for n_groups in EP_BUDGETS:
        # ASSUMED placement: round-robin over virtual GPUs.
        groups = torch.arange(n_experts) % n_groups
        configs = expert_parallel_configs(top_k, n_groups)
        calls = {config_name: [] for config_name in configs}
        for t in step_ids:
            for layer in range(n_layers):
                for config_name, config in configs.items():
                    calls[config_name].append(
                        run_call(logits[t, layer], config, valid[t], expert_groups=groups)
                    )
        for config_name, items in calls.items():
            rows.append(
                {
                    "source_file": name,
                    "n_groups": n_groups,
                    "config": config_name,
                    "mean_kept_experts": mean_of(items, "kept_experts"),
                    "mean_activated_experts": mean_of(items, "activated_experts"),
                    "mean_max_kept_per_group": mean_of(items, "max_kept_per_group"),
                    "mean_max_activated_per_group": mean_of(items, "max_activated_per_group"),
                    "mean_score_mass_kept": mean_of(items, "score_mass_kept"),
                    "mean_topk_mass_retained": mean_of(items, "topk_mass_retained"),
                    "n_calls": len(items),
                    "note": EP_NOTE,
                }
            )
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--logits", type=Path, nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--draft-tokens", type=int, default=3, help="s; windows have s+1 steps")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=16, help="windows per file in B")
    parser.add_argument("--max-steps", type=int, default=24, help="decode steps per file in C")
    args = parser.parse_args()

    overlap, speculative, parallel = [], [], []
    for path in args.logits:
        logits, valid = load_capture(path)
        print(f"loaded {path} logits={tuple(logits.shape)} valid={tuple(valid.shape)}")
        overlap += overlap_rows(path.name, logits, valid, args.seed)
        speculative += speculative_rows(
            path.name, logits, valid, args.top_k, args.draft_tokens, args.max_windows
        )
        parallel += expert_parallel_rows(path.name, logits, valid, args.top_k, args.max_steps)

    print("All results are APPROXIMATE simulations on captured logits.")
    write_csv(args.out_dir / "overlap.csv", overlap)
    write_csv(args.out_dir / "speculative.csv", speculative)
    write_csv(args.out_dir / "expert_parallel.csv", parallel)


if __name__ == "__main__":
    main()
