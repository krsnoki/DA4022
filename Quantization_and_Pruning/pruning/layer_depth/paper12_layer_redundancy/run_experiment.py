"""
run_experiment.py — Paper #12: Rethinking Layer Redundancy

Runs all 4 experiment combinations:
  2 calibration objectives  × 2 search algorithms
  = 4 pruned models

For each pruned model, evaluates:
  - Perplexity on WikiText-2
  - Zero-shot accuracy on HellaSwag and BoolQ

Usage:
    python run_experiment.py --config config.yaml
    python run_experiment.py --config config.yaml --dry-run   # CPU smoke test
"""

import argparse
import json
import os
import sys
import time
import yaml
import torch
from datetime import datetime

from depth_pruner import load_model_and_tokenizer, prune_layers, get_num_layers
from calibration_objectives import compute_perplexity, compute_task_likelihood_margin
from search_algorithms import one_shot_search, greedy_search
from evaluate import evaluate_all


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper #12: Layer redundancy depth pruning experiments"
    )
    parser.add_argument("--config",    type=str, default="config.yaml")
    parser.add_argument("--dry-run",   action="store_true",
                        help="CPU smoke test with tiny dummy model")
    parser.add_argument("--objective", type=str, default="all",
                        choices=["all", "perplexity", "tlm"],
                        help="Run only one objective type")
    parser.add_argument("--algorithm", type=str, default="all",
                        choices=["all", "one_shot", "greedy"],
                        help="Run only one search algorithm")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_results(results: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(
        out_dir,
        f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {fname}")
    return fname


def build_objective_fn(objective: str, model_ref, tokenizer, cfg, device):
    """Return a callable(model) → float for the given objective name."""
    n  = cfg["calibration"]["n_samples"]
    ml = cfg["calibration"]["max_length"]
    bs = cfg["calibration"]["batch_size"]
    sd = cfg["calibration"]["seed"]

    if objective == "perplexity":
        def fn(m):
            return compute_perplexity(
                m, tokenizer,
                dataset_name=cfg["calibration"]["perplexity"]["dataset"],
                dataset_config=cfg["calibration"]["perplexity"].get("dataset_revision", "en"),
                split=cfg["calibration"]["perplexity"]["dataset_split"],
                n_samples=n, max_length=ml, batch_size=bs,
                device=device, seed=sd,
            )
        return fn

    elif objective == "tlm":
        def fn(m):
            return compute_task_likelihood_margin(
                m, tokenizer,
                dataset_name=cfg["calibration"]["task_likelihood_margin"]["dataset"],
                split=cfg["calibration"]["task_likelihood_margin"]["dataset_split"],
                n_samples=n, device=device, seed=sd,
            )
        return fn

    raise ValueError(f"Unknown objective: {objective}")


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = "cpu"    if args.dry_run else cfg["hardware"]["device"]
    dtype  = torch.float32 if args.dry_run else torch.float16

    # ------------------------------------------------------------------ #
    # Load model                                                           #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"Paper #12 — Rethinking Layer Redundancy")
    print(f"Model  : {cfg['model']['model_id']}")
    print(f"Device : {device}  |  dtype: {dtype}")
    print(f"Budget : k={cfg['pruning']['budget']} layers to remove")
    print(f"{'='*60}\n")

    model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["model_id"],
        revision=cfg["model"]["revision"],
        dtype=dtype,
        device=device,
    )
    n_layers = get_num_layers(model)
    print(f"Model loaded. Total Transformer blocks: {n_layers}")

    k = cfg["pruning"]["budget"]

    # ------------------------------------------------------------------ #
    # Baseline evaluation (no pruning)                                     #
    # ------------------------------------------------------------------ #
    print("\n--- Baseline (unpruned) ---")
    t0       = time.time()
    baseline_metrics = evaluate_all(model, tokenizer, device=device, seed=cfg["calibration"]["seed"])
    baseline_time    = time.time() - t0
    print(f"  Perplexity (WikiText-2): {baseline_metrics['perplexity_wikitext2']}")
    print(f"  HellaSwag accuracy:      {baseline_metrics['hellaswag_acc']}%")
    print(f"  BoolQ accuracy:          {baseline_metrics['boolq_acc']}%")

    results = {
        "meta": {
            "model_id":  cfg["model"]["model_id"],
            "revision":  cfg["model"]["revision"],
            "dtype":     str(dtype),
            "device":    device,
            "n_layers":  n_layers,
            "k":         k,
            "seed":      cfg["calibration"]["seed"],
            "timestamp": datetime.now().isoformat(),
        },
        "baseline": {**baseline_metrics, "eval_time_s": round(baseline_time, 1)},
        "experiments": [],
    }

    # ------------------------------------------------------------------ #
    # Experiment matrix                                                    #
    # 4 experiments: 2 objectives x 2 search algorithms (Table 1 partial) #
    # ------------------------------------------------------------------ #
    objectives = ["perplexity", "tlm"]    if args.objective == "all" else [args.objective]
    algorithms = ["one_shot",   "greedy"] if args.algorithm == "all" else [args.algorithm]

    for obj_name in objectives:
        obj_fn = build_objective_fn(obj_name, model, tokenizer, cfg, device)

        for alg_name in algorithms:
            print(f"\n{'='*60}")
            print(f"Experiment: {alg_name}  +  {obj_name}")
            print(f"{'='*60}")

            # Search: find which layers to remove
            t_search = time.time()
            if alg_name == "one_shot":
                removed = one_shot_search(model, obj_fn, k=k, device=device)
            else:
                removed = greedy_search(model, obj_fn, k=k, device=device)
            search_time = time.time() - t_search

            print(f"\nLayers removed (original indices): {removed}")
            print(f"Search time: {search_time:.1f}s")

            # Prune
            pruned_model = prune_layers(model, removed, inplace=False)
            print(f"Pruned model layers: {get_num_layers(pruned_model)}")

            # Evaluate pruned model
            print("\nEvaluating pruned model:")
            t_eval = time.time()
            metrics = evaluate_all(pruned_model, tokenizer, device=device,
                                   seed=cfg["calibration"]["seed"])
            eval_time = time.time() - t_eval

            # Compute deltas vs baseline
            ppl_delta   = metrics["perplexity_wikitext2"] - baseline_metrics["perplexity_wikitext2"]
            hella_delta = metrics["hellaswag_acc"]        - baseline_metrics["hellaswag_acc"]
            boolq_delta = metrics["boolq_acc"]            - baseline_metrics["boolq_acc"]

            print(f"\n  Perplexity (WikiText-2) : {metrics['perplexity_wikitext2']}  (Δ {ppl_delta:+.2f})")
            print(f"  HellaSwag accuracy      : {metrics['hellaswag_acc']}%  (Δ {hella_delta:+.2f} pp)")
            print(f"  BoolQ accuracy          : {metrics['boolq_acc']}%  (Δ {boolq_delta:+.2f} pp)")

            results["experiments"].append({
                "objective":              obj_name,
                "algorithm":              alg_name,
                "k":                      k,
                "removed_layers":         removed,
                "search_time_s":          round(search_time, 1),
                "eval_time_s":            round(eval_time, 1),
                **metrics,
                "delta_perplexity":       round(ppl_delta,   4),
                "delta_hellaswag_acc":    round(hella_delta, 2),
                "delta_boolq_acc":        round(boolq_delta, 2),
                "status": "partial-reproduction",
            })

            del pruned_model

    save_results(results, cfg["output"]["results_dir"])
    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
