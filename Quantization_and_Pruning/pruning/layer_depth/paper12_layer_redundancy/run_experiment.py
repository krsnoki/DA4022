"""
run_experiment.py — Paper #12: Rethinking Layer Redundancy

Ties together the depth pruner, calibration objectives, and search algorithms.
Runs all 4 experiment combinations and saves results.

Usage:
    python run_experiment.py --config config.yaml

Author: (your name)
Status: implementing
"""

import argparse
import json
import os
import yaml
import torch
from datetime import datetime

from depth_pruner import load_model_and_tokenizer, prune_layers, get_num_layers
from calibration_objectives import compute_perplexity, compute_task_likelihood_margin
from search_algorithms import one_shot_search, greedy_search


def parse_args():
    parser = argparse.ArgumentParser(description="Run layer redundancy depth pruning experiments")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test mode on CPU with tiny model")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_results(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {fname}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = "cpu" if args.dry_run else cfg["hardware"]["device"]
    dtype = torch.float32 if args.dry_run else torch.float16

    print(f"Loading model: {cfg['model']['model_id']}")
    model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["model_id"],
        revision=cfg["model"]["revision"],
        dtype=dtype,
        device=device,
    )
    print(f"Model loaded. Total layers: {get_num_layers(model)}")

    k = cfg["pruning"]["budget"]
    results = {"config": cfg, "experiments": []}

    # ------------------------------------------------------------------ #
    # Experiments: 2 objectives × 2 search algorithms = 4 runs            #
    # ------------------------------------------------------------------ #

    experiments = [
        {"objective": "perplexity", "algorithm": "one_shot"},
        {"objective": "perplexity", "algorithm": "greedy"},
        {"objective": "tlm",        "algorithm": "one_shot"},
        {"objective": "tlm",        "algorithm": "greedy"},
    ]

    for exp in experiments:
        print(f"\n--- Running: {exp['algorithm']} + {exp['objective']} ---")

        # Build the objective function
        if exp["objective"] == "perplexity":
            obj_fn = lambda m: compute_perplexity(
                m, tokenizer,
                n_samples=cfg["calibration"]["n_samples"],
                device=device,
                seed=cfg["calibration"]["seed"],
            )
        else:
            obj_fn = lambda m: compute_task_likelihood_margin(
                m, tokenizer,
                n_samples=cfg["calibration"]["n_samples"],
                device=device,
                seed=cfg["calibration"]["seed"],
            )

        # Run search
        if exp["algorithm"] == "one_shot":
            removed = one_shot_search(model, obj_fn, k=k, device=device)
        else:
            removed = greedy_search(model, obj_fn, k=k, device=device)

        print(f"  Layers removed: {removed}")

        # Prune
        pruned_model = prune_layers(model, removed, inplace=False)

        # Evaluate — TODO: add eval functions
        result = {
            "objective":       exp["objective"],
            "algorithm":       exp["algorithm"],
            "k":               k,
            "removed_layers":  removed,
            "perplexity_wikitext2": None,  # TODO: fill after implementing eval
            "hellaswag_acc":        None,
            "boolq_acc":            None,
            "status":          "pending",
        }
        results["experiments"].append(result)

    save_results(results, cfg["output"]["results_dir"])


if __name__ == "__main__":
    main()
