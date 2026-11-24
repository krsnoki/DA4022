"""
baseline_runner.py — Paper #12: Rethinking Layer Redundancy

Runs the unmodified (unpruned) model and records baseline metrics.
Run this FIRST before any pruning experiments.

Usage:
    python baseline_runner.py --config config.yaml

Author: (your name)
Status: implementing
"""

import argparse
import json
import os
import yaml
import torch
from datetime import datetime

from depth_pruner import load_model_and_tokenizer, get_num_layers
from calibration_objectives import compute_perplexity, compute_task_likelihood_margin


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["hardware"]["device"]
    dtype = torch.float16

    print(f"Loading model: {cfg['model']['model_id']}")
    model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["model_id"],
        revision=cfg["model"]["revision"],
        dtype=dtype,
        device=device,
    )
    print(f"Total layers: {get_num_layers(model)}")

    results = {
        "model_id":       cfg["model"]["model_id"],
        "revision":       cfg["model"]["revision"],
        "dtype":          str(dtype),
        "device":         device,
        "n_layers":       get_num_layers(model),
        "pruning_budget": 0,
        "status":         "baseline",
    }

    # TODO: compute and record below metrics
    # results["perplexity_wikitext2"]  = compute_perplexity(model, tokenizer, ...)
    # results["hellaswag_acc"]         = evaluate_hellaswag(model, tokenizer, ...)
    # results["boolq_acc"]             = evaluate_boolq(model, tokenizer, ...)
    # results["peak_memory_mb"]        = torch.cuda.max_memory_allocated() / 1e6

    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    fname = os.path.join(
        cfg["output"]["results_dir"],
        f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline saved to {fname}")


if __name__ == "__main__":
    main()
