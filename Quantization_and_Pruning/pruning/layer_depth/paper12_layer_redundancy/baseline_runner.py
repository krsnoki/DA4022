"""
baseline_runner.py — Paper #12: Rethinking Layer Redundancy

Runs the UNMODIFIED (unpruned) model and records baseline metrics.
Run this FIRST before any pruning experiments so you have reference numbers.

Usage:
    python baseline_runner.py --config config.yaml
"""

import argparse
import json
import os
import time
import yaml
import torch
from datetime import datetime

from depth_pruner import load_model_and_tokenizer, get_num_layers
from evaluate import evaluate_all


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline evaluation — no pruning")
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["hardware"]["device"]
    dtype  = torch.float16

    print(f"Loading model: {cfg['model']['model_id']}")
    model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["model_id"],
        revision=cfg["model"]["revision"],
        dtype=dtype,
        device=device,
    )
    n_layers = get_num_layers(model)
    print(f"Total Transformer blocks: {n_layers}")

    # Peak memory before eval
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0      = time.time()
    metrics = evaluate_all(
        model, tokenizer,
        device=device,
        seed=cfg["calibration"]["seed"],
    )
    elapsed = time.time() - t0

    peak_mb = (
        torch.cuda.max_memory_allocated() / 1e6
        if device == "cuda" else None
    )

    results = {
        "model_id":              cfg["model"]["model_id"],
        "revision":              cfg["model"]["revision"],
        "dtype":                 str(dtype),
        "device":                device,
        "n_layers":              n_layers,
        "pruning_budget":        0,
        "seed":                  cfg["calibration"]["seed"],
        "timestamp":             datetime.now().isoformat(),
        "eval_time_s":           round(elapsed, 1),
        "peak_memory_mb":        round(peak_mb, 1) if peak_mb else "not_measured",
        "status":                "baseline",
        **metrics,
    }

    print(f"\n--- Baseline Results ---")
    print(f"  Perplexity (WikiText-2) : {metrics['perplexity_wikitext2']}")
    print(f"  HellaSwag accuracy      : {metrics['hellaswag_acc']}%")
    print(f"  BoolQ accuracy          : {metrics['boolq_acc']}%")
    if peak_mb:
        print(f"  Peak GPU memory         : {peak_mb:.1f} MB")

    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    fname = os.path.join(
        cfg["output"]["results_dir"],
        f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline saved → {fname}")


if __name__ == "__main__":
    main()
