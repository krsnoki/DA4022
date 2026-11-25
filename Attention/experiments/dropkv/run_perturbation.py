"""Measure how much KV-cache eviction changes the model's output.

Run from the repository root:
    python -m Attention.experiments.dropkv.run_perturbation \
        --config Attention/configs/dropkv/perturbation.yaml

For every prompt the model is run once, then every (method, budget) pair is
applied to the same cache. Two quantities are recorded:

    attn_shift     ||a_kept - a_full|| / ||a_full|| of the attention output of
                   the observation-window queries, averaged over queries,
                   heads and layers. This is the quantity the DropKV score is
                   derived from (Lemma 1 of the paper).
    next_token_kl  KL(full || compressed) of the next-token distribution at the
                   first decode step. The queries of that step are not part of
                   the observation window, so this is measured on queries the
                   scoring never saw.

Nothing is timed here. The model always runs in float32 so that small
differences are not hidden by half-precision rounding.
"""

import argparse
import csv
import json
import platform
import statistics
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from Attention.methods.kv_eviction.eviction import (
    QueryRecorder,
    _get_kv,
    _set_kv,
    cache_num_layers,
    compress_cache,
    decode_step,
)
from Dataloaders.synthetic.passkey_retrieval import build_passkey_dataset

FIELDS = [
    "context_tokens", "input_length", "method", "budget", "num_examples",
    "attn_shift_mean", "attn_shift_median", "next_token_kl_mean", "next_token_kl_median",
    "top1_agreement", "lower_shift_than_snapkv", "lower_kl_than_snapkv",
    "backbone_id", "backbone_revision", "dataset_id", "seed", "device", "dtype",
]


def encode(tokenizer, prompt, device):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


@torch.no_grad()
def measure_prompt(model, input_ids, cfg, seed):
    """Returns {(method, budget): {"shift", "kl", "top1", "per_layer"}} for one prompt."""
    recorder = QueryRecorder(model, cfg["window"])
    try:
        out = model(input_ids=input_ids, use_cache=True)
    finally:
        recorder.remove()
    cache = out.past_key_values
    # eviction and decoding replace the cache tensors instead of editing them,
    # so keeping the original tensors is enough to restore the full cache
    original = [_get_kv(cache, i) for i in range(cache_num_layers(cache))]
    n = input_ids.shape[1]
    token = out.logits[:, -1].argmax(dim=-1, keepdim=True)

    ref_logits, _ = decode_step(model, token, cache, n, n)
    ref = torch.log_softmax(ref_logits.float(), dim=-1)

    results = {}
    for budget in cfg["budgets"]:
        for method in cfg["methods"]:
            for i, (k, v) in enumerate(original):
                _set_kv(cache, i, k, v)
            per_layer = []
            kept = compress_cache(
                cache, recorder.queries, method, budget, cfg["window"], cfg["kernel_size"],
                generator=torch.Generator().manual_seed(seed), shift_out=per_layer,
            )
            logits, _ = decode_step(model, token, cache, n, kept)
            logp = torch.log_softmax(logits.float(), dim=-1)
            kl = (ref.exp() * (ref - logp)).sum().item()
            results[(method, budget)] = {
                "shift": statistics.mean(per_layer) if per_layer else 0.0,
                "kl": max(kl, 0.0),
                "top1": int(logp.argmax(-1).item() == ref.argmax(-1).item()),
                "per_layer": per_layer,
            }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--context-tokens", type=int, nargs="+", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.num_samples:
        cfg["num_samples"] = args.num_samples
    if args.context_tokens:
        cfg["context_tokens"] = args.context_tokens
    out_dir = Path(args.out or cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.float32
    ).to(device).eval()

    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else (platform.processor() or platform.machine())
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": dev_name,
        "device_type": device.type,
        "dtype": "float32",
        "config": cfg,
    }
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))

    rows, layer_rows = [], []
    for n_ctx in cfg["context_tokens"]:
        samples = build_passkey_dataset(tokenizer, n_ctx, cfg["num_samples"], cfg["seed"])
        per_prompt, lengths = [], []
        for i, sample in enumerate(samples):
            ids = encode(tokenizer, sample.prompt, device)
            lengths.append(ids.shape[1])
            per_prompt.append(measure_prompt(model, ids, cfg, cfg["seed"] + i))
            print(f"n={n_ctx} prompt {i + 1}/{len(samples)}", flush=True)

        for budget in cfg["budgets"]:
            for method in cfg["methods"]:
                vals = [r[(method, budget)] for r in per_prompt]
                shifts = [x["shift"] for x in vals]
                kls = [x["kl"] for x in vals]
                row = {
                    "context_tokens": n_ctx,
                    "input_length": int(statistics.mean(lengths)),
                    "method": method,
                    "budget": budget,
                    "num_examples": len(vals),
                    "attn_shift_mean": round(statistics.mean(shifts), 6),
                    "attn_shift_median": round(statistics.median(shifts), 6),
                    "next_token_kl_mean": round(statistics.mean(kls), 6),
                    "next_token_kl_median": round(statistics.median(kls), 6),
                    "top1_agreement": round(statistics.mean(x["top1"] for x in vals), 4),
                    "lower_shift_than_snapkv": "",
                    "lower_kl_than_snapkv": "",
                    "backbone_id": cfg["backbone_id"],
                    "backbone_revision": cfg["model_revision"],
                    "dataset_id": cfg["dataset_id"],
                    "seed": cfg["seed"],
                    "device": dev_name,
                    "dtype": "float32",
                }
                if method == "dropkv" and "snapkv" in cfg["methods"]:
                    other = [r[("snapkv", budget)] for r in per_prompt]
                    row["lower_shift_than_snapkv"] = round(
                        statistics.mean(a["shift"] < b["shift"] for a, b in zip(vals, other)), 4
                    )
                    row["lower_kl_than_snapkv"] = round(
                        statistics.mean(a["kl"] < b["kl"] for a, b in zip(vals, other)), 4
                    )
                rows.append(row)
                print(
                    f"n={row['input_length']:>5} {method:>7} budget={budget:<5} "
                    f"shift={row['attn_shift_mean']:.4f} kl={row['next_token_kl_mean']:.5f} "
                    f"top1={row['top1_agreement']:.2f}",
                    flush=True,
                )
                n_layers = len(vals[0]["per_layer"])
                for layer in range(n_layers):
                    layer_rows.append({
                        "context_tokens": n_ctx, "method": method, "budget": budget, "layer": layer,
                        "attn_shift_mean": round(statistics.mean(x["per_layer"][layer] for x in vals), 6),
                    })

    with open(out_dir / "perturbation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "perturbation_per_layer.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["context_tokens", "method", "budget", "layer", "attn_shift_mean"])
        writer.writeheader()
        writer.writerows(layer_rows)
    print(f"wrote {out_dir / 'perturbation.csv'}")


if __name__ == "__main__":
    main()
