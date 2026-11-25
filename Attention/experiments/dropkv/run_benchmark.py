"""Benchmark KV-cache eviction (DropKV vs baselines) on passkey retrieval.

Run from the repository root:
    python -m Attention.experiments.dropkv.run_benchmark \
        --config Attention/configs/dropkv/qwen2_5_0_5b.yaml

Every configuration (method x budget x context length) sees exactly the same
prompts. The unmodified model is the row with method "full".

Timing scope: tokenisation and model loading are excluded. "prefill" covers
the prompt forward pass plus scoring and eviction. "decode" is per generated
token. GPU work is synchronised at every boundary.
"""

import argparse
import csv
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from Attention.methods.kv_eviction.eviction import decode_step, prefill
from Dataloaders.synthetic.passkey_retrieval import build_passkey_dataset, is_correct

SCHEMA = [
    "experiment_id", "paper_id", "method", "commit", "backbone_id", "backbone_revision",
    "dataset_id", "split", "seed", "device", "framework_version", "dtype", "batch_size",
    "input_length", "output_length", "quality_metric", "quality_value",
    "latency_median_ms", "latency_p95_ms", "throughput", "peak_memory_mb", "energy_j",
    "status", "notes",
]
EXTRA = [
    "budget", "num_examples", "prefill_median_ms", "decode_median_ms_per_token",
    "decode_peak_memory_mb", "kv_cache_mb", "kv_cache_full_mb", "agreement_with_full",
]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(values, q):
    values = sorted(values)
    idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return values[idx]


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return out.stdout.strip()
    except OSError:
        return ""


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return platform.processor() or platform.machine()


def encode(tokenizer, prompt, device):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


@torch.no_grad()
def run_request(model, input_ids, method, budget, cfg, device, seed):
    """One prompt: prefill (+ eviction) and greedy decoding. Returns timings,
    memory and the generated ids."""
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator().manual_seed(seed)

    sync(device)
    t0 = time.perf_counter()
    logits, cache, info = prefill(
        model, input_ids, method=method, budget=budget,
        window=cfg["window"], kernel_size=cfg["kernel_size"], generator=gen,
    )
    sync(device)
    t_prefill = time.perf_counter() - t0
    peak_prefill = torch.cuda.max_memory_allocated() / 2**20 if cuda else None
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    eos = model.generation_config.eos_token_id
    eos = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    position, cache_len = info["prompt_len"], info["kept"]
    tokens, step_times = [], []
    for _ in range(cfg["max_new_tokens"]):
        token = logits.argmax(dim=-1, keepdim=True)
        tokens.append(token.item())
        if token.item() in eos:
            break
        sync(device)
        t1 = time.perf_counter()
        logits, cache = decode_step(model, token, cache, position, cache_len)
        sync(device)
        step_times.append(time.perf_counter() - t1)
        position += 1
        cache_len += 1
    peak_decode = torch.cuda.max_memory_allocated() / 2**20 if cuda else None

    return {
        "tokens": tokens,
        "prefill_s": t_prefill,
        "step_s": step_times,
        "total_s": t_prefill + sum(step_times),
        "peak_prefill_mb": peak_prefill,
        "peak_decode_mb": peak_decode,
        "kv_mb": info["kv_bytes"] / 2**20,
        "kv_full_mb": info["kv_bytes_full"] / 2**20,
        "input_len": info["prompt_len"],
    }


def summarise(results, texts, samples, reference_texts):
    steps = [s for r in results for s in r["step_s"]]
    totals = [r["total_s"] * 1e3 for r in results]
    correct = [is_correct(t, s.answer) for t, s in zip(texts, samples)]
    agree = None
    if reference_texts is not None:
        agree = sum(t == r for t, r in zip(texts, reference_texts)) / len(texts)
    peaks = [r["peak_prefill_mb"] for r in results if r["peak_prefill_mb"] is not None]
    peaks_dec = [r["peak_decode_mb"] for r in results if r["peak_decode_mb"] is not None]
    return {
        "input_length": int(statistics.mean(r["input_len"] for r in results)),
        "output_length": round(statistics.mean(len(r["tokens"]) for r in results), 1),
        "quality_value": round(sum(correct) / len(correct), 4),
        "latency_median_ms": round(statistics.median(totals), 2),
        "latency_p95_ms": round(percentile(totals, 0.95), 2),
        "throughput": round(len(steps) / sum(steps), 2) if steps else "",
        "peak_memory_mb": round(max(peaks), 1) if peaks else "",
        "prefill_median_ms": round(statistics.median(r["prefill_s"] for r in results) * 1e3, 2),
        "decode_median_ms_per_token": round(statistics.median(steps) * 1e3, 3) if steps else "",
        "decode_peak_memory_mb": round(max(peaks_dec), 1) if peaks_dec else "",
        "kv_cache_mb": round(statistics.mean(r["kv_mb"] for r in results), 2),
        "kv_cache_full_mb": round(statistics.mean(r["kv_full_mb"] for r in results), 2),
        "agreement_with_full": "" if agree is None else round(agree, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
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

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.manual_seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=dtype
    ).to(device).eval()

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device_name(device),
        "device_type": device.type,
        "gpu_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20)
        if device.type == "cuda" else None,
        "dtype": str(dtype).replace("torch.", ""),
        "config": cfg,
        "warmup_requests": cfg["warmup"],
        "energy": "not_measured",
    }
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))

    configs = [("full", 1.0)] + [(m, b) for b in cfg["budgets"] for m in cfg["methods"]]
    commit = git_commit()
    rows = []
    for n_ctx in cfg["context_tokens"]:
        samples = build_passkey_dataset(tokenizer, n_ctx, cfg["num_samples"], cfg["seed"])
        inputs = [encode(tokenizer, s.prompt, device) for s in samples]
        reference = None
        for method, budget in configs:
            for _ in range(cfg["warmup"]):
                run_request(model, inputs[0], method, budget, cfg, device, cfg["seed"])
            results = [
                run_request(model, x, method, budget, cfg, device, cfg["seed"] + i)
                for i, x in enumerate(inputs)
            ]
            texts = [tokenizer.decode(r["tokens"], skip_special_tokens=True) for r in results]
            if method == "full":
                reference = texts
            stats = summarise(results, texts, samples, None if method == "full" else reference)
            row = {
                "experiment_id": f"dropkv-{cfg['backbone_id']}-n{n_ctx}-{method}-b{int(budget * 100)}",
                "paper_id": "dropkv",
                "method": method,
                "commit": commit,
                "backbone_id": cfg["backbone_id"],
                "backbone_revision": cfg["model_revision"],
                "dataset_id": cfg["dataset_id"],
                "split": "synthetic",
                "seed": cfg["seed"],
                "device": env["device"],
                "framework_version": f"torch {torch.__version__} / transformers {transformers.__version__}",
                "dtype": env["dtype"],
                "batch_size": 1,
                "quality_metric": "passkey_accuracy",
                "energy_j": "",
                "status": "partial",
                "notes": f"window={cfg['window']} kernel={cfg['kernel_size']} energy not measured",
                "budget": budget,
                "num_examples": len(samples),
                **stats,
            }
            rows.append(row)
            print(
                f"n={stats['input_length']:>5} {method:>7} budget={budget:<4} "
                f"acc={stats['quality_value']:.2f} prefill={stats['prefill_median_ms']:.0f}ms "
                f"decode={stats['decode_median_ms_per_token']}ms/tok kv={stats['kv_cache_mb']}MB",
                flush=True,
            )
            with open(out_dir / "results.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=SCHEMA + EXTRA)
                writer.writeheader()
                writer.writerows(rows)
    print(f"wrote {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
