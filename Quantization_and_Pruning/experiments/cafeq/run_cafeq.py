#!/usr/bin/env python
"""Reproduction entry point for CafeQ and its baselines (WikiText-2 perplexity).

Examples (run from the repository root)::

    # CPU smoke run on a tiny random model: no downloads, ~seconds
    python Quantization_and_Pruning/experiments/cafeq/run_cafeq.py \
        --config Quantization_and_Pruning/configs/cafeq/tiny_smoke.yaml --tiny --baseline

    # Real run: unmodified baseline + CafeQ, 4-bit per-channel, Qwen2.5-0.5B
    python Quantization_and_Pruning/experiments/cafeq/run_cafeq.py \
        --config Quantization_and_Pruning/configs/cafeq/qwen2_5_0_5b_w4_perchannel.yaml --baseline

    # Baselines on the same config
    ... --method rtn
    ... --method random_hadamard

Outputs go to ``<output_dir>/<experiment_id>/`` (git-ignored): ``report_<method>.json`` with
per-layer errors and ``results.csv`` rows in the schema of ``Performance_Metrics/README_metrics.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from Quantization_and_Pruning.quantization.weight_only.cafeq import CafeQConfig, quantize_model  # noqa: E402

CSV_FIELDS = [
    "experiment_id", "paper_id", "method", "commit", "backbone_id", "backbone_revision", "dataset_id", "split",
    "seed", "device", "framework_version", "dtype", "batch_size", "input_length", "output_length",
    "quality_metric", "quality_value", "latency_median_ms", "latency_p95_ms", "throughput", "peak_memory_mb",
    "energy_j", "status", "notes",
]
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__}
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        env["cuda_device"] = torch.cuda.get_device_name(0)
        env["cuda_runtime"] = torch.version.cuda
    return env


def build_tiny_model(seed: int):
    """Randomly initialised 2-layer Llama-style model with GQA, for CPU smoke runs only."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    return LlamaForCausalLM(config).eval()


@torch.no_grad()
def evaluate_ppl(model, chunks: torch.Tensor, batch_size: int, device: torch.device) -> dict:
    """Perplexity over non-overlapping chunks + per-batch forward latency (prefill only)."""
    nll, n_tok, lat = 0.0, 0, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, chunks.shape[0], batch_size):
        batch = chunks[start : start + batch_size].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        logits = model(batch).logits.float()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        lat.append((time.perf_counter() - t0) * 1e3)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="sum"
        )
        nll += float(loss)
        n_tok += batch[:, 1:].numel()
    out = {
        "ppl": math.exp(nll / n_tok),
        "latency_median_ms": statistics.median(lat),
        "latency_p95_ms": sorted(lat)[max(0, math.ceil(0.95 * len(lat)) - 1)],
        "n_batches": len(lat),
    }
    out["throughput_tok_s"] = chunks.shape[1] * batch_size / (out["latency_median_ms"] / 1e3)
    out["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    return out


def csv_row(cfg: CafeQConfig, method: str, ev: dict, args, notes: str) -> dict:
    return {
        "experiment_id": cfg.experiment_id,
        "paper_id": cfg.paper_id,
        "method": method,
        "commit": git_commit(),
        "backbone_id": "tiny.llama.random" if args.tiny else cfg.backbone.id,
        "backbone_revision": "" if args.tiny else (cfg.backbone.revision or ""),
        "dataset_id": "synthetic.random_tokens" if args.tiny else cfg.evaluation.dataset_id,
        "split": "" if args.tiny else cfg.evaluation.split,
        "seed": cfg.seed,
        "device": cfg.device,
        "framework_version": f"torch {torch.__version__}",
        "dtype": cfg.backbone.dtype,
        "batch_size": cfg.evaluation.batch_size,
        "input_length": cfg.evaluation.seq_len,
        "output_length": 0,
        "quality_metric": "perplexity",
        "quality_value": round(ev["ppl"], 4),
        "latency_median_ms": round(ev["latency_median_ms"], 3),
        "latency_p95_ms": round(ev["latency_p95_ms"], 3),
        "throughput": round(ev["throughput_tok_s"], 2),
        "peak_memory_mb": "" if ev["peak_memory_mb"] is None else round(ev["peak_memory_mb"], 1),
        "energy_j": "",
        "status": "smoke_test" if args.tiny else args.status,
        "notes": notes,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--method", choices=["none", "rtn", "random_hadamard", "cafeq"], help="override quantization.method")
    p.add_argument("--baseline", action="store_true", help="also evaluate the unmodified model first")
    p.add_argument("--tiny", action="store_true", help="random tiny model + synthetic tokens (CPU smoke test)")
    p.add_argument("--no-eval", action="store_true", help="quantize only")
    p.add_argument("--max-chunks", type=int, help="override evaluation.max_chunks")
    p.add_argument("--output-dir", help="override output_dir (relative to the repo root, or absolute)")
    p.add_argument("--status", default="approximate",
                   help="result label for real runs: reproduced | partial | approximate | not reproduced")
    args = p.parse_args(argv)

    cfg = CafeQConfig.from_yaml(args.config)
    if args.method:
        cfg.quantization.method = args.method
    if args.max_chunks:
        cfg.evaluation.max_chunks = args.max_chunks
    if args.output_dir:
        cfg.output_dir = args.output_dir
    cfg.validate()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    out_dir = REPO_ROOT / cfg.output_dir / cfg.experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- model + evaluation data
    if args.tiny:
        model = build_tiny_model(cfg.seed).to(device)
        g = torch.Generator().manual_seed(cfg.seed + 1)
        n = cfg.evaluation.max_chunks or 4
        chunks = torch.randint(0, model.config.vocab_size, (n, cfg.evaluation.seq_len), generator=g)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from Dataloaders.evaluation.wikitext2 import load_wikitext2_chunks

        kw = {"revision": cfg.backbone.revision, "torch_dtype": DTYPES[cfg.backbone.dtype]}
        tokenizer = AutoTokenizer.from_pretrained(cfg.backbone.hf_name, revision=cfg.backbone.revision)
        model = AutoModelForCausalLM.from_pretrained(cfg.backbone.hf_name, **kw).to(device).eval()
        chunks = None
        if not args.no_eval:
            chunks = load_wikitext2_chunks(
                tokenizer, cfg.evaluation.split, cfg.evaluation.seq_len, cfg.evaluation.max_chunks
            )

    meta = {"config": cfg.to_dict(), "environment": environment(), "commit": git_commit(), "argv": sys.argv}
    rows = []
    fake_q = "fake-quant (dequantized fp weights; no low-bit kernels)"

    if args.baseline and not args.no_eval:
        ev = evaluate_ppl(model, chunks, cfg.evaluation.batch_size, device)
        print(f"[baseline] ppl={ev['ppl']:.4f}")
        rows.append(csv_row(cfg, "none", ev, args, "unmodified baseline"))

    report = quantize_model(model, cfg)
    report["meta"] = meta
    method = cfg.quantization.method
    print(f"[{method}] quantized in {report.get('quantization_seconds', 0):.1f}s; "
          f"estimated extra FLOPs {report.get('estimated_extra_flops_pct', 0):.2f}% (analytic, not measured)")

    if not args.no_eval and method != "none":
        ev = evaluate_ppl(model, chunks, cfg.evaluation.batch_size, device)
        report["eval"] = ev
        print(f"[{method}] ppl={ev['ppl']:.4f}")
        q = cfg.quantization
        notes = (f"w{q.bits} group={q.group_size or 'per-channel'}; {fake_q}; latency = prefill forward of one batch; "
                 f"quantization_seconds={report['quantization_seconds']:.1f}; calibration-free")
        rows.append(csv_row(cfg, method, ev, args, notes))

    (out_dir / f"report_{method}.json").write_text(json.dumps(report, indent=2, default=str))
    if rows:
        csv_path = out_dir / "results.csv"
        new = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"wrote {csv_path}")
    print(f"wrote {out_dir / f'report_{method}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
