"""Chunk-count sweep for the matmul -> softmax -> matmul pipeline.

Follows Performance_Metrics/README_metrics.md of krsnoki/DA4022:
  * one machine-readable JSON row per experiment configuration (--out, JSONL)
  * unmodified baseline (chunks=1) measured with identical inputs/settings
  * latency reported as median and p95 over timed iterations, with timing scope
  * unknown / not-measured values are null plus a status/notes entry, never 0
  * energy is `not_measured` unless NVML sampling actually ran
  * failures (e.g. OOM) are recorded as rows, not silently dropped

Examples
  python benchmark.py --m 32768 --k 2048 --n 131072 --dtype bfloat16 --chunks 1 2 4 8 16 32 --compile
  python benchmark.py --m 256 --k 64 --n 512 --dtype float32 --warmup 2 --iters 5 --chunks 1 2 4   # CPU smoke
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time

import torch

from chunked_ops import mm_softmax_mm
from energy import NvmlSampler

PAPER_ID = "OpenReview:QSv1lTk06x"
METHOD = "chunked_execution"
TIMING_SCOPE = "mm->softmax->mm loop only; inputs pre-allocated on device; cuda synchronized at boundaries"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def run_one(a, b, chunks, warmup, iters, compile_softmax):
    cuda = a.device.type == "cuda"
    out = torch.zeros((b.shape[1], a.shape[1]), device=a.device, dtype=a.dtype)
    for _ in range(warmup):
        mm_softmax_mm(a, b, chunks, compile_softmax=compile_softmax, out=out)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    sampler = NvmlSampler(a.device.index or 0, enabled=cuda)
    times_ms, events = [], []
    with sampler:
        wall0 = time.perf_counter()
        for _ in range(iters):
            if cuda:  # events avoid per-iteration host sync that would perturb power behaviour
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                mm_softmax_mm(a, b, chunks, compile_softmax=compile_softmax, out=out)
                e.record()
                events.append((s, e))
            else:
                t0 = time.perf_counter()
                mm_softmax_mm(a, b, chunks, compile_softmax=compile_softmax, out=out)
                times_ms.append((time.perf_counter() - t0) * 1e3)
        if cuda:
            torch.cuda.synchronize()
            times_ms = [s.elapsed_time(e) for s, e in events]
        wall = time.perf_counter() - wall0
    m, k = a.shape
    n = b.shape[1]
    flops_per_iter = 2 * (2 * m * k * n)  # two matmuls of equal cost
    es = sampler.summary()
    row = {
        "chunks": chunks,
        "latency_median_ms": statistics.median(times_ms),
        "latency_p95_ms": pct(times_ms, 0.95),
        "latency_samples": len(times_ms),
        "throughput_tflops": flops_per_iter * iters / wall / 1e12,
        "throughput_note": "TFLOP/s counting both matmuls; wall time of the whole timed loop",
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
        "peak_memory_note": "torch.cuda.max_memory_allocated" if cuda else "not_measured on CPU",
        "raw_latency_ms": times_ms,
    }
    if es.get("energy_j") is not None:
        row.update(
            energy_j_per_iter=es["energy_j"] / iters,
            energy_status="measured_nvml_estimate",
            energy_note="NVML power sampling @50ms, trapezoid integration, idle NOT subtracted; cross-check with wall-socket meter",
            avg_power_w=es["avg_power_w"],
            avg_sm_clock_mhz=es["avg_sm_clock_mhz"],
            sm_clock_std_mhz=es["sm_clock_std_mhz"],
            avg_temp_c=es["avg_temp_c"],
            temp_std_c=es["temp_std_c"],
        )
    else:
        row.update(energy_j_per_iter=None, energy_status="not_measured")
    return row, out.float().clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=32768)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--n", type=int, default=131072)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--compile", action="store_true", help="torch.compile the softmax")
    ap.add_argument("--zeros", action="store_true", help="zero matrices (data-dependent power control)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--status", default=None, help="override status label: reproduced|partial|approximate|not reproduced")
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    if 1 not in args.chunks:
        raise SystemExit("include chunks=1: the unmodified baseline is required")
    args.chunks = [1] + [c for c in args.chunks if c != 1]  # baseline first

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    mk = torch.zeros if args.zeros else torch.randn
    a = mk(args.m, args.k, device=device, dtype=dtype)
    b = mk(args.k, args.n, device=device, dtype=dtype)

    status = args.status or ("approximate" if device.type == "cpu" else "partial")
    common = {
        "paper_id": PAPER_ID,
        "method": METHOD,
        "commit": git_commit(),
        "backbone_id": None, "backbone_revision": None,
        "dataset_id": "synthetic_randn" if not args.zeros else "synthetic_zeros",
        "split": None,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else (platform.processor() or "cpu"),
        "accelerator_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if device.type == "cuda" else None,
        "driver_runtime": f"cuda {torch.version.cuda}" if device.type == "cuda" else None,
        "framework_version": f"torch {torch.__version__}",
        "python": platform.python_version(),
        "kernel_backend": "torch.compile softmax + cuBLAS matmul" if args.compile else "eager torch + cuBLAS matmul",
        "dtype": args.dtype,
        "input_shape": {"A": [args.m, args.k], "B": [args.k, args.n]},
        "batch_size": None,
        "warmup": args.warmup, "iters": args.iters,
        "timing_scope": TIMING_SCOPE,
        "quality_metric": "max_abs_diff_vs_baseline (output equivalence, not task accuracy)",
        "cost_usd_per_unit": None,
        "robustness": None,
        "status": status,
    }
    baseline_out = None
    with open(args.out, "a") as f:
        for c in args.chunks:
            exp_id = f"{METHOD}_m{args.m}_k{args.k}_n{args.n}_{args.dtype}_c{c}_s{args.seed}"
            try:
                row, out = run_one(a, b, c, args.warmup, args.iters, args.compile)
                if c == 1:
                    baseline_out = out
                    row.update(quality_value=0.0, quality_delta_vs_baseline=0.0)
                else:
                    d = (out - baseline_out).abs().max().item()
                    row.update(quality_value=d, quality_delta_vs_baseline=d)
                if c != 1:
                    base = base_row
                    row["speedup_vs_baseline"] = base["latency_median_ms"] / row["latency_median_ms"]
                else:
                    base_row = row
                row["is_baseline"] = c == 1
            except Exception as ex:  # report failures; do not silently drop (README rule 7)
                row = {"chunks": c, "is_baseline": c == 1, "status": "failed", "failure": f"{type(ex).__name__}: {str(ex)[:200]}"}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            rec = {"experiment_id": exp_id, **common, **row}
            f.write(json.dumps(rec) + "\n")
            show = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k in
                    ("chunks", "latency_median_ms", "latency_p95_ms", "throughput_tflops", "quality_delta_vs_baseline", "speedup_vs_baseline", "energy_status", "status", "failure")}
            print(show)


if __name__ == "__main__":
    main()