"""Small shared timing/result helpers for serial inference experiments."""

from contextlib import contextmanager
from collections import Counter
import time
import numpy as np
import torch


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed(device, samples, name):
    synchronize(device)
    start = time.perf_counter()
    yield
    synchronize(device)
    samples[name] = samples.get(name, 0.0) + time.perf_counter() - start


def summarize(rows):
    good = [r for r in rows if r["status"] != "failed"]
    if not good:
        return {"status": "failed", "num_examples": 0, "failures": len(rows)}
    times = [r["prefill_seconds"] + r["decode_seconds"] for r in good]
    quality = [r["correct"] for r in good if r["correct"] is not None]
    peaks = [r["peak_memory_mb"] for r in good if r["peak_memory_mb"] is not None]
    decode_seconds = sum(r["decode_seconds"] for r in good)
    return {
        "num_examples": len(good), "failures": len(rows) - len(good),
        "quality_metric": "synthetic_code_exact_match" if quality else None,
        "quality_value": float(np.mean(quality)) if quality and len(good) == len(rows) else None,
        "quality_completed_only": float(np.mean(quality)) if quality else None,
        "latency_median_ms": float(np.median(times) * 1000),
        "latency_p95_ms": float(np.percentile(times, 95) * 1000) if len(times) >= 20 else None,
        "prefill_median_ms": float(np.median([r["prefill_seconds"] for r in good]) * 1000),
        "decode_median_ms": float(np.median([r["decode_seconds"] for r in good]) * 1000),
        "throughput": sum(r["output_tokens"] for r in good) / decode_seconds if decode_seconds else None,
        "peak_memory_mb": max(peaks) if peaks else None,
        "kv_bytes_max_final": max(r["kv_bytes"] for r in good),
        "query_buffer_bytes_max": max(r["query_buffer_bytes"] for r in good),
        "eviction_median_ms": float(np.median([r["eviction_seconds"] for r in good]) * 1000),
        "energy_j": None,
        "answer_categories": dict(Counter(r.get("answer_category", "unrecorded") for r in good)),
        "truncated_final_turns": sum(bool(r.get("truncated_turns", [False])[-1]) for r in good),
        "first_response_mentions_expected": sum(bool(r.get("first_response_mentions_expected")) for r in good),
        "status": "partial" if quality else "smoke_only",
        "notes": "Scripted synthetic tool observations; throughput is decode tokens/s; latency spans both turns. Failed samples counted separately; energy unmeasured.",
    }
