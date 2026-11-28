"""Relative-depth evaluation (affine-invariant, as in Depth Anything) and runtime numbers.

The model predicts relative inverse depth. Per image we fit scale s and shift t
so that s * pred + t matches 1 / gt on valid pixels (least squares), invert
back to depth, and score delta1 and AbsRel.
"""

import time

import numpy as np
import torch

from Dataloaders.evaluation.nyuv2 import MAX_DEPTH, MIN_DEPTH, valid_mask


def align_scale_shift(pred_disp, gt_depth, mask):
    """Least-squares s, t in inverse-depth space; returns aligned depth (same shape)."""
    p = pred_disp[mask].astype(np.float64)
    target = 1.0 / gt_depth[mask].astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, target, rcond=None)
    disp = np.clip(s * pred_disp + t, 1.0 / MAX_DEPTH, None)
    return np.clip(1.0 / disp, MIN_DEPTH, MAX_DEPTH)


def depth_metrics(pred_depth, gt_depth, mask):
    p, g = pred_depth[mask], gt_depth[mask]
    ratio = np.maximum(p / g, g / p)
    return {"delta1": float((ratio < 1.25).mean()), "abs_rel": float((np.abs(p - g) / g).mean())}


def evaluate(predict_fn, samples, log_every=50):
    """predict_fn(rgb) -> disparity at 480x640. Mean over images of per-image metrics."""
    d1, ar = [], []
    for i, (rgb, gt) in enumerate(samples, 1):
        mask = valid_mask(gt)
        pred = align_scale_shift(predict_fn(rgb), gt, mask)
        m = depth_metrics(pred, gt, mask)
        d1.append(m["delta1"])
        ar.append(m["abs_rel"])
        if log_every and i % log_every == 0:
            print(f"  {i}/{len(samples)}  delta1 {np.mean(d1):.4f}  abs_rel {np.mean(ar):.4f}")
    return {"delta1": float(np.mean(d1)), "abs_rel": float(np.mean(ar)), "num_images": len(samples)}


@torch.no_grad()
def latency_and_memory(model, pixel_values, warmup=10, iters=50):
    """Model forward only (no preprocessing), batch 1. Median / p95 ms, peak MB and raw samples."""
    cuda = pixel_values.is_cuda
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        model(pixel_values=pixel_values)
    times = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(pixel_values=pixel_values)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else None
    return {"latency_median_ms": float(np.median(times)),
            "latency_p95_ms": float(np.percentile(times, 95)),
            "throughput": 1000.0 / float(np.median(times)),
            "peak_memory_mb": peak, "samples_ms": times}
