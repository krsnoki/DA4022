"""Uniform asymmetric fake quantization (paper eq. 3-4) and range helpers.

Everything here is a plain function on tensors. The modules in qmodules.py
decide *which* range to use; this file only does the arithmetic, so MinMax,
Percentile and ORA-Q all share the same rounding code.

Fake quantization = quantize to integers, then immediately dequantize back to
float. The result has the rounding error of real INT-b arithmetic but still
runs as float, so it measures accuracy, not speed.
"""

import torch

EPS = 1e-8


def qparams_from_range(lo, hi, bits):
    """Scale and zero point for the range [lo, hi] (eq. 4)."""
    qmax = 2 ** bits - 1
    scale = ((hi - lo) / qmax).clamp(min=EPS)
    zero_point = torch.round(-lo / scale)
    return scale, zero_point


def fake_quant(x, scale, zero_point, bits):
    """Quantize then dequantize (eq. 3). scale/zero_point must broadcast to x."""
    qmax = 2 ** bits - 1
    q = torch.clamp(torch.round(x / scale) + zero_point, 0, qmax)
    return scale * (q - zero_point)


def to_groups(x, axis, g):
    """Move `axis` last and split it into groups of size g.

    (..., C) -> (..., C // g, g). If C is not a multiple of g, the last slice
    is repeated as padding (only matters for odd token counts in matmul2).
    g=None means one group over the whole axis, i.e. per-tensor along it.
    Returns the grouped tensor and the info needed to undo it.
    """
    x = x.movedim(axis, -1)
    c = x.shape[-1]
    g = c if g is None else g
    pad = (-c) % g
    if pad:
        x = torch.cat([x, x[..., -1:].expand(*x.shape[:-1], pad)], dim=-1)
    xg = x.reshape(*x.shape[:-1], -1, g)
    return xg, (axis, c)


def from_groups(xg, info):
    """Exact inverse of to_groups."""
    axis, c = info
    x = xg.reshape(*xg.shape[:-2], -1)[..., :c]
    return x.movedim(-1, axis)


def reduce_dims(ndim, keep):
    """All dims of an ndim tensor except the ones in `keep` (negatives allowed)."""
    keep = {d % ndim for d in keep}
    return [d for d in range(ndim) if d not in keep]


def minmax(xg, keep):
    dims = reduce_dims(xg.dim(), keep)
    return xg.amin(dim=dims, keepdim=True), xg.amax(dim=dims, keepdim=True)


def percentile_range(samples, p):
    """Two-sided percentile range per row. samples: (rows, n)."""
    q = torch.tensor([1.0 - p, p], dtype=samples.dtype, device=samples.device)
    # torch.quantile refuses inputs over 16M elements, so go in row chunks
    step = max(1, 16_000_000 // samples.shape[1])
    los, his = [], []
    for i in range(0, samples.shape[0], step):
        lo, hi = torch.quantile(samples[i:i + step], q, dim=1)
        los.append(lo)
        his.append(hi)
    return torch.cat(los), torch.cat(his)


def best_percentile(samples, ps, bits):
    """Pick the p in `ps` whose range gives the least squared error on `samples`.

    One p per quantizer (shared by all its groups), as in Sec. 3.3.
    Returns (p, lo, hi) with lo/hi of shape (rows,).
    """
    best = None
    for p in ps:
        lo, hi = percentile_range(samples, p)
        scale, zp = qparams_from_range(lo[:, None], hi[:, None], bits)
        err = (fake_quant(samples, scale, zp, bits) - samples).pow(2).sum().item()
        if best is None or err < best[0]:
            best = (err, p, lo, hi)
    return best[1], best[2], best[3]


def fake_quant_weight(w, bits):
    """Per-output-channel min-max weight quantization (dim 0 = output channel).

    Works for Linear (out, in) and Conv2d (out, in, kh, kw). The paper does not
    state the weight granularity; we use this for every method.
    """
    flat = w.reshape(w.shape[0], -1)
    lo, hi = flat.amin(dim=1, keepdim=True), flat.amax(dim=1, keepdim=True)
    scale, zp = qparams_from_range(lo, hi, bits)
    return fake_quant(flat, scale, zp, bits).reshape_as(w)


def subsample(x, n):
    """Random n values from a flat tensor (torch.quantile caps out at 16M)."""
    x = x.reshape(-1)
    if x.numel() <= n:
        return x
    idx = torch.randint(0, x.numel(), (n,), device=x.device)
    return x[idx]
