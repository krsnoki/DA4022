"""Compact GPTQ (Frantar et al., 2023) for simulated low-bit weights.

Block-sequential variant: transformer blocks are processed in order; for block
``b`` we run the calibration set through the *partially quantized* model (blocks
``< b`` already quantized), collect the input Hessian ``H = 2/n * X X^T`` of
every target linear in block ``b`` with forward hooks, stop the forward pass
right after block ``b``, and quantize those linears.  Weights are written back
de-quantized (simulated quantization, like ``fake_quant.py``).

The quantization grid (symmetric/asymmetric, per-channel or per-group) is the
same as in ``activation/fake_quant.py`` so that RTN and GPTQ results are
directly comparable.  Activation quantization is *not* applied during
calibration (standard GPTQ practice); apply ``apply_fake_quant(...,
weights_already_quantized=True)`` afterwards for WxAy settings.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch import nn

from ..activation.fake_quant import FakeQuantConfig, quant_dequant, quant_params, select_linears


class _StopForward(Exception):
    pass


@torch.no_grad()
def gptq_quantize_weight(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    symmetric: bool = True,
    group_size: Optional[int] = None,
    clip_ratio: float = 1.0,
    percdamp: float = 0.01,
    blocksize: int = 128,
) -> torch.Tensor:
    """Return the GPTQ-quantized (de-quantized) version of ``W [out, in]``."""
    W = W.clone().float()
    H = H.clone().float()
    cols = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diag(H))
    H += damp * torch.eye(cols, device=H.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    if group_size:
        if cols % group_size:
            raise ValueError(f"in_features {cols} not divisible by group_size {group_size}")
        blocksize = group_size
    else:
        scale, zero = quant_params(W, bits, symmetric, clip_ratio)

    Q = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        if group_size:  # group == block, statistics from the error-updated weights
            scale, zero = quant_params(W1, bits, symmetric, clip_ratio)
        for i in range(i2 - i1):
            w = W1[:, i : i + 1]
            d = Hinv1[i, i]
            q = quant_dequant(w, scale, zero, bits, symmetric)
            Q1[:, i : i + 1] = q
            err = (w - q) / d
            W1[:, i:] -= err @ Hinv1[i : i + 1, i:]
            Err1[:, i : i + 1] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q


def _block_index(name: str) -> Optional[int]:
    m = re.search(r"\.layers\.(\d+)\.", "." + name + ".")
    return int(m.group(1)) if m else None


def _find_block_module(model: nn.Module, linear_name: str) -> nn.Module:
    m = re.search(r"^(.*\.layers\.\d+)\.", linear_name)
    if not m:
        raise ValueError(f"cannot locate transformer block for {linear_name}")
    return model.get_submodule(m.group(1))


@torch.no_grad()
def gptq_quantize_model(
    model: nn.Module,
    calib_batches: Iterable,
    forward_fn: Callable[[nn.Module, object], object],
    cfg: FakeQuantConfig,
    percdamp: float = 0.01,
    log: Callable[[str], None] = print,
) -> dict:
    """Quantize the weights of every linear selected by ``cfg`` with GPTQ, in place.

    Args:
        calib_batches: a re-iterable collection of calibration batches (list).
        forward_fn: ``forward_fn(model, batch)`` runs the model on one batch.
    """
    if cfg.w_bits is None:
        return {"gptq_layers": 0}
    calib_batches = list(calib_batches)
    linears = select_linears(model, cfg)
    by_block: Dict[int, List[str]] = defaultdict(list)
    for name in linears:
        b = _block_index(name)
        if b is None:
            raise ValueError(f"GPTQ expects linears inside '.layers.N.': {name}")
        by_block[b].append(name)

    n_done = 0
    for b in sorted(by_block):
        names = by_block[b]
        H: Dict[str, torch.Tensor] = {}
        nsamples: Dict[str, int] = {}

        def make_hook(nm):
            def hook(mod, inp, out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                n = x.shape[0]
                if nm not in H:
                    H[nm] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                    nsamples[nm] = 0
                H[nm] *= nsamples[nm] / (nsamples[nm] + n)
                nsamples[nm] += n
                x = math.sqrt(2.0 / nsamples[nm]) * x
                H[nm] += x.t() @ x

            return hook

        def stop_hook(mod, inp, out):
            raise _StopForward()

        handles = [linears[nm].register_forward_hook(make_hook(nm)) for nm in names]
        handles.append(_find_block_module(model, names[0]).register_forward_hook(stop_hook))
        try:
            for batch in calib_batches:
                try:
                    forward_fn(model, batch)
                except _StopForward:
                    pass
        finally:
            for h in handles:
                h.remove()

        for nm in names:
            lin = linears[nm]
            Wq = gptq_quantize_weight(
                lin.weight.data, H[nm], cfg.w_bits, cfg.w_symmetric, cfg.w_group_size, cfg.w_clip_ratio, percdamp
            )
            lin.weight.data.copy_(Wq.to(lin.weight.dtype))
            n_done += 1
        del H
        log(f"[gptq] block {b}: quantized {len(names)} linears")
    return {"gptq_layers": n_done, "gptq_calib_batches": len(calib_batches), "gptq_percdamp": percdamp}
