"""Apply CafeQ (or a baseline) to a Hugging Face decoder-only model.

Supported layouts: models whose decoder blocks live at ``model.model.layers`` and expose
``self_attn.{q,k,v,o}_proj`` and ``mlp.{gate,up,down}_proj`` -- e.g. Llama, Mistral, Qwen2/Qwen2.5,
Gemma and Gemma 2. The embedding table, ``lm_head``, norms and activations are left unquantized,
as in the paper (Section 4).

Methods
-------
``none``             no change (unmodified baseline)
``rtn``              uniform round-to-nearest (``uniform`` in the paper's tables)
``random_hadamard``  block-diagonal random Hadamard on the contraction dim, inverse applied online
                     (``random`` / ``Ran*`` in the paper)
``cafeq``            learned block-diagonal transforms for q/k/gate/up/down (Sec. 3.1), learned
                     paired transform for (v, o) (App. A) + alternating adaptive rounding (App. B)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ...common.transforms import pick_block_size, random_block_diagonal
from ...common.uniform import QuantSpec, quantize_dequantize, relative_l2
from .joint_rounding import _split_vo, adaptive_rounding_vo, vo_paired_error
from .block_diagonal import effective_weight, learn_block_transform, transform_weight
from .config import CafeQConfig
from .modules import CafeQLinear
from .paired import learn_paired_transform

__all__ = [
    "AttentionGeometry",
    "attention_geometry",
    "decoder_layers",
    "apply_vo_transform",
    "quantize_model",
]

QK_NAMES = ("self_attn.q_proj", "self_attn.k_proj")
FF_NAMES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


@dataclass(frozen=True)
class AttentionGeometry:
    n_heads: int
    n_kv: int
    head_dim: int

    @property
    def group(self) -> int:
        return self.n_heads // self.n_kv


def attention_geometry(config) -> AttentionGeometry:
    n_heads = int(config.num_attention_heads)
    n_kv = int(getattr(config, "num_key_value_heads", None) or n_heads)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // n_heads
    if n_heads % n_kv:
        raise ValueError(f"num_attention_heads={n_heads} not divisible by num_key_value_heads={n_kv}")
    return AttentionGeometry(n_heads, n_kv, int(head_dim))


def decoder_layers(model: nn.Module):
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise NotImplementedError(
            f"{type(model).__name__}: expected decoder blocks at `model.model.layers`; "
            "add an adapter for this architecture"
        )
    return layers


def _get(root: nn.Module, dotted: str) -> nn.Module:
    return root.get_submodule(dotted)


def _set(root: nn.Module, dotted: str, module: nn.Module) -> None:
    parent, _, child = dotted.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, module)


# ----------------------------------------------------------------------------- single matrices

def _quantize_single(
    linear: nn.Module,
    method: str,
    spec: QuantSpec,
    cfg: CafeQConfig,
    generator: torch.Generator,
    device: torch.device,
    log: Optional[Callable[[str], None]],
):
    w = linear.weight.detach().to(device=device, dtype=torch.float32)
    bias = None if linear.bias is None else linear.bias.detach()
    in_f = w.shape[1]
    info: Dict[str, float] = {"rel_l2_rtn": relative_l2(quantize_dequantize(w, spec), w)}
    inv = None
    if method == "rtn":
        wq = quantize_dequantize(w, spec)
        info["rel_l2_final"] = info["rel_l2_rtn"]
    elif method == "random_hadamard":
        k = pick_block_size(in_f, cfg.random_baseline.block_size, power_of_two=True)
        blocks = random_block_diagonal(in_f, k, "hadamard", generator).to(device)
        inv = blocks.transpose(-1, -2).contiguous()  # orthogonal
        wq = quantize_dequantize(transform_weight(w, blocks), spec)
        info.update(block_size=float(k), rel_l2_final=relative_l2(effective_weight(wq, inv), w))
    elif method == "cafeq":
        k = pick_block_size(in_f, cfg.block_transform.block_size)
        res = learn_block_transform(w, spec, cfg.block_transform, block_size=k, generator=generator, log=log)
        wq, inv = res.weight_q, res.inv_blocks
        info.update(res.info)
    else:
        raise ValueError(f"unsupported method {method!r}")
    dtype = linear.weight.dtype
    meta = {"method": method, "bits": spec.bits, "group_size": spec.group_size}
    new = CafeQLinear.build(
        wq.to(dtype=dtype, device=linear.weight.device),
        bias,
        None if inv is None else inv.to(device=linear.weight.device),
        meta,
    )
    return new, info


# ----------------------------------------------------------------------------- paired V/O

def apply_vo_transform(attn: nn.Module, ms: List[torch.Tensor], geom: AttentionGeometry) -> None:
    """In place: ``W_v <- W_v M`` and ``W_o <- M^{-1} W_o`` per KV group (paper layout).

    In ``nn.Linear`` layout this is ``v_rows_g <- M_g^T v_rows_g``, ``v_bias_g <- v_bias_g M_g`` and
    ``o_cols_h <- o_cols_h M_g^{-T}`` for every query head ``h`` attached to KV group ``g``.
    Exactly output-preserving (up to float error) because attention mixes value vectors linearly.
    """
    v, o = attn.v_proj, attn.o_proj
    dh, r = geom.head_dim, geom.group
    if len(ms) != geom.n_kv:
        raise ValueError(f"expected {geom.n_kv} transforms, got {len(ms)}")
    with torch.no_grad():
        wv = v.weight.data.float()
        wo = o.weight.data.float()
        bv = None if v.bias is None else v.bias.data.float()
        for g, m in enumerate(ms):
            m = m.float().to(wv.device)
            m_inv_t = torch.linalg.inv(m.double()).float().t()
            sl = slice(g * dh, (g + 1) * dh)
            wv[sl] = m.t() @ wv[sl]
            if bv is not None:
                bv[sl] = bv[sl] @ m
            for h in range(g * r, (g + 1) * r):
                hs = slice(h * dh, (h + 1) * dh)
                wo[:, hs] = wo[:, hs] @ m_inv_t
        v.weight.data.copy_(wv.to(v.weight.dtype))
        o.weight.data.copy_(wo.to(o.weight.dtype))
        if bv is not None:
            v.bias.data.copy_(bv.to(v.bias.dtype))


def _quantize_vo(
    attn: nn.Module,
    method: str,
    spec: QuantSpec,
    cfg: CafeQConfig,
    geom: AttentionGeometry,
    generator: torch.Generator,
    device: torch.device,
    log: Optional[Callable[[str], None]],
) -> Dict[str, object]:
    v, o = attn.v_proj, attn.o_proj
    # clone: `.to` may alias the parameter, and apply_vo_transform edits it in place below
    wv0 = v.weight.detach().to(device=device, dtype=torch.float32).clone()
    wo0 = o.weight.detach().to(device=device, dtype=torch.float32).clone()
    pqe_rtn = vo_paired_error(
        quantize_dequantize(wv0, spec), quantize_dequantize(wo0, spec), wv0, wo0, geom.n_heads, geom.n_kv, geom.head_dim
    )
    paired_on = method == "cafeq" and cfg.paired_transform.enabled
    if not paired_on:
        info: Dict[str, object] = {"rel_pqe_rtn": pqe_rtn}
        for name in ("v_proj", "o_proj"):
            new, sub = _quantize_single(getattr(attn, name), method, spec, cfg, generator, device, log)
            setattr(attn, name, new)
            info[name] = sub
        qv, qo = attn.v_proj.weight.detach().float(), attn.o_proj.weight.detach().float()
        if method == "rtn":
            info["rel_pqe_final"] = vo_paired_error(qv, qo, wv0, wo0, geom.n_heads, geom.n_kv, geom.head_dim)
        return info

    # 1) learn one M per KV group on (W1_g, [W2_h for h in group])
    w1, w2 = _split_vo(wv0, wo0, geom.n_heads, geom.n_kv, geom.head_dim)
    r, dh, d_model = geom.group, geom.head_dim, w1.shape[1]
    w2_cat = w2.reshape(geom.n_kv, r, dh, d_model).permute(0, 2, 1, 3).reshape(geom.n_kv, dh, r * d_model)
    ms, group_info = [], []
    for g in range(geom.n_kv):
        res = learn_paired_transform(w1[g], w2_cat[g], cfg.paired_transform, generator=generator, log=log)
        ms.append(res.m)
        group_info.append(res.info)

    # 2) apply the (exact) transform in float32, then quantize the transformed pair
    apply_vo_transform(attn, ms, geom)
    wv_t = v.weight.detach().to(device=device, dtype=torch.float32)
    wo_t = o.weight.detach().to(device=device, dtype=torch.float32)
    qv, qo = quantize_dequantize(wv_t, spec), quantize_dequantize(wo_t, spec)
    pqe_learned_indep = vo_paired_error(qv, qo, wv0, wo0, geom.n_heads, geom.n_kv, geom.head_dim)
    info = {"rel_pqe_rtn": pqe_rtn, "rel_pqe_learned_independent": pqe_learned_indep, "groups": group_info}

    # 3) alternating adaptive rounding (Algorithm 1)
    if cfg.adaptive_rounding.enabled and cfg.adaptive_rounding.iterations > 0:
        qv, qo, ar = adaptive_rounding_vo(
            wv_t, wo_t, geom.n_heads, geom.n_kv, geom.head_dim, spec, spec,
            iterations=cfg.adaptive_rounding.iterations, keep_best=cfg.adaptive_rounding.keep_best,
        )
        info["adaptive_rounding_trace"] = ar["trace"]
    info["rel_pqe_final"] = vo_paired_error(qv, qo, wv0, wo0, geom.n_heads, geom.n_kv, geom.head_dim)

    meta = {"method": "cafeq_paired", "bits": spec.bits, "group_size": spec.group_size}
    attn.v_proj = CafeQLinear.build(qv.to(v.weight.dtype).to(v.weight.device), v.bias, None, meta)
    attn.o_proj = CafeQLinear.build(qo.to(o.weight.dtype).to(o.weight.device), o.bias, None, meta)
    return info


# ----------------------------------------------------------------------------- whole model

def quantize_model(model: nn.Module, cfg: CafeQConfig, log: Optional[Callable[[str], None]] = print) -> Dict[str, object]:
    """Quantize ``model`` in place according to ``cfg``; returns a JSON-serialisable report.

    No data is read: the method only looks at the weights (``cfg.calibration_dataset`` is null).
    """
    cfg.validate()
    q = cfg.quantization
    report: Dict[str, object] = {
        "method": q.method,
        "bits": q.bits,
        "group_size": q.group_size,
        "calibration_dataset": None,
        "layers": [],
    }
    if q.method == "none":
        return report

    spec = QuantSpec(q.bits, q.group_size)
    geom = attention_geometry(model.config)
    device = torch.device(cfg.device)
    gen = torch.Generator().manual_seed(cfg.seed)
    names: List[str] = []
    if q.targets.qk:
        names += list(QK_NAMES)
    if q.targets.ff:
        names += list(FF_NAMES)

    extra_flops = 0  # online x @ M^{-1}: in_features * block per token
    base_flops = 0   # dense matmuls of all quantized linears: in * out per token
    t_start = time.perf_counter()
    for li, layer in enumerate(decoder_layers(model)):
        t0 = time.perf_counter()
        entry: Dict[str, object] = {"layer": li}
        for name in names:
            lin = _get(layer, name)
            new, info = _quantize_single(lin, q.method, spec, cfg, gen, device, log)
            _set(layer, name, new)
            entry[name] = info
            base_flops += new.in_features * new.out_features
            extra_flops += new.in_features * new.online_block_size
        if q.targets.vo:
            entry["self_attn.vo"] = _quantize_vo(layer.self_attn, q.method, spec, cfg, geom, gen, device, log)
            for name in ("v_proj", "o_proj"):
                mod = getattr(layer.self_attn, name)
                base_flops += mod.in_features * mod.out_features
                extra_flops += mod.in_features * mod.online_block_size
        entry["seconds"] = time.perf_counter() - t0
        report["layers"].append(entry)
        if log:
            log(f"layer {li}: quantized ({q.method}) in {entry['seconds']:.1f}s")

    report["quantization_seconds"] = time.perf_counter() - t_start
    # Analytic estimate only (not a measurement): extra multiply-adds of the online transforms.
    report["estimated_extra_flops_pct"] = 100.0 * extra_flops / max(base_flops, 1)
    return report
