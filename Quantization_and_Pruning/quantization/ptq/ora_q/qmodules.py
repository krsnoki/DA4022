"""Quantized building blocks and model surgery for Depth Anything (HF transformers).

Pieces:
  ActQuantizer       sits on one activation tensor; can be off / observe / quant / clip
  QuantLinear/Conv   replace nn.Linear / nn.Conv2d, 4-bit weights + an ActQuantizer on the input
  oraq_attention     same math as transformers' eager attention, with quantizers on Q, K (matmul1)
                     and A, V (matmul2)
  quantize_model     walks the model, swaps layers, attaches attention quantizers

Operator labels follow the paper: qkv, matmul1, matmul2, proj, fc1, fc2, conv.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AttentionInterface

from .quantizer import (best_percentile, fake_quant, fake_quant_weight, from_groups, minmax,
                        qparams_from_range, reduce_dims, subsample, to_groups)

PERCENTILES = (0.999, 0.9999, 0.99999)
SAMPLE_BUDGET = 32768   # values kept per image per quantizer for percentile search
CLIP_Q = 0.995          # top 0.5% clipping used by the clip score (Sec. 2.3 / 3.2)


class ActQuantizer(nn.Module):
    """Fake quantizer for one activation tensor.

    cfg keys:
      g         group size along `axis` (None = one group = per-tensor)
      per_head  give each attention head its own scales (dim 1 of a (B, H, N, d) tensor)
      calib     'minmax' or 'percentile'
      mode      'static' (scales fixed after calibration) or 'dynamic' (computed per input)
    """

    def __init__(self, op, cfg, bits, axis):
        super().__init__()
        self.op = op
        self.bits = bits
        self.axis = axis
        self.g = cfg.get("g")
        self.per_head = cfg.get("per_head", False)
        self.calib = cfg.get("calib", "minmax")
        self.mode = cfg.get("mode", "static")
        self.state = "off"
        self.record_ranges = False
        self.chosen_p = None
        self.register_buffer("scale", None)
        self.register_buffer("zero_point", None)
        self._reset_stats()

    def _reset_stats(self):
        self._lo = self._hi = None
        self._samples = []
        self.ranges = []        # per-image group ranges, for the stability score S

    def _keep(self):
        keep = [-2]             # the group index
        if self.per_head:
            keep.append(1)
        return keep

    def forward(self, x):
        if self.state == "off":
            return x
        if self.state == "clip":
            hi = torch.quantile(subsample(x.detach().float(), 2_000_000), CLIP_Q)
            return x.clamp(max=hi.to(x.dtype))

        xg, info = to_groups(x, self.axis, self.g)
        keep = self._keep()
        if self.state == "observe":
            self._observe(xg.detach(), keep)
            return x

        # state == "quant"
        if self.mode == "dynamic":
            lo, hi = minmax(xg, keep + [0])      # also keep batch: ranges per input
            scale, zp = qparams_from_range(lo, hi, self.bits)
        else:
            scale, zp = self.scale, self.zero_point
        return from_groups(fake_quant(xg, scale, zp, self.bits), info)

    def _observe(self, xg, keep):
        lo, hi = minmax(xg, keep)
        self._lo = lo if self._lo is None else torch.minimum(self._lo, lo)
        self._hi = hi if self._hi is None else torch.maximum(self._hi, hi)

        if self.mode == "static" and self.calib == "percentile":
            rows = _rows(xg, keep)
            m = max(256, SAMPLE_BUDGET // rows.shape[0])
            if rows.shape[1] > m:
                idx = torch.randint(0, rows.shape[1], (m,), device=rows.device)
                rows = rows[:, idx]
            self._samples.append(rows.float().cpu())

        if self.record_ranges:
            lo_b, hi_b = minmax(xg, keep + [0])
            for b in range(xg.shape[0]):
                self.ranges.append((hi_b[b] - lo_b[b]).flatten().float().cpu())

    @torch.no_grad()
    def finalize(self):
        """Turn collected statistics into static scale / zero point."""
        if self.mode == "dynamic" or self._lo is None:
            self._samples = []
            return
        lo, hi = self._lo, self._hi
        if self.calib == "percentile":
            samples = torch.cat(self._samples, dim=1)
            self.chosen_p, plo, phi = best_percentile(samples, PERCENTILES, self.bits)
            lo = plo.reshape(lo.shape).to(lo.device)
            hi = phi.reshape(hi.shape).to(hi.device)
        self.scale, self.zero_point = qparams_from_range(lo, hi, self.bits)
        self._samples = []

    def extra_repr(self):
        return (f"op={self.op}, bits={self.bits}, axis={self.axis}, g={self.g}, "
                f"per_head={self.per_head}, calib={self.calib}, mode={self.mode}, state={self.state}")


def _rows(xg, keep):
    """Reshape so every kept group is one row: (num_groups, values)."""
    nd = xg.dim()
    kept = sorted(d % nd for d in keep)
    other = reduce_dims(nd, keep)
    t = xg.permute(*kept, *other)
    n_rows = 1
    for d in kept:
        n_rows *= xg.shape[d]
    return t.reshape(n_rows, -1)


class QuantLinear(nn.Module):
    def __init__(self, lin, act, wbits):
        super().__init__()
        w = lin.weight.detach().clone()
        self.register_buffer("weight", fake_quant_weight(w, wbits) if wbits else w)
        self.register_buffer("bias", None if lin.bias is None else lin.bias.detach().clone())
        self.act = act

    def forward(self, x):
        return F.linear(self.act(x), self.weight, self.bias)


class QuantConv2d(nn.Module):
    def __init__(self, conv, act, wbits):
        super().__init__()
        assert conv.padding_mode == "zeros", "only zero padding is handled"
        w = conv.weight.detach().clone()
        self.register_buffer("weight", fake_quant_weight(w, wbits) if wbits else w)
        self.register_buffer("bias", None if conv.bias is None else conv.bias.detach().clone())
        self.stride, self.padding = conv.stride, conv.padding
        self.dilation, self.groups = conv.dilation, conv.groups
        self.act = act

    def forward(self, x):
        return F.conv2d(self.act(x), self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def _q(module, name):
    q = getattr(module, name, None)
    return q if q is not None else (lambda t: t)


def oraq_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """transformers' eager attention with quantizers before both matmuls.

    query/key/value: (B, heads, N, head_dim). Dropout is ignored (eval only).
    """
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    q = _q(module, "q_quant")(query)
    k = _q(module, "k_quant")(key)
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling          # matmul1
    if attention_mask is not None:
        scores = scores + attention_mask
    attn = F.softmax(scores, dim=-1)
    a = _q(module, "a_quant")(attn)
    v = _q(module, "v_quant")(value)
    out = torch.matmul(a, v).transpose(1, 2).contiguous()           # matmul2
    return out, attn


AttentionInterface.register("oraq", oraq_attention)


def op_of(name, module):
    """Map a layer name to the paper's operator label (None = leave in FP)."""
    if isinstance(module, nn.Linear):
        for suffix, op in ((".attention.q_proj", "qkv"), (".attention.k_proj", "qkv"),
                           (".attention.v_proj", "qkv"), (".attention.o_proj", "proj"),
                           (".mlp.fc1", "fc1"), (".mlp.fc2", "fc2")):
            if name.endswith(suffix):
                return op
    if isinstance(module, nn.Conv2d) and name.split(".")[0] in ("neck", "head"):
        return "conv"
    return None


def quantize_model(model, ops, act_bits=4, weight_bits=4):
    """Swap layers in place and attach attention quantizers.

    ops: {op_label: cfg}. Ops not listed stay in full precision. weight_bits=None
    keeps FP weights (used for diagnosis). The patch embedding conv and the
    ConvTranspose2d upsamplers in the neck are never quantized.
    """
    swaps = []
    for name, mod in model.named_modules():
        op = op_of(name, mod)
        if op is not None and op in ops:
            swaps.append((name, mod, op))
    for name, mod, op in swaps:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        if isinstance(mod, nn.Linear):
            new = QuantLinear(mod, ActQuantizer(op, ops[op], act_bits, axis=-1), weight_bits)
        else:
            new = QuantConv2d(mod, ActQuantizer(op, ops[op], act_bits, axis=1), weight_bits)
        setattr(parent, child, new)

    for mod in model.modules():
        if type(mod).__name__ == "Dinov2Attention":
            if "matmul1" in ops:
                mod.q_quant = ActQuantizer("matmul1", ops["matmul1"], act_bits, axis=-1)
                mod.k_quant = ActQuantizer("matmul1", ops["matmul1"], act_bits, axis=-1)
            if "matmul2" in ops:
                mod.a_quant = ActQuantizer("matmul2", ops["matmul2"], act_bits, axis=-1)   # source tokens
                mod.v_quant = ActQuantizer("matmul2", ops["matmul2"], act_bits, axis=-2)   # source tokens
    model.set_attn_implementation("oraq")
    return model


def quantizers(model, ops=None):
    return [m for m in model.modules()
            if isinstance(m, ActQuantizer) and (ops is None or m.op in ops)]


def set_state(model, state, ops=None):
    for q in quantizers(model):
        q.state = state if (ops is None or q.op in ops) else "off"


@torch.no_grad()
def calibrate(model, calib_inputs, record_ranges=False):
    """Observe all quantizers on the calibration inputs in one FP pass, then fix scales."""
    for q in quantizers(model):
        q._reset_stats()
        q.record_ranges = record_ranges
    set_state(model, "observe")
    for pv in calib_inputs:
        model(pixel_values=pv)
    for q in quantizers(model):
        q.finalize()
    set_state(model, "quant")


def count_by_op(model):
    counts = {}
    for q in quantizers(model):
        counts[q.op] = counts.get(q.op, 0) + 1
    return counts
