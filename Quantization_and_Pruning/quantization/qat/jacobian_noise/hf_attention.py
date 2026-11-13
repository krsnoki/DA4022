"""Hugging Face Transformers integration.

Registers an attention implementation called ``"jacobian_noise"`` through
``transformers.AttentionInterface``.  With noise disabled (always the case in
``eval()`` mode) it is numerically the same as the library's ``eager``
attention, so the same implementation is used for training *and* for quantized
evaluation (where it can additionally fake-quantize Q/K, see ``fake_quant.py``).

Usage::

    ctrl = attach_jacobian_noise(model, JacobianNoiseConfig(mode="jacobian"))
    for batch in loader:
        loss = model(**batch).loss; loss.backward()
        optimizer.step(); ctrl.step()
    detach_jacobian_noise(model)          # restore original attention impl before save_pretrained

Noise placement (Alg. 1 of the paper): ``Z = Q K^T / sqrt(d)``; ``Z~ = Z + eps``;
``A = softmax(Z~ + mask)``.  The mask is additive, so adding the noise before or
after it is mathematically identical; we add it before so that ``finfo.min``
mask entries can never overflow to ``-inf`` in half precision.

Tested with transformers 4.57 on Qwen2/Llama (causal, GQA) and SigLIP
(bidirectional, vision and text towers).  Any model whose attention modules have
``q_proj``/``k_proj`` and dispatch through ``ALL_ATTENTION_FUNCTIONS`` should work.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch import nn

from .controller import JacobianNoiseConfig, JacobianNoiseController

ATTN_IMPL_NAME = "jacobian_noise"
_REGISTERED = False


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def jacobian_noise_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Eager attention + Jacobian-guided logit noise (+ optional Q/K fake-quant)."""
    n_rep = getattr(module, "num_key_value_groups", 1)
    key = _repeat_kv(key, n_rep)
    value = _repeat_kv(value, n_rep)

    if getattr(module, "_fq_qk_bits", None):
        from ...activation.fake_quant import fake_quant_qk

        query, key = fake_quant_qk(module, query, key)

    scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
    mask = None
    if attention_mask is not None:
        mask = attention_mask[:, :, :, : key.shape[-2]]

    ctrl: Optional[JacobianNoiseController] = getattr(module, "_jn_controller", None)
    layer = getattr(module, "_jn_layer_idx", None)
    if ctrl is not None and layer is not None:
        if ctrl.collecting and (module.training or ctrl.measuring):
            with torch.no_grad():
                clean = scores if mask is None else scores + mask
                ctrl.accumulate(layer, torch.softmax(clean.float(), dim=-1))
        if module.training and ctrl.injects_noise:
            sigma = ctrl.sigma_for(layer, scores.shape[-2], scores.device)
            if sigma is not None:
                noise = ctrl.sample_noise(scores) * sigma.view(1, 1, -1, 1).to(scores.dtype)
                scores = scores + noise

    if mask is not None:
        scores = scores + mask
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probs = nn.functional.dropout(probs, p=dropout, training=module.training)
    out = torch.matmul(probs, value).transpose(1, 2).contiguous()
    return out, probs


def register_attention() -> None:
    """Register ``jacobian_noise`` attention + its (eager-style) mask builder. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return
    from transformers import AttentionInterface

    AttentionInterface.register(ATTN_IMPL_NAME, jacobian_noise_attention_forward)
    try:  # transformers >= 4.53 builds masks per attention implementation
        from transformers.masking_utils import AttentionMaskInterface, eager_mask

        AttentionMaskInterface.register(ATTN_IMPL_NAME, eager_mask)
    except ImportError:  # pragma: no cover - older versions build eager masks by default
        pass
    _REGISTERED = True


def find_attention_modules(model: nn.Module, include: Optional[str] = None) -> Dict[str, nn.Module]:
    """Attention modules (have q_proj and k_proj), optionally filtered by a name substring/regex."""
    import re

    pat = re.compile(include) if include else None
    out = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "q_proj") and hasattr(mod, "k_proj") and hasattr(mod, "config"):
            if pat is None or pat.search(name):
                out[name] = mod
    return out


def _all_configs(model: nn.Module):
    seen = {}
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation") and id(cfg) not in seen:
            seen[id(cfg)] = cfg
    root = getattr(model, "config", None)
    for key in getattr(root, "sub_configs", {}) or {}:
        sub = getattr(root, key, None)
        if sub is not None and id(sub) not in seen:
            seen[id(sub)] = sub
    return list(seen.values())


def set_attn_implementation(model: nn.Module, name: str) -> Dict[int, str]:
    """Set the attention implementation on every config in the model; return previous values."""
    previous = {}
    for cfg in _all_configs(model):
        previous[id(cfg)] = cfg._attn_implementation
        cfg._attn_implementation_internal = name
    return previous


def attach_jacobian_noise(
    model: nn.Module,
    config: Optional[JacobianNoiseConfig] = None,
    include: Optional[str] = None,
) -> JacobianNoiseController:
    """Switch ``model`` to the ``jacobian_noise`` attention and attach a controller.

    Args:
        model: a Hugging Face ``PreTrainedModel`` (or any module containing HF attention).
        config: noise schedule; ``mode="none"`` gives the identical code path with no noise.
        include: optional regex on module names restricting which attention
            layers get noise (e.g. ``"vision_model"`` for SigLIP's image tower).
    """
    register_attention()
    modules = find_attention_modules(model, include)
    if not modules:
        raise ValueError("no attention modules with q_proj/k_proj found")
    ctrl = JacobianNoiseController(config or JacobianNoiseConfig(), num_layers=len(modules))
    for idx, (name, mod) in enumerate(modules.items()):
        mod._jn_controller = ctrl
        mod._jn_layer_idx = idx
    ctrl.layer_names = list(modules)
    ctrl._previous_attn = set_attn_implementation(model, ATTN_IMPL_NAME)
    return ctrl


def use_jacobian_noise_attention(model: nn.Module) -> None:
    """Use the ``jacobian_noise`` attention without a controller (e.g. for Q/K fake-quant at eval)."""
    register_attention()
    set_attn_implementation(model, ATTN_IMPL_NAME)


def detach_jacobian_noise(model: nn.Module, attn_implementation: Optional[str] = None) -> None:
    """Remove controllers and restore the previous (or given) attention implementation."""
    prev: Dict[int, str] = {}
    for mod in model.modules():
        ctrl = getattr(mod, "_jn_controller", None)
        if ctrl is not None:
            prev = getattr(ctrl, "_previous_attn", prev)
            del mod._jn_controller
            del mod._jn_layer_idx
    for cfg in _all_configs(model):
        cfg._attn_implementation_internal = attn_implementation or prev.get(id(cfg), "eager")
