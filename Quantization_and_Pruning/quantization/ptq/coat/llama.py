from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from threading import RLock

import torch
from torch import Tensor, nn

from Backbones.Generative_Pretrained.llama import LlamaAdapter

from .rotations import RotationDimensions, RotationPlan


class RotatedLlama(nn.Module):
    def __init__(self, adapter: LlamaAdapter, rotations: RotationPlan):
        super().__init__()
        self.adapter = adapter
        self.model = adapter.model
        self.rotations = rotations
        self._rotation_lock = RLock()
        self._rotations_active = False

    @contextmanager
    def online_rotations(self) -> Iterator[None]:
        with self._rotation_lock:
            if self._rotations_active:
                yield
                return
            with ExitStack() as hooks:
                for i, (r3, r4) in enumerate(zip(self.rotations.r3, self.rotations.r4, strict=True)):
                    def rotate_qk(q: Tensor, k: Tensor, apply=r3) -> tuple[Tensor, Tensor]:
                        return apply(q), apply(k)

                    hooks.enter_context(self.adapter.post_rope(i, rotate_qk))
                    hooks.enter_context(self.adapter.gated_mlp_output(i, r4))
                self._rotations_active = True
                try:
                    yield
                finally:
                    self._rotations_active = False

    def forward(self, *args, **kwargs):
        with self.online_rotations():
            return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        with self.online_rotations():
            return self.model.generate(*args, **kwargs)


def _validate_layout(adapter: LlamaAdapter, rotations: RotationPlan) -> RotationDimensions:
    cfg = adapter.model.config
    dims = RotationDimensions.from_config(cfg)
    d, p, f, n = dims.hidden_size, dims.head_dim, dims.intermediate_size, dims.num_layers
    if cfg._attn_implementation != "eager":
        raise ValueError("COAT requires eager Llama attention")
    if len(adapter.layers) != n or rotations.r1.shape != (d, d) or rotations.r2.shape != (n, p, p):
        raise ValueError("R1 or R2 does not match the Llama layout")
    if [r.hadamard.width for r in rotations.r3] != [p] * n or [r.hadamard.width for r in rotations.r4] != [f] * n:
        raise ValueError("R3 or R4 does not match the Llama layout")

    hq, hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    if hq % hkv:
        raise ValueError("query heads must divide into equal KV groups")
    shapes = {
        "q_proj": (hq * p, d), "k_proj": (hkv * p, d), "v_proj": (hkv * p, d),
        "o_proj": (d, hq * p), "gate_proj": (f, d), "up_proj": (f, d), "down_proj": (d, f),
    }
    if adapter.embeddings.weight.shape != adapter.lm_head.weight.shape or adapter.lm_head.weight.shape[1] != d:
        raise ValueError("embedding or LM head does not match the Llama layout")
    for i in range(n):
        for name, proj in adapter.projections(i).items():
            if not isinstance(proj, nn.Linear) or proj.weight.shape != shapes[name]:
                raise ValueError(f"layer {i} {name} does not match the floating-point Llama layout")
    return dims


def _fuse_layer(
    adapter: LlamaAdapter, i: int, r1: Tensor, r2: Tensor, r4: nn.Module, dims: RotationDimensions,
) -> None:
    d, p = dims.hidden_size, dims.head_dim
    cfg = adapter.model.config
    hq, hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    attn_norm, ffn_norm = adapter.norms(i)
    attn_gain, ffn_gain = attn_norm.weight.to(r1), ffn_norm.weight.to(r1)
    proj = adapter.projections(i)

    for name, gain in (("q_proj", attn_gain), ("k_proj", attn_gain), ("gate_proj", ffn_gain), ("up_proj", ffn_gain)):
        w = proj[name].weight
        w.copy_((w.to(r1) * gain) @ r1)

    v_proj, o_proj, down = proj["v_proj"], proj["o_proj"], proj["down_proj"]
    # R2 is shared across KV heads. Every query head uses its KV group's same basis.
    w = (v_proj.weight.to(r1) * attn_gain) @ r1
    v_proj.weight.copy_((r2.T @ w.reshape(hkv, p, d)).reshape_as(v_proj.weight))
    if v_proj.bias is not None:
        v_proj.bias.copy_((v_proj.bias.to(r1).reshape(hkv, p) @ r2).flatten())

    w = r1.T @ o_proj.weight.to(r1)
    o_proj.weight.copy_((w.reshape(d, hq, p) @ r2).reshape_as(o_proj.weight))
    if o_proj.bias is not None:
        o_proj.bias.copy_(o_proj.bias.to(r1) @ r1)

    w = r1.T @ down.weight.to(r1)
    for dst, rows in zip(down.weight.split(256), w.split(256), strict=True):
        dst.copy_(r4(rows))
    if down.bias is not None:
        down.bias.copy_(down.bias.to(r1) @ r1)

    attn_norm.weight.fill_(1)
    ffn_norm.weight.fill_(1)


@torch.no_grad()
def fuse_llama(adapter: LlamaAdapter, rotations: RotationPlan) -> RotatedLlama:
    """Fuse into an independently loaded Llama instance in place."""
    model = adapter.model
    if getattr(model, "_coat_fused", False):
        raise ValueError("COAT fusion requires a fresh adaptation instance")
    dims = _validate_layout(adapter, rotations)
    emb = adapter.embeddings.weight
    head = adapter.lm_head
    rotations = rotations.to(device=emb.device)
    dtype = torch.float64 if emb.dtype == torch.float64 else torch.float32
    r1, r2 = rotations.r1.to(dtype), rotations.r2.to(dtype)
    gain = adapter.final_norm.weight.to(r1)
    tied = emb.data_ptr() == head.weight.data_ptr()
    model._coat_fused = True

    if tied and not torch.all(gain == 1):
        head.weight = nn.Parameter(head.weight.detach().clone(), requires_grad=head.weight.requires_grad)
        model.config.tie_word_embeddings = False
        tied = False

    emb.copy_(emb.to(r1) @ r1)
    if not tied:
        head.weight.copy_((head.weight.to(r1) * gain) @ r1)
    adapter.final_norm.weight.fill_(1)

    for i in range(dims.num_layers):
        _fuse_layer(adapter, i, r1, r2[i], rotations.r4[i], dims)
    return RotatedLlama(adapter, rotations).eval()
