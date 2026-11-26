from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.cache_utils import Cache

from ...int4 import affine_int4, dequantize_affine, fake_quantize_int4, pack_nibbles, unpack_nibbles
from ..gptq import PackedWeight
from .llama import RotatedLlama


@dataclass(frozen=True)
class RuntimeConfig:
    activation_bits: int = 4
    kv_bits: int = 4


@dataclass(frozen=True)
class PackedTokens:
    codes: Tensor
    scale: Tensor
    zero: Tensor
    width: int

    @classmethod
    def from_tensor(cls, x: Tensor) -> "PackedTokens":
        codes, scale, zero = affine_int4(x)
        return cls(pack_nibbles(codes), scale, zero, x.shape[-1])

    def dequantize(self, dtype: torch.dtype) -> Tensor:
        codes = unpack_nibbles(self.codes, self.width)
        return dequantize_affine(codes, self.scale, self.zero).to(dtype)

    def append(self, new: "PackedTokens") -> "PackedTokens":
        if self.width != new.width:
            raise ValueError("cannot append KV states with a different head width")
        return PackedTokens(
            torch.cat((self.codes, new.codes), dim=-2),
            torch.cat((self.scale, new.scale), dim=-2),
            torch.cat((self.zero, new.zero), dim=-2),
            self.width,
        )

    def select_batch(self, indices: Tensor) -> "PackedTokens":
        ix = indices.to(self.codes.device)
        return PackedTokens(self.codes[ix], self.scale[ix], self.zero[ix], self.width)


class PackedKVCache(Cache):
    def __init__(self, *, retain: bool = True):
        super().__init__()
        self.retain = retain
        self.layers: dict[int, tuple[PackedTokens, PackedTokens]] = {}

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        pair = self.layers.get(0 if layer_idx is None else layer_idx)
        return 0 if pair is None else pair[0].codes.shape[-2]

    def get_max_cache_shape(self) -> None:
        return None

    @torch.no_grad()
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if not self.retain:
            return fake_quantize_int4(key_states), fake_quantize_int4(value_states)

        k, v = PackedTokens.from_tensor(key_states), PackedTokens.from_tensor(value_states)
        if layer_idx in self.layers:
            old_k, old_v = self.layers[layer_idx]
            k, v = old_k.append(k), old_v.append(v)
        result = k.dequantize(key_states.dtype), v.dequantize(value_states.dtype)
        self.layers[layer_idx] = k, v
        return result

    def reorder_cache(self, beam_idx: Tensor) -> None:
        self.layers = {
            i: (k.select_batch(beam_idx), v.select_batch(beam_idx))
            for i, (k, v) in self.layers.items()
        }

    def export_state(self) -> dict:
        return {
            i: tuple((p.codes, p.scale, p.zero, p.width) for p in pair)
            for i, pair in self.layers.items()
        }

    @classmethod
    def from_state(cls, state: Mapping) -> "PackedKVCache":
        cache = cls()
        cache.layers = {i: (PackedTokens(*k), PackedTokens(*v)) for i, (k, v) in state.items()}
        return cache


class PackedLinear(nn.Module):
    def __init__(self, weight: PackedWeight, bias: Tensor | None, *, quantize_input: bool):
        super().__init__()
        self.in_features = weight.in_features
        self.out_features = weight.codes.shape[0]
        self.quantize_input = quantize_input
        self.register_buffer("codes", weight.codes)
        self.register_buffer("scale", weight.scale)
        self.register_buffer("bias", None if bias is None else bias.detach())

    def forward(self, x: Tensor) -> Tensor:
        if self.quantize_input:
            x = fake_quantize_int4(x)
        w = (unpack_nibbles(self.codes, self.in_features).float() - 7) * self.scale[:, None]
        return F.linear(x, w.to(x.dtype), self.bias)


class QuantizedLlama(RotatedLlama):
    @property
    def config(self):
        return self.model.config

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                *, use_cache=None, **kwargs):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        cache = _select_cache(past_key_values, use_cache)
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=cache, use_cache=use_cache, **kwargs,
        )

    def generate(self, inputs=None, generation_config=None, **kwargs):
        cfg = generation_config if generation_config is not None else self.model.generation_config
        if kwargs.get("cache_implementation", cfg.cache_implementation) is not None:
            raise ValueError("COAT generation uses PackedKVCache")
        use_cache = kwargs.get("use_cache", cfg.use_cache)
        kwargs["past_key_values"] = _select_cache(kwargs.get("past_key_values"), use_cache)
        return super().generate(inputs=inputs, generation_config=generation_config, **kwargs)


def _select_cache(cache: PackedKVCache | None, use_cache: bool) -> PackedKVCache | None:
    if cache is not None and (not use_cache or not isinstance(cache, PackedKVCache) or not cache.retain):
        raise ValueError("past_key_values must be a retaining PackedKVCache with use_cache=True")
    if cache is None and use_cache:
        return PackedKVCache()
    return cache


def install_quantized_runtime(
    rotated: RotatedLlama,
    weights: Mapping[int, Mapping[str, PackedWeight]],
    config: RuntimeConfig = RuntimeConfig(),
) -> QuantizedLlama:
    if (config.activation_bits, config.kv_bits) != (4, 4):
        raise ValueError("the COAT reference runtime supports A4KV4")
    adapter = rotated.adapter
    if set(weights) != set(range(len(adapter.layers))):
        raise ValueError("expected packed weights for every decoder layer")

    current_kv = PackedKVCache(retain=False)

    def select_attention_cache(_, args, kwargs):
        cache = kwargs.get("past_key_value")
        if cache is None:
            kwargs["past_key_value"] = current_kv
        elif not isinstance(cache, PackedKVCache):
            raise ValueError("quantized attention requires PackedKVCache")
        return args, kwargs

    def quantize_norm(_, args, out):
        return fake_quantize_int4(out)

    for i in range(len(adapter.layers)):
        projections = adapter.projections(i)
        if set(weights[i]) != set(projections):
            raise ValueError(f"expected packed weights for all projections in layer {i}")
        for name, proj in projections.items():
            w = weights[i][name]
            if (not isinstance(proj, nn.Linear) or w.in_features != proj.in_features
                    or w.codes.shape != (proj.out_features, (proj.in_features + 1) // 2)
                    or w.codes.dtype != torch.uint8 or w.scale.shape != (proj.out_features,)
                    or w.scale.dtype != torch.float32):
                raise ValueError(f"invalid packed weight for layer {i} {name}")

    for i, layer in enumerate(adapter.layers):
        for name, proj in adapter.projections(i).items():
            owner = layer.self_attn if name in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp
            packed = PackedLinear(weights[i][name], proj.bias, quantize_input=name in ("o_proj", "down_proj"))
            setattr(owner, name, packed.to(device=proj.weight.device))
        for norm in adapter.norms(i):
            norm.register_forward_hook(quantize_norm)
        layer.self_attn.register_forward_pre_hook(select_attention_cache, with_kwargs=True)
    return QuantizedLlama(adapter, rotated.rotations).eval()
