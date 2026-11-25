"""Batch-one Qwen3 adapter for transformers==4.57.1, inference only.

Per-head retained indices differ, but every layer/head has equal physical length.
Old keys remain in chronological order; physical cache positions build the causal
mask while explicit logical position_ids preserve RoPE coordinates.
"""

import time
import torch
import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .phases import PhaseParser
from .policy import AgentKVConfig, QueryBuffer, gather_cache, policy_indices


class Qwen3Session:
    def __init__(self, model, mode="agentkv", config=None):
        if transformers.__version__ != "4.57.1":
            raise ValueError("this adapter requires transformers==4.57.1")
        if mode not in ("full", "recent", "agentkv"):
            raise ValueError("unknown cache policy")
        if model.config.model_type != "qwen3" or model.training:
            raise ValueError("requires a Qwen3 model in eval mode")
        if getattr(model.config, "use_sliding_window", False) or "sliding_attention" in model.config.layer_types:
            raise ValueError("sliding-window layers are unsupported")
        if model.config.rope_scaling:
            raise ValueError("only default RoPE is supported")
        if getattr(model, "_agentkv_session_active", False):
            raise ValueError("only one session may attach to a model at a time")
        self.model, self.mode = model, mode
        self.config = config or AgentKVConfig()
        self.device = next(model.parameters()).device
        self.handles = []
        self.closed = False
        self.reset()
        model._agentkv_session_active = True
        if mode != "full":
            for i, layer in enumerate(model.model.layers):
                self.handles.append(layer.self_attn.q_norm.register_forward_hook(self._capture(i)))
                self.handles.append(layer.self_attn.register_forward_hook(self._observe(i), with_kwargs=True))

    def reset(self):
        self.cache = DynamicCache()
        self.parser = PhaseParser()
        self.logical_length = 0
        self.decode_tokens = 0
        self.compressions = 0
        self.eviction_seconds = 0.0
        self.positions = {}
        self.raw_queries = {}
        self.phases = []
        self.buffers = {} if self.mode == "full" else {
            i: QueryBuffer(self.config.query_budget, self.mode)
            for i in range(len(self.model.model.layers))
        }

    def _capture(self, index):
        def hook(module, args, output):
            self.raw_queries[index] = output.detach()
        return hook

    def _observe(self, index):
        def hook(module, args, kwargs, output):
            q = self.raw_queries.pop(index).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            rotated, _ = apply_rotary_pos_emb(q, q, cos, sin)
            self.buffers[index].update(rotated[0], self.phases)
        return hook

    @torch.inference_mode()
    def append(self, token_ids, fragments, *, decode=False):
        """Append new tokens only; never re-prefill an already-cached transcript.

        For multi-token input pass decoded fragments aligned to input IDs. Decode
        calls accept one token, allowing exact interval scheduling. Returns final
        next-token logits. This does not automatically compress prefill.
        """
        if self.closed:
            raise RuntimeError("session is closed")
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        if ids.ndim != 1 or ids.numel() == 0 or ids.numel() != len(fragments):
            raise ValueError("expected nonempty one-dimensional IDs and matching fragments")
        if decode and ids.numel() != 1:
            raise ValueError("decode calls must contain exactly one token")
        if self.logical_length + len(ids) > self.model.config.max_position_embeddings:
            raise ValueError("logical sequence exceeds configured context limit")
        self.phases = [self.parser.feed(text) for text in fragments]
        logical = torch.arange(self.logical_length, self.logical_length + len(ids), device=self.device)
        physical = torch.arange(self.cache.get_seq_length(), self.cache.get_seq_length() + len(ids), device=self.device)
        result = self.model(
            input_ids=ids[None], position_ids=logical[None], cache_position=physical,
            past_key_values=self.cache, use_cache=True, logits_to_keep=1,
        )
        self.cache = result.past_key_values
        for i, layer in enumerate(self.cache.layers):
            new_positions = logical.expand(layer.keys.shape[1], -1)
            old = self.positions.get(i)
            self.positions[i] = new_positions.clone() if old is None else torch.cat((old, new_positions), dim=-1)
        self.logical_length += len(ids)
        if decode:
            self.decode_tokens += 1
            if self.decode_tokens % self.config.interval == 0:
                self.compress()
        return result.logits[0, -1]

    @torch.inference_mode()
    def compress(self):
        if self.mode == "full" or self.cache.get_seq_length() <= self.config.budget:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for i, layer in enumerate(self.cache.layers):
            indices = policy_indices(self.buffers[i].representatives(), layer.keys[0], self.positions[i], self.config)
            layer.keys = gather_cache(layer.keys, indices)
            layer.values = gather_cache(layer.values, indices)
            self.positions[i] = self.positions[i].gather(1, indices)
        self.compressions += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.eviction_seconds += time.perf_counter() - started

    def report(self):
        return {
            "logical_tokens": self.logical_length,
            "cached_tokens": self.cache.get_seq_length(),
            "kv_bytes": sum(t.numel() * t.element_size() for layer in self.cache.layers for t in (layer.keys, layer.values)),
            "query_buffer_bytes": sum(b.nbytes for b in self.buffers.values()),
            "position_bytes": sum(p.numel() * p.element_size() for p in self.positions.values()),
            "representatives_per_layer": {str(i): sum(len(b) for b in q.buffers.values()) for i, q in self.buffers.items()},
            "representatives_by_phase": {str(i): {p: len(b) for p, b in q.buffers.items()} for i, q in self.buffers.items()},
            "compression_events": self.compressions,
            "eviction_seconds": self.eviction_seconds,
        }

    def close(self):
        if self.closed:
            return
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.raw_queries.clear()
        self.model._agentkv_session_active = False
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
