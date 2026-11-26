"""Tensor reference for AgentKV section 3.3 (independently implemented).

Queries: [query_heads, representatives, head_dim].
Keys: [kv_heads, cached_tokens, head_dim]. Scores/indices: [kv_heads, tokens].
Scoring deliberately has no historical causal mask: representatives estimate
future-query utility over all currently available keys. Generation itself is causal.
"""

from collections import deque
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .phases import PHASES


@dataclass(frozen=True)
class AgentKVConfig:
    budget: int = 512
    query_budget: int = 32
    interval: int = 128
    sink_tokens: int = 32
    recent_tokens: int = 32
    pool_kernel: int = 1
    scoring_profile: str = "reference_v1"

    def __post_init__(self):
        if self.scoring_profile not in ("reference_v1", "upstream_mean"):
            raise ValueError("unknown scoring_profile")
        for name in ("budget", "query_budget", "interval", "pool_kernel"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("sink_tokens", "recent_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.query_budget % 4:
            raise ValueError("query_budget must be divisible by four")
        if self.pool_kernel % 2 == 0:
            raise ValueError("pool_kernel must be odd")
        if self.sink_tokens + self.recent_tokens > self.budget:
            raise ValueError("protected entries must fit within budget")


class QueryBuffer:
    """One buffer per layer/request; clones prevent retaining large prefill storage."""

    def __init__(self, capacity: int, mode: str = "agentkv"):
        if mode not in ("agentkv", "recent"):
            raise ValueError("mode must be agentkv or recent")
        if type(capacity) is not int or capacity <= 0 or capacity % 4:
            raise ValueError("capacity must be positive and divisible by four")
        self.mode = mode
        self.buffers = {p: deque(maxlen=capacity // 4) for p in PHASES} if mode == "agentkv" else {"recent": deque(maxlen=capacity)}

    def update(self, queries: torch.Tensor, phases: list[str]):
        if queries.ndim != 3 or queries.shape[1] != len(phases):
            raise ValueError("expected queries [heads, tokens, dim] and one phase per token")
        if any(p not in PHASES for p in phases):
            raise ValueError("unknown phase")
        # Only copy the last capacity entries per phase from a prefill chunk.
        for name, buffer in self.buffers.items():
            indices = [i for i, phase in enumerate(phases) if self.mode == "recent" or phase == name]
            for i in indices[-buffer.maxlen:]:
                buffer.append(queries[:, i, :].detach().clone())

    def representatives(self):
        entries = [q for buffer in self.buffers.values() for q in buffer]
        if not entries:
            raise ValueError("cannot score with an empty query buffer")
        return torch.stack(entries, dim=1)

    @property
    def nbytes(self):
        return sum(q.numel() * q.element_size() for b in self.buffers.values() for q in b)


def attention_scores(queries: torch.Tensor, keys: torch.Tensor, pool_kernel: int = 1, group_reduction: str = "post_softmax_max"):
    if queries.ndim != 3 or keys.ndim != 3:
        raise ValueError("queries and keys must be rank three")
    hq, nq, dim = queries.shape
    hk, nk, kd = keys.shape
    if min(hq, hk, nq, nk, dim) <= 0 or dim != kd or hq % hk:
        raise ValueError("incompatible or empty query/key shapes")
    if queries.device != keys.device or not queries.is_floating_point() or not keys.is_floating_point():
        raise ValueError("queries and keys must be floating tensors on the same device")
    if type(pool_kernel) is not int or pool_kernel <= 0 or pool_kernel % 2 == 0:
        raise ValueError("pool_kernel must be positive and odd")
    # Grouped query heads share each KV head; reduce by max after averaging reps.
    grouped = queries.float().reshape(hk, hq // hk, nq, dim)
    logits = torch.einsum("hgrd,hsd->hgrs", grouped, keys.float()) / math.sqrt(dim)
    if group_reduction == "post_softmax_max":
        scores = logits.softmax(dim=-1).mean(dim=-2).amax(dim=1)
    elif group_reduction == "mean":
        scores = logits.softmax(dim=-1).mean(dim=(1, 2))
    else:
        raise ValueError("unknown group_reduction")
    if pool_kernel > 1:
        scores = F.max_pool1d(scores.unsqueeze(1), pool_kernel, stride=1, padding=pool_kernel // 2).squeeze(1)
    return scores


def policy_indices(queries, keys, positions, config):
    """Versioned scoring; upstream_mean excludes protected slots before softmax.

    Implements the released AgentKV preset's SnapKV mean-GQA scoring. The
    reference_v1 branch preserves the first two pilots exactly. Hyperparameters
    remain explicit: the released preset uses sink=16/recent=8/pool=1/q=32.
    """
    if config.scoring_profile == "reference_v1":
        return select_indices(attention_scores(queries, keys, config.pool_kernel), positions, config)
    heads, count, _ = keys.shape
    if count <= config.budget:
        return torch.arange(count, device=keys.device).expand(heads, -1)
    protected = positions < config.sink_tokens
    if config.recent_tokens:
        protected[:, -config.recent_tokens:] = True
    counts = protected.sum(-1)
    if not (counts == counts[0]).all():
        raise ValueError("upstream_mean requires equal protected counts across KV heads")
    slots = torch.arange(count, device=keys.device).expand(heads, -1)
    candidates = slots[~protected].reshape(heads, -1)
    candidate_keys = keys.gather(1, candidates[..., None].expand(-1, -1, keys.shape[-1]))
    scores = attention_scores(queries, candidate_keys, config.pool_kernel, group_reduction="mean")
    # Positive attention scores for candidates, protected slots promoted by selector.
    full_scores = torch.zeros((heads, count), device=keys.device, dtype=scores.dtype)
    full_scores.scatter_(1, candidates, scores)
    return select_indices(full_scores, positions, config)


def select_indices(scores: torch.Tensor, positions: torch.Tensor, config: AgentKVConfig):
    """Return chronological indices per KV head; ties prefer earlier entries."""
    if scores.ndim != 2 or positions.shape != scores.shape or positions.dtype != torch.long:
        raise ValueError("scores and int64 positions must have shape [kv_heads, tokens]")
    if positions.device != scores.device or not torch.isfinite(scores).all():
        raise ValueError("scores must be finite and share the positions device")
    if (positions < 0).any() or (positions[:, 1:] <= positions[:, :-1]).any():
        raise ValueError("positions must be nonnegative and strictly increasing")
    heads, count = scores.shape
    if count <= config.budget:
        return torch.arange(count, device=scores.device).expand(heads, -1)
    protected = positions < config.sink_tokens
    if config.recent_tokens:
        protected[:, -config.recent_tokens:] = True
    ranked = scores.clone()
    ranked[protected] = torch.inf
    chosen = torch.argsort(ranked, dim=-1, descending=True, stable=True)[:, :config.budget]
    return chosen.sort(dim=-1).values


def gather_cache(tensor: torch.Tensor, indices: torch.Tensor):
    """Physical gather, preserving dtype, with independent keep sets per KV head."""
    if tensor.ndim != 4 or tensor.shape[0] != 1 or indices.shape[0] != tensor.shape[1]:
        raise ValueError("expected batch-one cache [1, kv_heads, tokens, dim]")
    return tensor.gather(2, indices[None, :, :, None].expand(1, -1, -1, tensor.shape[-1])).contiguous()
