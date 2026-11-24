"""Noise schedule for Jacobian-guided noise injection.

The controller owns all state of the method and is model-agnostic.  The
attention function in ``hf_attention.py`` asks it two things on every forward
pass of an attention layer:

* ``sigma_for(layer, q_len)``: the per-query-position noise std for that layer;
* ``accumulate(layer, probs)``: (only on collection steps) the clean softmax
  probabilities, from which ``||J_softmax||_F^2`` statistics are gathered.

Paper recipe (Pandey et al., Sec. 2.4, Alg. 1, App. B.2), which is the default:

* ``sigma_i^(l) = alpha * sqrt( E_{batch,heads} ||J||_F^2 at query row i, layer l )``
* clamp to ``[sigma_min, sigma_max] = [0.005, 10.0]``, ``alpha = 0.5``
* constant ``sigma = 0.1`` for the first ``warmup_steps = 100`` optimiser steps
* sigma refreshed every ``update_interval = 100`` optimiser steps
* noise only while training; ``sigma = 0`` at inference.

Modes (``JacobianNoiseConfig.mode``):

``none``             no noise (the fine-tuning baseline "B").
``jacobian``         paper method: per-layer, per-query-position sigma ("B+N").
``jacobian_layer``   one sigma per layer (ablation; the paper claims row-wise is better
                     but reports no numbers).
``uniform_matched``  one global sigma whose RMS equals the RMS of the sigmas the
                     ``jacobian`` mode would use at the same moment.  Same total noise
                     energy, no Jacobian shaping -> isolates the effect of the Jacobian.
``constant``         fixed ``constant_sigma`` everywhere (sanity ablation).

Implementation choices where the paper is silent (documented in the README):

* sigma statistics come from the *clean* (noise-free, masked) softmax and are
  computed under ``torch.no_grad`` -- sigma is a schedule, not a learned quantity;
* statistics are gathered on the micro-batches of the last optimiser step before
  each update boundary (``collect="last_step"``) or over the whole interval
  (``collect="window"``);
* padded query rows are excluded if the trainer provides ``set_query_mask``;
* positions never observed fall back to the mean sigma of the same layer.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional

import torch

from .jacobian import softmax_jacobian_fro_sq

VALID_MODES = ("none", "jacobian", "jacobian_layer", "uniform_matched", "constant")
JACOBIAN_MODES = ("jacobian", "jacobian_layer", "uniform_matched")


@dataclass
class JacobianNoiseConfig:
    mode: str = "jacobian"
    alpha: float = 0.5
    sigma_min: float = 0.005
    sigma_max: float = 10.0
    warmup_steps: int = 100
    warmup_sigma: float = 0.1
    update_interval: int = 100
    constant_sigma: float = 0.1
    collect: str = "last_step"  # "last_step" | "window"
    ema: float = 0.0  # 0 -> replace sigma with the newest estimate
    max_positions: int = 8192
    seed: Optional[int] = None  # optional dedicated RNG seed for the noise

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.collect not in ("last_step", "window"):
            raise ValueError("collect must be 'last_step' or 'window'")
        if self.alpha < 0 or self.sigma_min < 0 or self.sigma_max < self.sigma_min:
            raise ValueError("need alpha >= 0 and 0 <= sigma_min <= sigma_max")
        if self.update_interval < 1 or self.warmup_steps < 0:
            raise ValueError("need update_interval >= 1 and warmup_steps >= 0")
        if not 0.0 <= self.ema < 1.0:
            raise ValueError("ema must be in [0, 1)")

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "JacobianNoiseConfig":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown JacobianNoiseConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _LayerStats:
    sum_jsq: torch.Tensor  # [P] float32
    count: torch.Tensor  # [P] float32
    sigma: torch.Tensor  # [P] float32 (current schedule)
    sigma_set: bool = False
    max_seen: int = 0  # longest query length observed so far


class JacobianNoiseController:
    """Holds the sigma schedule for every attention layer of one model."""

    def __init__(self, config: Optional[JacobianNoiseConfig] = None, num_layers: int = 0):
        self.config = config or JacobianNoiseConfig()
        self.num_layers = num_layers
        self.global_step = 0
        self.enabled = True
        self.measuring = False  # collect statistics also in eval mode (diagnostics)
        self._layers: Dict[int, _LayerStats] = {}
        self._query_mask: Optional[torch.Tensor] = None
        self._generator: Optional[torch.Generator] = None
        self._global_sigma: Optional[float] = None  # for jacobian_layer/uniform_matched bookkeeping
        self.history: List[dict] = []

    # ------------------------------------------------------------------ state
    @property
    def injects_noise(self) -> bool:
        return self.enabled and self.config.mode != "none"

    @property
    def in_warmup(self) -> bool:
        return self.global_step < self.config.warmup_steps

    @property
    def collecting(self) -> bool:
        """True if the current optimiser step must gather Jacobian statistics."""
        if self.measuring:
            return True
        cfg = self.config
        if cfg.mode not in JACOBIAN_MODES or not self.enabled:
            return False
        if cfg.collect == "window":
            # gather during the whole interval that ends at the next boundary,
            # including the last warm-up interval.
            return self.global_step + 1 + cfg.update_interval > cfg.warmup_steps
        nxt = self.global_step + 1
        return nxt >= cfg.warmup_steps and (nxt - cfg.warmup_steps) % cfg.update_interval == 0

    def set_query_mask(self, mask: Optional[torch.Tensor]) -> None:
        """Optional ``[batch, q_len]`` mask (1 = real token) used to ignore padded rows."""
        self._query_mask = None if mask is None else mask.detach()

    @contextlib.contextmanager
    def disabled(self) -> Iterator[None]:
        prev = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = prev

    @contextlib.contextmanager
    def measure(self) -> Iterator[None]:
        """Collect ||J||^2 statistics without injecting noise (e.g. on an eval set)."""
        prev_m, prev_e = self.measuring, self.enabled
        self.measuring, self.enabled = True, False
        self.reset_statistics()
        try:
            yield
        finally:
            self.measuring, self.enabled = prev_m, prev_e

    # ----------------------------------------------------------------- layers
    def _layer(self, layer: int, device: torch.device) -> _LayerStats:
        st = self._layers.get(layer)
        if st is None or st.sigma.device != device:
            P = self.config.max_positions
            if st is None:
                st = _LayerStats(
                    sum_jsq=torch.zeros(P, device=device),
                    count=torch.zeros(P, device=device),
                    sigma=torch.full((P,), float(self.config.warmup_sigma), device=device),
                )
            else:
                st = _LayerStats(st.sum_jsq.to(device), st.count.to(device), st.sigma.to(device), st.sigma_set, st.max_seen)
            self._layers[layer] = st
        return st

    # ----------------------------------------------------------- noise sigma
    def sigma_for(self, layer: int, q_len: int, device: torch.device) -> Optional[torch.Tensor]:
        """Noise std per query position ``[q_len]`` (float32), or None for no noise."""
        if not self.injects_noise:
            return None
        cfg = self.config
        if q_len > cfg.max_positions:
            raise ValueError(f"sequence length {q_len} exceeds max_positions={cfg.max_positions}")
        if cfg.mode == "constant":
            return torch.full((q_len,), float(cfg.constant_sigma), device=device)
        st = self._layer(layer, device)
        if self.in_warmup or not st.sigma_set:
            return torch.full((q_len,), float(cfg.warmup_sigma), device=device)
        return st.sigma[:q_len]

    def sample_noise(self, like: torch.Tensor) -> torch.Tensor:
        if self.config.seed is None:
            return torch.randn_like(like)
        if self._generator is None or self._generator.device != like.device:
            self._generator = torch.Generator(device=like.device)
            self._generator.manual_seed(int(self.config.seed))
        return torch.randn(like.shape, generator=self._generator, device=like.device, dtype=like.dtype)

    # ------------------------------------------------------------ statistics
    @torch.no_grad()
    def accumulate(self, layer: int, probs: torch.Tensor) -> None:
        """Add ``||J||_F^2`` of clean softmax rows ``probs [B, H, Q, K]`` to the stats."""
        B, H, Q, _ = probs.shape
        st = self._layer(layer, probs.device)
        if Q > self.config.max_positions:
            raise ValueError(f"sequence length {Q} exceeds max_positions={self.config.max_positions}")
        st.max_seen = max(st.max_seen, Q)
        jsq = softmax_jacobian_fro_sq(probs).sum(dim=1)  # [B, Q], summed over heads
        m = self._query_mask
        if m is not None and m.shape == (B, Q):
            m = m.to(device=jsq.device, dtype=jsq.dtype)
            st.sum_jsq[:Q] += (jsq * m).sum(0)
            st.count[:Q] += m.sum(0) * H
        else:
            st.sum_jsq[:Q] += jsq.sum(0)
            st.count[:Q] += float(B * H)

    def reset_statistics(self) -> None:
        for st in self._layers.values():
            st.sum_jsq.zero_()
            st.count.zero_()

    @torch.no_grad()
    def mean_jacobian_sq(self) -> Dict[int, torch.Tensor]:
        """Per layer: mean ``||J||_F^2`` per position (NaN where unobserved)."""
        out = {}
        for idx, st in sorted(self._layers.items()):
            mean = st.sum_jsq / st.count.clamp_min(1.0)
            out[idx] = torch.where(st.count > 0, mean, torch.full_like(mean, float("nan")))
        return out

    @torch.no_grad()
    def _update_sigma(self) -> None:
        cfg = self.config
        observed = {i: st for i, st in self._layers.items() if bool((st.count > 0).any())}
        if not observed:
            return
        raw: Dict[int, torch.Tensor] = {}
        for i, st in observed.items():
            seen = st.count > 0
            mean = st.sum_jsq / st.count.clamp_min(1.0)
            if cfg.mode == "jacobian_layer":
                layer_mean = st.sum_jsq[seen].sum() / st.count[seen].sum()
                s = torch.full_like(mean, cfg.alpha * math.sqrt(float(layer_mean)))
            else:
                s = cfg.alpha * mean.sqrt()
                fill = s[seen].mean()
                # unseen positions: keep previous value if one exists, else layer mean
                prev = st.sigma if st.sigma_set else torch.full_like(s, float(fill))
                s = torch.where(seen, s, prev)
            raw[i] = s.clamp(cfg.sigma_min, cfg.sigma_max)

        if cfg.mode == "uniform_matched":
            # RMS of the per-row Jacobian sigmas over all observed (layer, position) pairs
            sq = torch.cat([raw[i][observed[i].count > 0] ** 2 for i in raw])
            g = float(sq.mean().sqrt())
            self._global_sigma = g
            raw = {i: torch.full_like(s, g) for i, s in raw.items()}

        for i, s in raw.items():
            st = observed[i]
            if st.sigma_set and cfg.ema > 0:
                s = cfg.ema * st.sigma + (1 - cfg.ema) * s
            st.sigma.copy_(s)
            st.sigma_set = True
        self.history.append({"step": self.global_step + 1, **self.summary()})

    def step(self) -> None:
        """Call once after every optimiser step."""
        cfg = self.config
        if self.measuring:
            self.global_step += 1
            return
        if cfg.mode in JACOBIAN_MODES and self.enabled:
            nxt = self.global_step + 1
            boundary = nxt >= cfg.warmup_steps and (nxt - cfg.warmup_steps) % cfg.update_interval == 0
            if boundary:
                self._update_sigma()
                self.reset_statistics()
        self.global_step += 1

    # -------------------------------------------------------------- reporting
    @torch.no_grad()
    def summary(self) -> dict:
        """Small JSON-serialisable summary for logging."""
        out: dict = {"global_step": self.global_step, "mode": self.config.mode, "in_warmup": self.in_warmup}
        sig = [float(st.sigma[: max(1, st.max_seen)].mean()) for _, st in sorted(self._layers.items()) if st.sigma_set]
        if sig:
            out["sigma_mean"] = sum(sig) / len(sig)
            out["sigma_per_layer"] = [round(x, 5) for x in sig]
        if self._global_sigma is not None:
            out["sigma_global"] = self._global_sigma
        return out

    def state_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "global_step": self.global_step,
            "layers": {
                i: {"sigma": st.sigma.cpu(), "sigma_set": st.sigma_set, "max_seen": st.max_seen}
                for i, st in self._layers.items()
            },
            "history": self.history,
        }

    def load_state_dict(self, state: dict) -> None:
        self.config = JacobianNoiseConfig.from_dict(state["config"])
        self.global_step = int(state["global_step"])
        self.history = list(state.get("history", []))
        self._layers = {}
        for i, d in state["layers"].items():
            st = self._layer(int(i), torch.device("cpu"))
            st.sigma.copy_(d["sigma"])
            st.sigma_set = bool(d["sigma_set"])
            st.max_seen = int(d.get("max_seen", 0))
