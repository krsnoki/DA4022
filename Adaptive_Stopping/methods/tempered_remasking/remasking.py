"""Remasking (unmasking-order) policies for masked diffusion language models.

Implements the four algorithms of Appendix C of
"A Tale of Two Temperatures: Simple, Efficient, and Diverse Sampling from
Diffusion Language Models" (Olausson, Jazbec, et al., ICML 2026), plus the two
reference baselines used throughout the paper:

==========================  =============  ===================================
Policy                      Paper ref.     Selection rule for U_t
==========================  =============  ===================================
LowConfidence (LC)          Alg. 1, Eq. 3  argtop-K_{k in M_t} c_t^k
TemperedLowConfidence(TLC)  Alg. 2         K draws w/o replacement from
                                           Cat(c~_t), c~_t^k ∝ (c_t^k)^{1/T_pos}
RandomRemasking             Sec. 2.2       K uniform draws w/o replacement
                                           (= TLC with T_pos = ∞)
ConfidenceThreshold (CT)    Alg. 3, Eq. 4  {k in M_t : c_t^k > λ}   (Fast-dLLM)
TemperedConfidenceThreshold Alg. 4         {k in M_t : b_t^k = 1},
  (TCT)                                    b_t^k ~ Ber(σ((c_t^k − λ)/T_pos))
Autoregressive (AR)         Sec. 2.3/4.1   leftmost masked position
==========================  =============  ===================================

Token sampling (line 4 of every algorithm) and the confidence definition
(line 5 / footnote 1) live in :func:`sample_tokens`.

All functions are batched: tensors have a leading batch dimension ``B`` and a
sequence dimension ``N``. ``candidates`` is a boolean ``[B, N]`` tensor marking
the positions that may be unmasked at this step (masked positions inside the
current semi-autoregressive block); the policies never select anything else.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

__all__ = [
    "CONFIDENCE_MODES",
    "TokenProposal",
    "gumbel_noise",
    "sample_tokens",
    "get_num_transfer_tokens",
    "topk_per_row",
    "RemaskingPolicy",
    "LowConfidence",
    "TemperedLowConfidence",
    "RandomRemasking",
    "ConfidenceThreshold",
    "TemperedConfidenceThreshold",
    "Autoregressive",
    "build_policy",
]

#: Supported confidence definitions.
#:
#: ``"sampled"``          c = q(x | x_t), x ~ q(· | x_t; T_token)   (footnote 1; the
#:                        LLaDA / Fast-dLLM reference implementations; paper default)
#: ``"max"``              c = max_v q(v | x_t)                      (Sec. 2.2 definition;
#:                        identical to "sampled" when T_token = 0)
#: ``"tempered_sampled"`` c = q(x | x_t; T_token)                   (sampled-token
#:                        confidence taken *post*-tempering)
#: ``"tempered_max"``     c = max_v q(v | x_t; T_token)             (post-tempering
#:                        max-confidence, exactly c^j(T) of Prop. D.8)
CONFIDENCE_MODES = ("sampled", "max", "tempered_sampled", "tempered_max")


# --------------------------------------------------------------------------- #
# Randomness helpers
# --------------------------------------------------------------------------- #
def _uniform(shape, device, generator: Optional[torch.Generator], dtype=torch.float64) -> torch.Tensor:
    """U(0, 1) samples strictly inside (0, 1) so that logs are finite."""
    u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    finfo = torch.finfo(dtype)
    return u.clamp_(min=finfo.tiny, max=1.0 - finfo.eps)


def gumbel_noise(shape, device, generator: Optional[torch.Generator] = None, dtype=torch.float64) -> torch.Tensor:
    """Standard Gumbel(0, 1) noise ``-log(-log U)``.

    float64 by default: the LLaDA authors report that low-precision Gumbel
    noise measurably degrades sample quality for masked diffusion models.
    """
    return -torch.log(-torch.log(_uniform(shape, device, generator, dtype)))


# --------------------------------------------------------------------------- #
# Token sampling and confidences (lines 4-5 of Algorithms 1-4)
# --------------------------------------------------------------------------- #
@dataclass
class TokenProposal:
    """Per-position proposal ``x_0^k ~ q_θ^k(· | x_t; T_token)`` and its confidence.

    Attributes:
        tokens:     ``[B, N]`` long, proposed clean token at every position.
        confidence: ``[B, N]`` float64, confidence ``c_t^k`` used by the policy.
        entropy:    ``[B, N]`` float64 predictive entropy (nats) of the untempered
                    ``q_θ^k(· | x_t)``, or ``None`` if not requested.
    """

    tokens: torch.Tensor
    confidence: torch.Tensor
    entropy: Optional[torch.Tensor] = None


@torch.no_grad()
def sample_tokens(
    logits: torch.Tensor,
    token_temperature: float,
    generator: Optional[torch.Generator] = None,
    confidence_mode: str = "sampled",
    compute_entropy: bool = False,
    dtype: torch.dtype = torch.float64,
    max_chunk_elements: int = 1 << 26,
) -> TokenProposal:
    """Sample ``x_0^k`` for every position and compute its confidence.

    Args:
        logits: ``[B, N, V]`` model logits for ``q_θ(· | x_t)``.
        token_temperature: ``T_token >= 0``. ``0`` means greedy (argmax).
        generator: optional RNG for reproducibility.
        confidence_mode: one of :data:`CONFIDENCE_MODES`.
        compute_entropy: also return the per-position predictive entropy.
        dtype: precision for the Gumbel-max sampling and softmax.
        max_chunk_elements: the batch is processed in row chunks of at most this
            many logits so that the float64 temporaries stay bounded (a full
            LLaDA batch, V ≈ 126k, would otherwise need several GB per copy).

    Returns:
        :class:`TokenProposal`.
    """
    if confidence_mode not in CONFIDENCE_MODES:
        raise ValueError(f"confidence_mode must be one of {CONFIDENCE_MODES}, got {confidence_mode!r}")
    if token_temperature < 0:
        raise ValueError("token_temperature must be >= 0")
    B = logits.shape[0]
    per_row = max(1, logits[0].numel()) if B else 1
    rows = max(1, max_chunk_elements // per_row)
    if B <= rows:
        return _sample_chunk(logits, token_temperature, generator, confidence_mode, compute_entropy, dtype)
    parts = [
        _sample_chunk(logits[s : s + rows], token_temperature, generator, confidence_mode, compute_entropy, dtype)
        for s in range(0, B, rows)
    ]
    return TokenProposal(
        tokens=torch.cat([p.tokens for p in parts]),
        confidence=torch.cat([p.confidence for p in parts]),
        entropy=torch.cat([p.entropy for p in parts]) if compute_entropy else None,
    )


def _sample_chunk(logits, token_temperature, generator, confidence_mode, compute_entropy, dtype) -> TokenProposal:
    z = logits.to(dtype)
    if token_temperature == 0:
        tokens = z.argmax(dim=-1)
    else:
        # Gumbel-max trick: argmax(z / T + G) ~ Cat(softmax(z / T)).
        tokens = (z / token_temperature + gumbel_noise(z.shape, z.device, generator, dtype)).argmax(dim=-1)

    log_p = torch.log_softmax(z, dim=-1)  # untempered q_θ(· | x_t)
    if confidence_mode == "sampled":
        confidence = log_p.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).exp()
    elif confidence_mode == "max":
        confidence = log_p.max(dim=-1).values.exp()
    elif token_temperature == 0:
        # tempered modes: lim_{T->0} softmax(z/T) is one-hot on the argmax (no ties).
        confidence = torch.ones_like(tokens, dtype=dtype)
    else:
        log_pt = torch.log_softmax(z / token_temperature, dim=-1)
        if confidence_mode == "tempered_sampled":
            confidence = log_pt.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).exp()
        else:  # tempered_max
            confidence = log_pt.max(dim=-1).values.exp()

    entropy = None
    if compute_entropy:
        entropy = -(log_p.exp() * log_p).sum(dim=-1)
    return TokenProposal(tokens=tokens, confidence=confidence, entropy=entropy)


# --------------------------------------------------------------------------- #
# Fixed-K schedule (LLaDA's `get_num_transfer_tokens`)
# --------------------------------------------------------------------------- #
def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Number of tokens ``K`` to unmask at each of ``steps`` steps, per row.

    Matches the reference LLaDA implementation: the masked count of a block is
    split as evenly as possible over the steps, the remainder going to the
    earliest steps (linear noise schedule).

    Args:
        mask_index: ``[B, n]`` boolean mask of the block's masked positions.
        steps: number of denoising steps allotted to the block (``>= 1``).

    Returns:
        ``[B, steps]`` long tensor whose rows sum to the masked count.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = base.expand(-1, steps).clone()
    arange = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    num += (arange < remainder).long()
    return num


# --------------------------------------------------------------------------- #
# Selection primitives
# --------------------------------------------------------------------------- #
def topk_per_row(scores: torch.Tensor, candidates: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Boolean mask of the ``k[b]`` highest-scoring candidate positions of row ``b``.

    ``k`` is clipped to the number of candidates in the row. Non-candidates are
    never selected. Ties are broken towards lower indices (stable sort).
    """
    if scores.shape != candidates.shape:
        raise ValueError("scores and candidates must have the same shape")
    k = k.to(device=scores.device, dtype=torch.long).reshape(-1)
    if k.numel() != scores.shape[0]:
        raise ValueError("k must have one entry per row")
    masked_scores = scores.masked_fill(~candidates, float("-inf"))
    order = torch.argsort(masked_scores, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(-1, order, torch.arange(scores.shape[-1], device=scores.device).expand_as(order).contiguous())
    k = torch.minimum(k.clamp_min(0), candidates.sum(dim=-1))
    return (ranks < k.unsqueeze(-1)) & candidates


def _check_tpos(tpos: float) -> float:
    tpos = float(tpos)
    if math.isnan(tpos) or tpos < 0:
        raise ValueError(f"position temperature must be >= 0 (or inf), got {tpos}")
    return tpos


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
class RemaskingPolicy(ABC):
    """A rule choosing the unmasking set ``U_t ⊆ M_t`` from confidences.

    Class attributes:
        name:       short identifier.
        fixed_k:    ``True`` if the policy unmasks a scheduled number ``K`` of
                    positions per step (LC, TLC, random); ``False`` if the set
                    size is adaptive (CT, TCT) or always one (AR).
        adaptive:   ``True`` for threshold policies whose set may be empty.
        stochastic: ``True`` if the selection itself is random (given ``c_t``).
        uses_blocks: ``False`` for AR, which ignores the semi-AR block schedule.
    """

    name: str = "base"
    fixed_k: bool = False
    adaptive: bool = False
    stochastic: bool = False
    uses_blocks: bool = True

    @abstractmethod
    def select(
        self,
        confidence: torch.Tensor,
        candidates: torch.Tensor,
        num_transfer: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return a boolean ``[B, N]`` unmasking set (subset of ``candidates``)."""

    def config(self) -> dict:
        return {"name": self.name}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        args = ", ".join(f"{k}={v}" for k, v in self.config().items() if k != "name")
        return f"{type(self).__name__}({args})"


class LowConfidence(RemaskingPolicy):
    """Algorithm 1 / Eq. (3): unmask the K most confident masked positions."""

    name = "low_confidence"
    fixed_k = True

    def select(self, confidence, candidates, num_transfer=None, generator=None):
        if num_transfer is None:
            raise ValueError("LowConfidence requires num_transfer (K per row)")
        return topk_per_row(confidence, candidates, num_transfer)


class TemperedLowConfidence(RemaskingPolicy):
    """Algorithm 2 (TLC): K draws without replacement from ``Cat(c~_t)``.

    ``c~_t^k ∝ (c_t^k)^{1/T_pos} · 1(k ∈ M_t)``. Sampling K items without
    replacement from a categorical (sequentially renormalising after each
    draw) is implemented exactly with the Gumbel-top-k trick: the top-K of
    ``log c~_t^k + G_k`` with i.i.d. ``G_k ~ Gumbel(0, 1)``.

    ``T_pos = 0`` recovers deterministic LC (Alg. 1) and ``T_pos = ∞`` recovers
    uniform random remasking, as stated in Sec. 3.2.
    """

    name = "tlc"
    fixed_k = True

    def __init__(self, tpos: float = 1.0):
        self.tpos = _check_tpos(tpos)
        self.stochastic = self.tpos > 0

    def select(self, confidence, candidates, num_transfer=None, generator=None):
        if num_transfer is None:
            raise ValueError("TemperedLowConfidence requires num_transfer (K per row)")
        if self.tpos == 0:
            return topk_per_row(confidence, candidates, num_transfer)
        g = gumbel_noise(confidence.shape, confidence.device, generator)
        if math.isinf(self.tpos):
            scores = g
        else:
            log_c = torch.log(confidence.to(torch.float64).clamp_min(torch.finfo(torch.float64).tiny))
            scores = log_c / self.tpos + g
        return topk_per_row(scores, candidates, num_transfer)

    def config(self):
        return {"name": self.name, "tpos": self.tpos}


class RandomRemasking(TemperedLowConfidence):
    """Random remasking (Sec. 2.2, Zheng et al. 2025): TLC at ``T_pos = ∞``."""

    name = "random"

    def __init__(self):
        super().__init__(tpos=float("inf"))

    def config(self):
        return {"name": self.name}


class ConfidenceThreshold(RemaskingPolicy):
    """Algorithm 3 / Eq. (4) (CT; Fast-dLLM): unmask all ``c_t^k > λ``."""

    name = "confidence_threshold"
    adaptive = True

    def __init__(self, threshold: float = 0.6):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold λ must lie in [0, 1]")
        self.threshold = float(threshold)

    def select(self, confidence, candidates, num_transfer=None, generator=None):
        return (confidence > self.threshold) & candidates

    def config(self):
        return {"name": self.name, "threshold": self.threshold}


class TemperedConfidenceThreshold(RemaskingPolicy):
    """Algorithm 4 (TCT): stochastic inclusion ``b ~ Ber(σ((c − λ)/T_pos))``.

    ``T_pos = 0`` recovers the hard threshold of CT (strict ``c > λ``);
    ``T_pos = ∞`` includes every masked position independently w.p. 1/2.
    """

    name = "tct"
    adaptive = True

    def __init__(self, threshold: float = 0.6, tpos: float = 0.1):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold λ must lie in [0, 1]")
        self.threshold = float(threshold)
        self.tpos = _check_tpos(tpos)
        self.stochastic = self.tpos > 0

    def inclusion_probability(self, confidence: torch.Tensor) -> torch.Tensor:
        """``σ((c − λ)/T_pos)`` with the ``T_pos ∈ {0, ∞}`` limits handled exactly."""
        c = confidence.to(torch.float64)
        if self.tpos == 0:
            return (c > self.threshold).to(torch.float64)
        if math.isinf(self.tpos):
            return torch.full_like(c, 0.5)
        return torch.sigmoid((c - self.threshold) / self.tpos)

    def select(self, confidence, candidates, num_transfer=None, generator=None):
        if self.tpos == 0:
            return (confidence > self.threshold) & candidates
        p = self.inclusion_probability(confidence)
        u = torch.rand(p.shape, device=p.device, dtype=torch.float64, generator=generator)
        return (u < p) & candidates

    def config(self):
        return {"name": self.name, "threshold": self.threshold, "tpos": self.tpos}


class Autoregressive(RemaskingPolicy):
    """Left-to-right generation: always unmask the leftmost masked position.

    Used as the AR baseline of Ni et al. (2026) with the same dLLM; one token
    per forward pass, ignoring confidences and the block schedule.
    """

    name = "autoregressive"
    uses_blocks = False

    def select(self, confidence, candidates, num_transfer=None, generator=None):
        sel = torch.zeros_like(candidates)
        has = candidates.any(dim=-1)
        first = candidates.to(torch.long).argmax(dim=-1)  # index of first True
        rows = torch.nonzero(has, as_tuple=False).squeeze(-1)
        sel[rows, first[rows]] = True
        return sel


_ALIASES = {
    "lc": "low_confidence",
    "low_confidence": "low_confidence",
    "tlc": "tlc",
    "tempered_low_confidence": "tlc",
    "random": "random",
    "ct": "confidence_threshold",
    "fast_dllm": "confidence_threshold",
    "confidence_threshold": "confidence_threshold",
    "tct": "tct",
    "tempered_confidence_threshold": "tct",
    "ar": "autoregressive",
    "autoregressive": "autoregressive",
}


def build_policy(name: str, tpos: Optional[float] = None, threshold: Optional[float] = None) -> RemaskingPolicy:
    """Construct a policy by name.

    Paper defaults (Sec. 4.1): TLC ``T_pos = 1``; TCT ``T_pos = 0.1``,
    ``λ = 0.6``; CT ``λ = 0.6``.
    """
    key = _ALIASES.get(name.lower())
    if key is None:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(set(_ALIASES))}")
    if key == "low_confidence":
        return LowConfidence()
    if key == "tlc":
        return TemperedLowConfidence(1.0 if tpos is None else tpos)
    if key == "random":
        return RandomRemasking()
    if key == "confidence_threshold":
        return ConfidenceThreshold(0.6 if threshold is None else threshold)
    if key == "tct":
        return TemperedConfidenceThreshold(0.6 if threshold is None else threshold, 0.1 if tpos is None else tpos)
    return Autoregressive()
