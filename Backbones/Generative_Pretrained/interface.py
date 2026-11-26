"""Shared generation interface consumed by adaptive-compute methods.

Methods in other modules (for example ``Adaptive_Stopping``) talk to a
generative backbone only through :class:`GenerativeBackbone`. Concrete
wrappers live next to this file (``hf_causal_lm.py``, ``vllm_causal_lm.py``,
``mock_mcq_lm.py``).

Token accounting conventions (used by every wrapper):

* ``Generation.num_tokens`` counts decoded (new) tokens only.
* ``OptionScores.aux_tokens`` is informational: the tokens the backbone
  actually processed for that call. Budgets charge the paper-protocol
  estimate instead (scoring suffix + 1 scored position, prefix caching
  assumed), see ``Adaptive_Stopping/common/rollout.py``.
* Prompt prefill tokens are reported separately by callers through
  :meth:`GenerativeBackbone.count_tokens`; they are not part of the
  decode or total budgets used by the SCPS paper protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Generation:
    """One sampled continuation."""

    text: str
    num_tokens: int
    finished: bool  # True if generation ended on EOS/stop, False if cut by max_tokens
    token_ids: Optional[List[int]] = None


@dataclass(frozen=True)
class OptionScores:
    """Log-probabilities of each option label being the next answer token.

    ``logprobs`` is renormalized over the given labels (log-softmax), so
    ``sum(exp(v)) == 1`` up to float error.
    """

    logprobs: Dict[str, float]
    aux_tokens: int

    def probs(self) -> Dict[str, float]:
        return {k: math.exp(v) for k, v in self.logprobs.items()}


@runtime_checkable
class GenerativeBackbone(Protocol):
    """Minimal interface for sampling-based test-time compute methods."""

    def render_prompt(self, user_message: str, assistant_prefix: str = "") -> str:
        """Turn a user message (and optional partial assistant answer) into model input text."""

    def generate(
        self,
        prompt: str,
        n: int,
        max_tokens: int,
        temperature: float,
        seed: Optional[int] = None,
    ) -> List[Generation]:
        """Sample ``n`` continuations of ``prompt``."""

    def count_tokens(self, text: str) -> int:
        """Number of tokens in ``text`` under this backbone's tokenizer."""

    def get_model_metadata(self) -> dict:
        """Registry metadata: id, revision, dtype, device and similar fields."""


@runtime_checkable
class OptionScoringBackbone(GenerativeBackbone, Protocol):
    """Backbone that can also score the next answer label without sampling."""

    def score_next_options(self, prompt: str, labels: Sequence[str]) -> OptionScores:
        """Log-probabilities of each label as the next token after ``prompt``."""


def log_softmax(values: Dict[str, float]) -> Dict[str, float]:
    """Numerically stable log-softmax over a dict of scores."""
    if not values:
        return {}
    m = max(values.values())
    z = m + math.log(sum(math.exp(v - m) for v in values.values()))
    return {k: v - z for k, v in values.items()}


def logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


__all__ = [
    "Generation",
    "OptionScores",
    "GenerativeBackbone",
    "OptionScoringBackbone",
    "log_softmax",
    "logsumexp",
]
