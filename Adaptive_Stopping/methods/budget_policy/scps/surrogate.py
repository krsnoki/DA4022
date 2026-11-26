"""Training-free plug-in surrogate for the SCPS score (Equation 7).

The paper deploys a learned surrogate bundle (MLP heads over pooled
Qwen features, trained on a disjoint 512-question split) that predicts
answer distributions, pairwise collisions and completion lengths. Those
weights are not released. This module replaces them with estimates that
need no training:

* answer distribution of an action: terminal-option log-probabilities of
  its source (root prompt, targeted prompt or unfinished prefix), sharpened
  by the sampling temperature and updated online with a Dirichlet
  posterior over observed answers from the same source;
* targeted actions: mixture ``s * 1{target} + (1 - s) * p_root`` with the
  steering rate ``s`` learned online (Beta posterior);
* null mass: Beta posterior over completions with no parseable answer;
* collision/overlap: conditional independence of actions given the
  current information, so ``omega(e, f) = sum_a pi(a) q_e(a) q_f(a)``;
* length/cost: running mean of decode tokens per completed sample.

Because of this substitution, results produced with this surrogate must be
labelled ``approximate`` relative to the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence


def softmax_from_logprobs(logprobs: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    if not logprobs:
        return {}
    t = max(temperature, 1e-6)
    scaled = {k: v / t for k, v in logprobs.items()}
    m = max(scaled.values())
    ex = {k: math.exp(v - m) for k, v in scaled.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}


def uniform(labels: Sequence[str]) -> Dict[str, float]:
    return {lab: 1.0 / len(labels) for lab in labels}


@dataclass
class BetaPosterior:
    a: float
    b: float

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def update(self, successes: int, failures: int) -> None:
        self.a += successes
        self.b += failures


@dataclass
class SourceStats:
    """Observed outcomes for one action source."""

    counts: Dict[str, int] = field(default_factory=dict)
    nulls: int = 0
    executions: int = 0

    @property
    def n_answers(self) -> int:
        return sum(self.counts.values())


@dataclass
class Source:
    """An action source: a prompt to sample from plus its prior answer distribution."""

    key: str
    kind: str  # "root" | "target" | "prefix"
    prompt: str
    prior: Dict[str, float]  # predicted answer distribution before observations (sums to 1)
    target: Optional[str] = None
    prefix_text: str = ""
    prefix_tokens: int = 0


class PluginSurrogate:
    """Estimates v, omega and cost for exact-answer SCPS actions."""

    def __init__(
        self,
        labels: Sequence[str],
        posterior: Dict[str, float],
        root_prior: Dict[str, float],
        *,
        prior_strength: float = 4.0,
        null_prior: Sequence[float] = (1.0, 9.0),
        steer_prior: Sequence[float] = (1.0, 1.0),
        length_prior: float = 48.0,
        length_prior_weight: float = 1.0,
    ):
        self.labels = tuple(labels)
        self.pi = dict(posterior)
        self.root_prior = dict(root_prior)
        self.prior_strength = float(prior_strength)
        self.null = BetaPosterior(*null_prior)
        self.steer = BetaPosterior(*steer_prior)
        self._len_sum = float(length_prior) * length_prior_weight
        self._len_n = float(length_prior_weight)
        self.stats: Dict[str, SourceStats] = {}

    # ------------------------------------------------------------ estimates
    @property
    def mean_length(self) -> float:
        return self._len_sum / self._len_n

    def stats_for(self, key: str) -> SourceStats:
        return self.stats.setdefault(key, SourceStats())

    def _dirichlet(self, prior: Dict[str, float], st: SourceStats) -> Dict[str, float]:
        c0 = self.prior_strength
        n = st.n_answers
        return {a: (c0 * prior.get(a, 0.0) + st.counts.get(a, 0)) / (c0 + n) for a in self.labels}

    def answer_distribution(self, src: Source) -> Dict[str, float]:
        """Predicted per-sample answer distribution (sums to ``1 - null_mass``)."""
        st = self.stats_for(src.key)
        if src.kind == "target":
            s = self.steer.mean
            base = self._dirichlet(self.root_prior, self.stats_for("root"))
            dist = {a: (1.0 - s) * base[a] for a in self.labels}
            dist[src.target] += s
        else:
            dist = self._dirichlet(src.prior, st)
        keep = 1.0 - self.null.mean
        return {a: keep * p for a, p in dist.items()}

    @staticmethod
    def hit_prob(p: float, n: int) -> float:
        return 1.0 - (1.0 - p) ** n

    def residual_support(self, p: Dict[str, float], n: int, seen: Iterable[str]) -> float:
        """``v(e) = sum_{a unseen} pi(a) * (1 - (1 - p_e(a))^n)``."""
        seen = set(seen)
        return sum(self.pi[a] * self.hit_prob(p[a], n) for a in self.labels if a not in seen)

    def overlap(self, p: Dict[str, float], n: int, p2: Dict[str, float], n2: int, seen: Iterable[str]) -> float:
        """``omega(e, f)`` under conditional independence given the current information."""
        seen = set(seen)
        return sum(
            self.pi[a] * self.hit_prob(p[a], n) * self.hit_prob(p2[a], n2) for a in self.labels if a not in seen
        )

    def seen_mass(self, p: Dict[str, float], seen: Iterable[str]) -> float:
        """Per-sample probability of landing on an already-covered class."""
        return sum(p[a] for a in set(seen))

    def expected_cost(self, src: Source, n: int, max_new_tokens: int) -> float:
        cap = max(0, max_new_tokens - src.prefix_tokens)
        return n * min(self.mean_length, cap)

    # -------------------------------------------------------------- updates
    def observe(self, src: Source, answers: List[Optional[str]], lengths: List[int]) -> None:
        st = self.stats_for(src.key)
        st.executions += 1
        nulls = sum(1 for a in answers if a is None)
        st.nulls += nulls
        for a in answers:
            if a is not None:
                st.counts[a] = st.counts.get(a, 0) + 1
        self.null.update(nulls, len(answers) - nulls)
        if src.kind == "target":
            hits = sum(1 for a in answers if a == src.target)
            self.steer.update(hits, len(answers) - hits)
        for L in lengths:
            self._len_sum += L
            self._len_n += 1

    def snapshot(self) -> dict:
        return {
            "null_mean": round(self.null.mean, 4),
            "steer_mean": round(self.steer.mean, 4),
            "mean_length": round(self.mean_length, 2),
        }


__all__ = ["PluginSurrogate", "Source", "SourceStats", "BetaPosterior", "softmax_from_logprobs", "uniform"]
