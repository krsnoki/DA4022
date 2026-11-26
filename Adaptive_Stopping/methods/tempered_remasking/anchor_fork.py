"""A synthetic masked "dLLM" that satisfies the anchor-fork degeneracy *exactly*.

This makes the idealised model of Sec. 3.1 / Appendix D concrete and
executable with the *same* sampler used for real dLLMs, so that the theory
(Lemma D.3, Prop. D.4, Remark D.5, Prop. D.9, Cor. D.7) can be checked by
Monte Carlo against exact dynamic-programming values. It mirrors Figure 1
(``import ?????`` with "numpy"/"statistics" solutions):

* the **fork** ℓ is binary, values ``FORK_A`` / ``FORK_B`` (the semantic outcome
  ``⟦x_0⟧`` is the fork value, so ℓ is a (0, 0)-fork token, Def. D.1);
* each **anchor** ``a`` has a fixed confidence ``c_a`` and, while the fork is
  masked, prefers the "A" solution (``ANCHOR_A`` w.p. ``c_a``); once the fork is
  revealed the anchors follow it;
* revealing anchor ``a`` lowers the fork's entropy by exactly ``η_a``
  (Assumption D.2, condition 2), tilting it towards the majority value of the
  revealed anchors (ties -> "A");
* **filler** positions have a fixed confidence and never interact.

The constructor validates condition 1 (the persistent confidence gap
``c_a > c_ℓ`` in every reachable state).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .theory import binary_entropy, inverse_binary_entropy, lc_fork_entropy, ar_fork_entropy

__all__ = ["AnchorForkModel", "MASK", "FORK_A", "FORK_B", "ANCHOR_A", "ANCHOR_B", "FILL_X", "FILL_Y", "VOCAB_SIZE"]

MASK, FORK_A, FORK_B, ANCHOR_A, ANCHOR_B, FILL_X, FILL_Y = range(7)
VOCAB_SIZE = 7
_NEG = -1.0e4  # effectively zero probability, but finite (keeps p·log p = 0)


@dataclass
class AnchorForkModel:
    """Callable synthetic model (``ids [B, N] -> logits [B, N, V]``).

    Args:
        length: number of generated positions L (positions are 0-based within
            the generated region; ``prompt_length`` dummy prompt tokens precede it).
        fork_position: index of the fork ℓ.
        anchors: ``{position: (c_a, η_a)}``.
        h0: fork entropy ``H_q(x^ℓ | x_t)`` with no anchor revealed (<= ln 2).
        filler_confidence: confidence of every other position.
        prompt_length: number of prompt positions prepended by the caller.
    """

    length: int
    fork_position: int
    anchors: Dict[int, tuple]
    h0: float = math.log(2)
    filler_confidence: float = 0.9
    prompt_length: int = 0
    _anchor_pos: List[int] = field(init=False, repr=False)

    mask_token_id: int = MASK
    eos_token_ids: tuple = ()

    def __post_init__(self):
        self._anchor_pos = sorted(self.anchors)
        if self.fork_position in self.anchors or not 0 <= self.fork_position < self.length:
            raise ValueError("invalid fork position")
        if any(not 0 <= a < self.length for a in self._anchor_pos):
            raise ValueError("anchor positions must lie in [0, length)")
        etas = {a: e for a, (_, e) in self.anchors.items()}
        if not all(e > 0 for e in etas.values()) or not self.h0 > sum(etas.values()):
            raise ValueError("need η_a > 0 and H_0 > Σ η_a (Assumption D.2)")
        if self.h0 > math.log(2) + 1e-12:
            raise ValueError("binary fork: h0 <= ln 2")
        m = len(self._anchor_pos)
        if m > 16:
            raise ValueError("at most 16 anchors (exact tables are exponential)")
        # fork's dominant-value probability for each revealed-anchor subset (bitmask)
        self._p_fork = []
        for mask in range(2**m):
            h = self.h0 - sum(etas[a] for i, a in enumerate(self._anchor_pos) if mask >> i & 1)
            self._p_fork.append(inverse_binary_entropy(max(h, 0.0)))
        # Assumption D.2, condition 1: persistent confidence gap
        for mask in range(2**m):
            for i, a in enumerate(self._anchor_pos):
                if not mask >> i & 1 and not self.anchors[a][0] > self._p_fork[mask]:
                    raise ValueError(
                        f"confidence gap violated: c_a={self.anchors[a][0]} <= c_fork={self._p_fork[mask]:.4f}"
                    )

    # ------------------------------------------------------------------ #
    # model forward
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        P = self.prompt_length
        if N != P + self.length:
            raise ValueError(f"expected sequence length {P + self.length}, got {N}")
        dev = x.device
        logits = torch.full((B, N, VOCAB_SIZE), _NEG, dtype=torch.float64, device=dev)

        def set_binary(pos: int, tok_hi: torch.Tensor, tok_lo: torch.Tensor, p_hi: torch.Tensor):
            rows = torch.arange(B, device=dev)
            logits[rows, pos, tok_hi] = torch.log(p_hi)
            logits[rows, pos, tok_lo] = torch.log1p(-p_hi).clamp_min(_NEG)

        # fillers and prompt (every position that is neither the fork nor an anchor)
        special = {P + self.fork_position} | {P + a for a in self._anchor_pos}
        fc = torch.full((B,), self.filler_confidence, dtype=torch.float64, device=dev)
        for j in range(N):
            if j not in special:
                set_binary(j, torch.full((B,), FILL_X, device=dev), torch.full((B,), FILL_Y, device=dev), fc)

        gen = x[:, P:]
        fork_tok = gen[:, self.fork_position]
        fork_masked = fork_tok == MASK
        # anchors
        for a in self._anchor_pos:
            ca = torch.full((B,), self.anchors[a][0], dtype=torch.float64, device=dev)
            follow_b = fork_tok == FORK_B
            hi = torch.where(follow_b, torch.full_like(fork_tok, ANCHOR_B), torch.full_like(fork_tok, ANCHOR_A))
            lo = torch.where(follow_b, torch.full_like(fork_tok, ANCHOR_A), torch.full_like(fork_tok, ANCHOR_B))
            set_binary(P + a, hi, lo, ca)
        # fork
        bitmask = torch.zeros(B, dtype=torch.long, device=dev)
        n_a = torch.zeros(B, dtype=torch.long, device=dev)
        n_b = torch.zeros(B, dtype=torch.long, device=dev)
        for i, a in enumerate(self._anchor_pos):
            t = gen[:, a]
            bitmask += (t != MASK).long() << i
            n_a += (t == ANCHOR_A).long()
            n_b += (t == ANCHOR_B).long()
        p_table = torch.tensor(self._p_fork, dtype=torch.float64, device=dev)
        p = p_table[bitmask]
        tilt_b = n_b > n_a
        hi = torch.where(tilt_b, torch.full_like(n_a, FORK_B), torch.full_like(n_a, FORK_A))
        lo = torch.where(tilt_b, torch.full_like(n_a, FORK_A), torch.full_like(n_a, FORK_B))
        # once revealed, the fork's own prediction is irrelevant; keep it well defined
        p = torch.where(fork_masked, p, torch.full_like(p, 0.5))
        set_binary(P + self.fork_position, hi, lo, p)
        return logits

    # ------------------------------------------------------------------ #
    # exact quantities
    # ------------------------------------------------------------------ #
    @property
    def etas(self) -> Dict[int, float]:
        return {a: e for a, (_, e) in self.anchors.items()}

    def fork_confidence(self, revealed: frozenset) -> float:
        mask = sum(1 << i for i, a in enumerate(self._anchor_pos) if a in revealed)
        return self._p_fork[mask]

    def lemma_constants(self) -> Dict[int, tuple]:
        """``(δ_a, Δ_a)`` of Lemma D.3 for each anchor, over all reachable states."""
        out = {}
        for a in self._anchor_pos:
            others = [b for b in self._anchor_pos if b != a]
            vals = []
            for r in range(len(others) + 1):
                for S in itertools.combinations(others, r):
                    vals.append(math.log(self.anchors[a][0] / self.fork_confidence(frozenset(S))))
            out[a] = (min(vals), max(vals))
        return out

    def exact_tlc(self, tpos: float, token_temperature: float = 1.0) -> dict:
        """Exact statistics of TLC with K = 1, no blocks, confidence = max-prob.

        Exhaustive dynamic programme over (revealed-anchor set, #"A"-valued
        anchors). Fillers do not change any relevant confidence, so they only
        delay events and can be marginalised (as in the proof of Lemma D.3).

        Returns:
            ``expected_fork_entropy`` (E[H_q at fork reveal], the quantity of
            Prop. D.4), ``p_anchor_first`` per anchor, ``p_fork_A`` (marginal
            probability of the "A" solution) and ``semantic_entropy`` (entropy
            of the fork value across samples).
        """

        def w(c: float) -> float:
            if tpos == 0:
                raise ValueError("use T_pos > 0; T_pos = 0 is LC (see lc_fork_entropy)")
            return 1.0 if math.isinf(tpos) else c ** (1.0 / tpos)

        def sample_prob_hi(p_hi: float) -> float:
            """Probability that the dominant value is *sampled* under T_token."""
            if token_temperature == 0:
                return 1.0
            a = math.log(p_hi) / token_temperature
            b = math.log(1 - p_hi) / token_temperature if p_hi < 1 else -math.inf
            return 1.0 / (1.0 + math.exp(b - a))

        anchors = self._anchor_pos
        exp_h = 0.0
        p_first = {a: 0.0 for a in anchors}
        p_fork_a = 0.0
        # state: (frozenset revealed, n_A) -> prob
        layer = {(frozenset(), 0): 1.0}
        while layer:
            nxt: Dict[tuple, float] = {}
            for (S, nA), prob in layer.items():
                c_fork = self.fork_confidence(S)
                wf = w(c_fork)
                rem = [a for a in anchors if a not in S]
                W = wf + sum(w(self.anchors[a][0]) for a in rem)
                # fork chosen now
                pf = prob * wf / W
                h_now = self.h0 - sum(self.etas[a] for a in S)
                exp_h += pf * h_now
                nB = len(S) - nA
                p_hi = sample_prob_hi(c_fork)
                p_fork_a += pf * (p_hi if not nB > nA else 1 - p_hi)
                # an anchor chosen now
                for a in rem:
                    pa = prob * w(self.anchors[a][0]) / W
                    p_first[a] += pa
                    q_a = sample_prob_hi(self.anchors[a][0])  # anchors prefer "A" while fork masked
                    key_a = (S | {a}, nA + 1)
                    key_b = (S | {a}, nA)
                    nxt[key_a] = nxt.get(key_a, 0.0) + pa * q_a
                    if q_a < 1:
                        nxt[key_b] = nxt.get(key_b, 0.0) + pa * (1 - q_a)
            layer = nxt
        return {
            "expected_fork_entropy": exp_h,
            "p_anchor_first": p_first,
            "p_fork_A": p_fork_a,
            "semantic_entropy": binary_entropy(min(max(p_fork_a, 0.0), 1.0)),
        }

    def lc_fork_entropy(self) -> float:
        return lc_fork_entropy(self.h0, self.etas)

    def ar_fork_entropy(self) -> float:
        return ar_fork_entropy(self.h0, self.etas, self.fork_position)

    def prompt(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """Dummy prompt of ``prompt_length`` filler tokens."""
        return torch.full((batch_size, self.prompt_length), FILL_X, dtype=torch.long, device=device)
