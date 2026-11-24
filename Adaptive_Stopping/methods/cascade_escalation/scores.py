"""Deferral (confidence) scores computed from a single generation's token log-probs.

These are the five "off-the-shelf" white-box scores used by Bouchard (2026),
"Is Escalation Worth It?" (arXiv 2605.06350, App. B.3). They need no extra
model call: they are computed from the top-K log-probabilities that the model
already produced while generating its answer.

For a response of L generated tokens, let ``p_j`` be the probability of the
chosen token at position j and ``q_{j,1} >= q_{j,2} >= ...`` the top-K
probabilities at that position, renormalised to sum to one over the K
candidates.

* ``length_norm``  (LNP)  : (prod_j p_j)^(1/L), the geometric-mean token prob.
* ``min_prob``     (MTP)  : min_j p_j.
* ``prob_margin``  (PM)   : mean_j (q_{j,1} - q_{j,2}).
* ``mean_negentropy`` (ATN@K): mean_j [1 - H_K(q_j) / log K].
* ``min_negentropy``  (MTN@K): min_j  [1 - H_K(q_j) / log K].

Higher score = more confident. A query is escalated when ``score < tau``.

Our implementation follows the definitions in the paper. The paper computed
them through the UQLM library; small numerical differences from UQLM are
possible (e.g. handling of empty responses), see the paper card.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

SCORE_NAMES = (
    "mean_negentropy",
    "min_negentropy",
    "prob_margin",
    "min_prob",
    "length_norm",
)

# Value assigned when a response has no generated tokens (no evidence of
# confidence): the least-confident value, so such queries escalate first.
EMPTY_RESPONSE_SCORE = 0.0


def token_scores_from_topk(
    chosen_logprobs: Sequence[float],
    topk_logprobs: Sequence[Sequence[float]],
    k: int | None = None,
) -> Dict[str, float]:
    """Compute the five scores for one response.

    Args:
        chosen_logprobs: length-L natural-log probability of each generated token.
        topk_logprobs: length-L list; entry j holds the top-K log-probs at
            position j (any order; at least 2 entries per position).
        k: number of candidates to use (defaults to all provided). The paper
            uses K=15 where available (K=5 for some hosted Llama/Qwen models).

    Returns:
        dict mapping each name in :data:`SCORE_NAMES` to a float in [0, 1].
    """
    chosen = np.asarray(chosen_logprobs, dtype=np.float64)
    if chosen.size == 0:
        return {name: EMPTY_RESPONSE_SCORE for name in SCORE_NAMES}
    if len(topk_logprobs) != chosen.size:
        raise ValueError("chosen_logprobs and topk_logprobs must have equal length")

    width = min(len(row) for row in topk_logprobs)
    if k is not None:
        width = min(width, int(k))
    if width < 2:
        raise ValueError("need at least 2 top-k log-probs per position")
    top = np.asarray([sorted(row, reverse=True)[:width] for row in topk_logprobs], dtype=np.float64)

    # renormalise over the K candidates (numerically stable softmax of log-probs)
    top = top - top.max(axis=1, keepdims=True)
    q = np.exp(top)
    q /= q.sum(axis=1, keepdims=True)

    p = np.exp(np.clip(chosen, -np.inf, 0.0))
    ent = -(q * np.log(np.clip(q, 1e-300, None))).sum(axis=1)
    negent = 1.0 - ent / np.log(width)
    negent = np.clip(negent, 0.0, 1.0)

    return {
        "mean_negentropy": float(negent.mean()),
        "min_negentropy": float(negent.min()),
        "prob_margin": float((q[:, 0] - q[:, 1]).mean()),
        "min_prob": float(p.min()),
        "length_norm": float(np.exp(np.clip(chosen, -745.0, 0.0).mean())),
    }


def score_matrix(records: Sequence[Dict[str, float]], name: str) -> np.ndarray:
    """Extract one score from a list of per-query score dicts as a float array."""
    if name not in SCORE_NAMES:
        raise KeyError(f"unknown score {name!r}; choose from {SCORE_NAMES}")
    return np.asarray([r[name] for r in records], dtype=np.float64)
