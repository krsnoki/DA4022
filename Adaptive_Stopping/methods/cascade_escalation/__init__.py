"""LLM cascade escalation (Bouchard 2026, "Is Escalation Worth It?", arXiv 2605.06350).

Confidence-thresholded model cascades: answer with the cheapest model and
escalate to the next model only when its confidence score is below a threshold.
"""

from .policy import CascadePolicy, CascadeStoppingPolicy, evaluate_policy, stop_index
from .scores import SCORE_NAMES, token_scores_from_topk
from .search import (
    Frontier,
    budget_grid,
    build_frontiers,
    enumerate_sequence,
    pareto_frontier,
    select_and_evaluate,
    threshold_grid,
)
from .learned_scores import ALL_SCORE_KINDS, ScoreProvider

__all__ = [
    "ALL_SCORE_KINDS",
    "CascadePolicy",
    "CascadeStoppingPolicy",
    "Frontier",
    "SCORE_NAMES",
    "ScoreProvider",
    "budget_grid",
    "build_frontiers",
    "enumerate_sequence",
    "evaluate_policy",
    "pareto_frontier",
    "select_and_evaluate",
    "stop_index",
    "threshold_grid",
    "token_scores_from_topk",
]
