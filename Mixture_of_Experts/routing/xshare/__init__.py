"""XShare: batch-aware expert selection for MoE inference (OpenReview KmY5XUgEYe)."""

from .selection import (
    RoutingResult,
    XShareConfig,
    aggregate_scores,
    expert_scores,
    gpu_aware_greedy,
    greedy_select,
    route_within,
    select_batch,
    select_expert_parallel,
    select_per_request,
    select_speculative,
    warmup_set,
    xshare_route,
)

__all__ = [
    "RoutingResult",
    "XShareConfig",
    "aggregate_scores",
    "expert_scores",
    "gpu_aware_greedy",
    "greedy_select",
    "route_within",
    "select_batch",
    "select_expert_parallel",
    "select_per_request",
    "select_speculative",
    "warmup_set",
    "xshare_route",
]
