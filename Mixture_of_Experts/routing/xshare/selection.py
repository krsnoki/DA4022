"""Batch-aware expert selection from the XShare paper.

Paper: "XShare: Collaborative in-Batch Expert Sharing for Faster MoE Inference"
(Vankov et al., https://openreview.net/forum?id=KmY5XUgEYe).

Everything here works on one MoE layer and one decode step. The input is the
router output for the tokens of that step, shape ``[n_tokens, n_experts]``.
The output is a boolean keep-mask over experts (the set ``S`` in the paper)
and the per-token routing inside that set.

The module has no dependency on any backbone or serving framework. The
integration layers (vLLM, Hugging Face) call :func:`xshare_route`.

Algorithms, with the paper's numbering:

* Algorithm 1 ``GREEDYSELECT``            -> :func:`greedy_select`
* Algorithm 2 batch-aware selection       -> :func:`select_batch`
* Algorithm 3 ``PERREQUESTSELECT``        -> :func:`select_per_request`
* Algorithm 4 speculative-decoding-aware  -> :func:`select_speculative`
* Algorithm 5 GPU-aware greedy            -> :func:`gpu_aware_greedy`
* Algorithm 6 expert-parallel-aware       -> :func:`select_expert_parallel`

Choices the paper leaves open are parameters of :class:`XShareConfig`. The
paper card records which values we use and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from torch import Tensor

ScoreKind = Literal["softmax", "logits"]
GateNorm = Literal["topk_softmax", "softmax_topk", "softmax_topk_renorm"]
BudgetMode = Literal["additional", "total"]


@dataclass(frozen=True)
class XShareConfig:
    """Settings for one selection call.

    Attributes:
        top_k: experts each token is routed to (``k`` in the paper).
        warmup_k0: each token's top-``k0`` experts are always kept (warm-up).
            ``0`` disables the warm-up.
        budget: the greedy budget ``b``. ``None`` disables pruning: every
            expert is kept and routing equals the unpruned baseline. ``0``
            with ``warmup_k0 > 0`` is OEA (Oncescu et al., 2025).
        budget_mode: ``"additional"`` adds ``budget`` experts on top of the
            warm-up set. ``"total"`` caps the whole set at ``budget``, as
            Algorithm 1 is written. The paper's tables match "additional".
        per_request_budget: per-request budget (``m_r``, labelled ``g`` in the
            tables). ``None`` skips the per-request stage (Algorithms 3-4).
        per_gpu_budget: per-GPU budget ``m_g`` (Algorithms 5-6). ``None``
            skips the GPU-aware stage.
        score: what is summed over the batch. ``"softmax"`` uses softmax of
            the router logits over all experts. ``"logits"`` sums raw logits.
        gate_norm: how gate weights are computed for the routed experts.
            ``"topk_softmax"``: softmax over the selected top-k logits
            (gpt-oss, Mixtral). ``"softmax_topk"``: softmax over all experts,
            then take the top-k values unchanged. ``"softmax_topk_renorm"``:
            the same, then rescale the k values to sum to one.
        min_keep: lower bound on the size of the kept set. ``None`` means
            ``top_k``, so every token can always be routed to ``k`` experts.
    """

    top_k: int
    warmup_k0: int = 0
    budget: Optional[int] = None
    budget_mode: BudgetMode = "additional"
    per_request_budget: Optional[int] = None
    per_gpu_budget: Optional[int] = None
    score: ScoreKind = "softmax"
    gate_norm: GateNorm = "topk_softmax"
    min_keep: Optional[int] = None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.warmup_k0 < 0:
            raise ValueError("warmup_k0 must not be negative")
        for name in ("budget", "per_request_budget", "per_gpu_budget", "min_keep"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
        if self.budget_mode not in ("additional", "total"):
            raise ValueError(f"unknown budget_mode {self.budget_mode!r}")
        if self.score not in ("softmax", "logits"):
            raise ValueError(f"unknown score {self.score!r}")
        if self.gate_norm not in ("topk_softmax", "softmax_topk", "softmax_topk_renorm"):
            raise ValueError(f"unknown gate_norm {self.gate_norm!r}")

    @property
    def prunes(self) -> bool:
        """True when this configuration can remove experts."""
        return (
            self.budget is not None
            or self.per_request_budget is not None
            or self.per_gpu_budget is not None
        )


@dataclass
class RoutingResult:
    """Result of :func:`xshare_route`.

    Attributes:
        keep_mask: ``[n_experts]`` bool, the selected set ``S``.
        indices: ``[n_tokens, top_k]`` expert index per token and slot.
        weights: ``[n_tokens, top_k]`` gate weight per token and slot.
        diagnostics: plain numbers and small tensors for logging.
    """

    keep_mask: Tensor
    indices: Tensor
    weights: Tensor
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def _check_logits(router_logits: Tensor) -> None:
    if router_logits.dim() != 2:
        raise ValueError(
            f"router_logits must be [n_tokens, n_experts], got {tuple(router_logits.shape)}"
        )


def _valid_mask(router_logits: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
    n_tokens = router_logits.shape[0]
    if valid_mask is None:
        return torch.ones(n_tokens, dtype=torch.bool, device=router_logits.device)
    if valid_mask.shape != (n_tokens,):
        raise ValueError(f"valid_mask must be [n_tokens], got {tuple(valid_mask.shape)}")
    return valid_mask.to(device=router_logits.device, dtype=torch.bool)


def _ranked(values: Tensor) -> Tensor:
    """Indices that sort ``values`` from high to low, ties to the lower index.

    A stable sort makes the selection deterministic when scores tie.
    """
    return torch.sort(values, dim=-1, descending=True, stable=True).indices


def expert_scores(router_logits: Tensor, score: ScoreKind = "softmax") -> Tensor:
    """Per-token expert scores ``g_{i,j}`` used by the selection proxy."""
    _check_logits(router_logits)
    logits = router_logits.float()
    if score == "softmax":
        return torch.softmax(logits, dim=-1)
    if score == "logits":
        return logits
    raise ValueError(f"unknown score {score!r}")


def aggregate_scores(scores: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
    """Sum scores over the valid tokens: ``sum_i g_{i,j}``, shape ``[n_experts]``."""
    valid = _valid_mask(scores, valid_mask)
    return (scores * valid.unsqueeze(-1).to(scores.dtype)).sum(dim=0)


def warmup_set(scores: Tensor, k0: int, valid_mask: Optional[Tensor] = None) -> Tensor:
    """Union of each valid token's top-``k0`` experts, as a bool mask."""
    n_experts = scores.shape[-1]
    keep = torch.zeros(n_experts, dtype=torch.bool, device=scores.device)
    k0 = min(k0, n_experts)
    if k0 == 0:
        return keep
    valid = _valid_mask(scores, valid_mask)
    # Count votes instead of indexing with the mask: this avoids reading a
    # value back from the GPU, which would stall the decode step.
    top = _ranked(scores)[:, :k0]
    votes = valid.unsqueeze(-1).expand(-1, k0).reshape(-1).to(torch.int32)
    counts = torch.zeros(n_experts, dtype=torch.int32, device=scores.device)
    counts.index_add_(0, top.reshape(-1), votes)
    return counts > 0


def greedy_select(
    aggregated: Tensor,
    budget: int,
    init_mask: Optional[Tensor] = None,
    budget_mode: BudgetMode = "additional",
) -> Tensor:
    """Algorithm 1: add the highest-scoring experts to an initial set.

    The proxy objective is a plain sum over experts, so taking the best
    remaining experts by aggregated score is optimal for a fixed set size.

    Args:
        aggregated: ``[n_experts]`` summed scores.
        budget: number of experts to add (``"additional"``) or the size cap
            of the whole set (``"total"``). In "total" mode the initial set
            is never shrunk, so the result can exceed the cap when the
            initial set is already larger.
        init_mask: the starting set ``S0``. Defaults to the empty set.
    """
    n_experts = aggregated.shape[-1]
    if init_mask is None:
        keep = torch.zeros(n_experts, dtype=torch.bool, device=aggregated.device)
    else:
        keep = init_mask.clone().to(dtype=torch.bool, device=aggregated.device)
    candidates = aggregated.float().masked_fill(keep, float("-inf"))
    if budget_mode == "additional":
        # Experts already in the set sort last. If the budget is larger than
        # what is left, the surplus picks land on them and change nothing.
        # This path never reads a value back from the GPU.
        n_add = max(0, min(budget, n_experts))
    elif budget_mode == "total":
        n_init = int(keep.sum())
        n_add = max(0, min(budget - n_init, n_experts - n_init))
    else:
        raise ValueError(f"unknown budget_mode {budget_mode!r}")
    if n_add == 0:
        return keep
    keep[_ranked(candidates)[:n_add]] = True
    return keep


def _ensure_min_keep(keep: Tensor, aggregated: Tensor, min_keep: int) -> Tensor:
    """Top the set up to ``min_keep`` experts with the best remaining ones."""
    min_keep = min(min_keep, keep.shape[-1])
    if min_keep <= 0:
        return keep
    # Rank kept experts first, then the rest by score, and keep the first
    # ``min_keep``. If the set is already large enough this changes nothing.
    priority = aggregated.float().masked_fill(keep, float("inf"))
    keep = keep.clone()
    keep[_ranked(priority)[:min_keep]] = True
    return keep


# --------------------------------------------------------------------------
# Selection algorithms
# --------------------------------------------------------------------------


def select_batch(
    scores: Tensor,
    budget: int,
    warmup_k0: int = 0,
    valid_mask: Optional[Tensor] = None,
    budget_mode: BudgetMode = "additional",
) -> Tensor:
    """Algorithm 2 without the routing step: warm-up, then global greedy."""
    init = warmup_set(scores, warmup_k0, valid_mask)
    aggregated = aggregate_scores(scores, valid_mask)
    return greedy_select(aggregated, budget, init, budget_mode)


def select_per_request(
    scores: Tensor,
    per_request_budget: int,
    warmup_k0: int = 0,
    budget_mode: BudgetMode = "additional",
) -> Tensor:
    """Algorithm 3: selection for the tokens of one request.

    ``scores`` holds only that request's valid tokens.
    """
    init = warmup_set(scores, warmup_k0)
    return greedy_select(scores.sum(dim=0), per_request_budget, init, budget_mode)


def select_speculative(
    scores: Tensor,
    request_ids: Tensor,
    per_request_budget: int,
    budget: int = 0,
    warmup_k0: int = 0,
    valid_mask: Optional[Tensor] = None,
    budget_mode: BudgetMode = "additional",
) -> Tensor:
    """Algorithm 4: per-request selection, union, then global greedy.

    Args:
        request_ids: ``[n_tokens]`` integer id of the request each token
            belongs to. Under speculative decoding one request contributes
            its last accepted token and its draft tokens.
        budget: the global greedy budget applied after the union.
    """
    n_tokens, n_experts = scores.shape
    if request_ids.shape != (n_tokens,):
        raise ValueError(f"request_ids must be [n_tokens], got {tuple(request_ids.shape)}")
    valid = _valid_mask(scores, valid_mask)
    request_ids = request_ids.to(scores.device)
    keep = torch.zeros(n_experts, dtype=torch.bool, device=scores.device)
    for request in torch.unique(request_ids[valid], sorted=True):
        rows = valid & (request_ids == request)
        keep |= select_per_request(scores[rows], per_request_budget, warmup_k0, budget_mode)
    aggregated = aggregate_scores(scores, valid)
    return greedy_select(aggregated, budget, keep, budget_mode)


def gpu_aware_greedy(
    aggregated: Tensor,
    expert_groups: Tensor,
    per_gpu_budget: int,
    init_mask: Optional[Tensor] = None,
) -> Tensor:
    """Algorithm 5: round-robin greedy over GPU groups.

    Each pass adds, for every group in turn, that group's best unselected
    expert. The loop stops as soon as the set has ``per_gpu_budget *
    n_groups`` experts (the paper's loop finishes the pass; see the paper
    card), or when no group has an expert left. The budget includes the
    initial set.

    This is a plain Python loop that reads values one at a time, written for
    the offline expert-parallel simulation. It is not meant for a GPU decode
    path, where each read would stall the device.

    Starting from the empty set, no group ends up more than one expert ahead
    of another, so the largest group load is at most ``ceil(|S| / n_groups)``
    as long as no group runs out of experts. A non-empty, unbalanced initial
    set (the warm-up in Algorithm 6) is kept as is and can break that bound.

    Args:
        expert_groups: ``[n_experts]`` integer group (GPU) id per expert.
    """
    n_experts = aggregated.shape[-1]
    if expert_groups.shape != (n_experts,):
        raise ValueError(f"expert_groups must be [n_experts], got {tuple(expert_groups.shape)}")
    expert_groups = expert_groups.to(aggregated.device)
    if init_mask is None:
        keep = torch.zeros(n_experts, dtype=torch.bool, device=aggregated.device)
    else:
        keep = init_mask.clone().to(dtype=torch.bool, device=aggregated.device)
    groups = torch.unique(expert_groups, sorted=True).tolist()
    target = min(per_gpu_budget * len(groups), n_experts)
    order = _ranked(aggregated.float()).tolist()
    # Per group, its experts from best to worst.
    queues = {g: [e for e in order if int(expert_groups[e]) == g] for g in groups}
    size = int(keep.sum())
    while size < target:
        added = False
        for g in groups:
            if size >= target:
                break
            queue = queues[g]
            while queue and bool(keep[queue[0]]):
                queue.pop(0)
            if queue:
                keep[queue.pop(0)] = True
                size += 1
                added = True
        if not added:
            break
    return keep


def select_expert_parallel(
    scores: Tensor,
    expert_groups: Tensor,
    per_gpu_budget: int,
    warmup_k0: int = 0,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Algorithm 6 without the routing step: warm-up, then GPU-aware greedy."""
    init = warmup_set(scores, warmup_k0, valid_mask)
    aggregated = aggregate_scores(scores, valid_mask)
    return gpu_aware_greedy(aggregated, expert_groups, per_gpu_budget, init)


# --------------------------------------------------------------------------
# Routing inside the selected set
# --------------------------------------------------------------------------


def route_within(
    router_logits: Tensor,
    keep_mask: Optional[Tensor],
    top_k: int,
    gate_norm: GateNorm = "topk_softmax",
) -> tuple[Tensor, Tensor]:
    """Route each token to its top-k experts inside the kept set.

    With ``keep_mask=None`` or an all-true mask this is the backbone's own
    top-k routing, which is the unpruned baseline.

    Returns:
        ``(indices, weights)``, both ``[n_tokens, top_k]``. If fewer than
        ``top_k`` experts are kept, the surplus slots point at dropped
        experts and get weight zero.
    """
    _check_logits(router_logits)
    n_experts = router_logits.shape[-1]
    if top_k > n_experts:
        raise ValueError("top_k is larger than the number of experts")
    logits = router_logits
    if keep_mask is not None:
        if keep_mask.shape != (n_experts,):
            raise ValueError(f"keep_mask must be [n_experts], got {tuple(keep_mask.shape)}")
        keep_mask = keep_mask.to(device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(~keep_mask, float("-inf"))

    if gate_norm == "topk_softmax":
        values, indices = torch.topk(logits, top_k, dim=-1)
        weights = torch.softmax(values, dim=-1)
        # If a token has no kept expert at all, every value is -inf and the
        # softmax is NaN. Those slots must carry weight zero.
        weights = torch.nan_to_num(weights, nan=0.0)
    elif gate_norm in ("softmax_topk", "softmax_topk_renorm"):
        # Probabilities come from the unmasked logits: dropping an expert
        # must not change the gate weight of the experts that stay.
        probs = torch.softmax(router_logits, dim=-1)
        if keep_mask is not None:
            probs = probs.masked_fill(~keep_mask, 0.0)
        _, indices = torch.topk(logits, top_k, dim=-1)
        weights = torch.gather(probs, -1, indices)
        if gate_norm == "softmax_topk_renorm":
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    else:
        raise ValueError(f"unknown gate_norm {gate_norm!r}")

    if keep_mask is not None:
        # Slots that fell on a dropped expert carry no weight.
        weights = weights * keep_mask[indices].to(weights.dtype)
    return indices, weights


def _diagnostics(
    scores: Tensor,
    keep: Tensor,
    indices: Tensor,
    weights: Tensor,
    valid: Tensor,
    expert_groups: Optional[Tensor],
) -> dict:
    n_experts = scores.shape[-1]
    used = indices[valid][weights[valid] > 0] if bool(valid.any()) else indices.new_empty(0)
    load = torch.bincount(used.reshape(-1), minlength=n_experts)
    aggregated = aggregate_scores(scores, valid)
    total_mass = float(aggregated.sum())
    out = {
        "n_tokens": int(valid.sum()),
        "n_experts": n_experts,
        "kept_experts": int(keep.sum()),
        "activated_experts": int((load > 0).sum()),
        "expert_load": load,
        # Share of the batch's summed score that the kept set covers.
        # Meaningful for the default "softmax" score.
        "score_mass_kept": float(aggregated[keep].sum()) / total_mass if total_mass else 1.0,
    }
    if expert_groups is not None:
        groups = expert_groups.to(keep.device)
        n_groups = int(groups.max()) + 1
        kept_per_group = torch.bincount(groups[keep], minlength=n_groups)
        active_per_group = torch.bincount(groups[load > 0], minlength=n_groups)
        out["kept_per_group"] = kept_per_group
        out["activated_per_group"] = active_per_group
        out["max_kept_per_group"] = int(kept_per_group.max())
        out["max_activated_per_group"] = int(active_per_group.max())
    return out


def xshare_route(
    router_logits: Tensor,
    config: XShareConfig,
    valid_mask: Optional[Tensor] = None,
    request_ids: Optional[Tensor] = None,
    expert_groups: Optional[Tensor] = None,
    with_diagnostics: bool = True,
) -> RoutingResult:
    """Select the batch's expert set and route every token inside it.

    Which algorithm runs depends on the configuration:

    * ``per_gpu_budget`` set       -> Algorithm 6 (needs ``expert_groups``)
    * ``per_request_budget`` set   -> Algorithm 4 (needs ``request_ids``)
    * ``budget`` set               -> Algorithm 2
    * none of them set             -> no pruning (baseline routing)

    Args:
        router_logits: ``[n_tokens, n_experts]`` raw router output for one
            MoE layer at one decode step.
        valid_mask: ``[n_tokens]`` bool. Tokens marked false (padding) take
            no part in the selection. They are still routed, so the output
            shapes match the input.
        request_ids: ``[n_tokens]`` request id per token (Algorithm 4).
        expert_groups: ``[n_experts]`` GPU group per expert (Algorithms 5-6,
            and per-group diagnostics for any algorithm).
    """
    _check_logits(router_logits)
    n_experts = router_logits.shape[-1]
    valid = _valid_mask(router_logits, valid_mask)
    scores = expert_scores(router_logits, config.score)

    if not config.prunes:
        keep = torch.ones(n_experts, dtype=torch.bool, device=router_logits.device)
    else:
        if config.per_gpu_budget is not None:
            if expert_groups is None:
                raise ValueError("per_gpu_budget needs expert_groups")
            keep = select_expert_parallel(
                scores, expert_groups, config.per_gpu_budget, config.warmup_k0, valid
            )
        elif config.per_request_budget is not None:
            if request_ids is None:
                raise ValueError("per_request_budget needs request_ids")
            keep = select_speculative(
                scores,
                request_ids,
                config.per_request_budget,
                config.budget or 0,
                config.warmup_k0,
                valid,
                config.budget_mode,
            )
        else:
            keep = select_batch(
                scores, config.budget or 0, config.warmup_k0, valid, config.budget_mode
            )
        min_keep = config.top_k if config.min_keep is None else config.min_keep
        keep = _ensure_min_keep(keep, aggregate_scores(scores, valid), min_keep)
        # A batch with no valid token has nothing to select from: keep all.
        keep = keep | ~valid.any()

    indices, weights = route_within(router_logits, keep, config.top_k, config.gate_norm)
    diagnostics = (
        _diagnostics(scores, keep, indices, weights, valid, expert_groups)
        if with_diagnostics
        else {}
    )
    return RoutingResult(keep, indices, weights, diagnostics)
