"""Quantile initialisation of the two 16-entry AAAC tables.

Everything here works on *normalised* weights ``u = W / scale`` (see
``scaling.py``) and on per-element importance ``I[k]`` (the calibration column
energy, broadcast over output rows).

1. ``weighted_quantiles``: inverse-CDF quantiles of a weighted 1-D sample.
2. ``quantile_codebook``: 16 quantiles at the bin mid-points (j + 0.5) / 16, so
   every codeword starts with the same share of importance mass.
3. ``initial_codebooks``: builds the two tables. Selection groups are ranked by
   their importance-weighted mean ``|u|``; the lower half ("peaky" groups whose
   mass sits near 0 because an outlier sets the absmax) initialises table 0, the
   upper half ("flat" groups) initialises table 1. Ties are broken by the group's
   flat position (row-major), so the result is fully deterministic. If one half
   carries zero total importance, its table is seeded from the whole layer.

No random numbers are used anywhere.
"""

from __future__ import annotations

import torch

from .config import NUM_CODEWORDS


def weighted_quantiles(
    values: torch.Tensor,
    weights: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    """Return the weighted inverse-CDF quantiles of ``values`` at ``probs`` (float32).

    For each ``p`` the result is the smallest sorted value ``v[i]`` whose
    cumulative weight ``sum(w[:i+1])`` is ``>= p * sum(w)``. The output is
    always an element of ``values`` (no interpolation), and non-decreasing when
    ``probs`` is. Elements with zero weight are never selected for ``p > 0``.
    If every weight is zero, uniform weights are used instead.
    """
    values = values.detach().reshape(-1)
    weights = weights.detach().reshape(-1)
    probs = probs.detach().reshape(-1)
    if values.numel() == 0:
        raise ValueError("values must not be empty")
    if weights.shape != values.shape:
        raise ValueError(f"weights shape {tuple(weights.shape)} != values shape {tuple(values.shape)}")
    if not torch.isfinite(values).all() or not torch.isfinite(weights).all():
        raise ValueError("values and weights must be finite")
    if (weights < 0).any():
        raise ValueError("weights must be non-negative")
    if not torch.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("probs must lie in [0, 1]")

    # float64 cumulative sums keep the CDF accurate over millions of elements.
    weights = weights.to(dtype=torch.float64, device=values.device)
    if not (weights.sum() > 0):
        weights = torch.ones_like(weights)

    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    cdf = torch.cumsum(weights[order], dim=0)
    targets = probs.to(dtype=torch.float64, device=values.device) * cdf[-1]
    positions = torch.searchsorted(cdf, targets, right=False)
    positions = positions.clamp_(max=values.numel() - 1)  # guards p == 1 against rounding
    return sorted_values[positions].to(torch.float32)


def quantile_codebook(
    values: torch.Tensor,
    weights: torch.Tensor,
    num_codewords: int = NUM_CODEWORDS,
) -> torch.Tensor:
    """Return ``num_codewords`` sorted weighted quantiles at mid-points ``(j + 0.5) / n``."""
    if num_codewords < 1:
        raise ValueError("num_codewords must be positive")
    probs = (torch.arange(num_codewords, dtype=torch.float64) + 0.5) / num_codewords
    return weighted_quantiles(values, weights, probs.to(values.device))


def selection_group_spread(
    normalized: torch.Tensor,
    importance: torch.Tensor,
    selection_size: int,
) -> torch.Tensor:
    """Importance-weighted mean ``|u|`` of every selection group, float32 ``[out, in // S]``.

    Groups whose importance sums to zero fall back to the unweighted mean.
    """
    out_features, in_features = _check_inputs(normalized, importance, selection_size)
    num_sel = in_features // selection_size
    magnitude = normalized.to(torch.float32).abs().reshape(out_features, num_sel, selection_size)
    weight = importance.to(device=normalized.device, dtype=torch.float32).reshape(1, num_sel, selection_size)
    weight_sum = weight.sum(dim=-1)  # [1, num_sel]
    weighted = (magnitude * weight).sum(dim=-1) / weight_sum.clamp(min=torch.finfo(torch.float32).tiny)
    return torch.where(weight_sum > 0, weighted, magnitude.mean(dim=-1))


def initial_codebooks(
    normalized: torch.Tensor,
    importance: torch.Tensor,
    selection_size: int,
    num_codewords: int = NUM_CODEWORDS,
) -> torch.Tensor:
    """Build the two starting tables, float32 ``[2, num_codewords]``, each sorted ascending.

    ``normalized`` is ``[out, in]``; ``importance`` is ``[in]`` and is broadcast
    over rows. With a single selection group both tables are identical.
    """
    out_features, in_features = _check_inputs(normalized, importance, selection_size)
    num_sel = in_features // selection_size
    num_groups = out_features * num_sel

    spread = selection_group_spread(normalized, importance, selection_size).reshape(-1)
    order = torch.argsort(spread, stable=True)
    half = num_groups // 2
    low, high = order[:half], order[half:]
    if half == 0:  # one selection group in total
        low = high

    group_values = normalized.to(torch.float32).reshape(num_groups, selection_size)
    group_weights = (
        importance.to(device=normalized.device, dtype=torch.float32)
        .reshape(1, num_sel, selection_size)
        .expand(out_features, num_sel, selection_size)
        .reshape(num_groups, selection_size)
    )
    all_groups = torch.arange(num_groups, device=normalized.device)
    tables = []
    for members in (low, high):
        # A half whose importance is all zero cannot affect the objective; seed its
        # table from the whole layer instead of from unweighted, unimportant values.
        if not (group_weights[members].sum() > 0):
            members = all_groups
        tables.append(quantile_codebook(group_values[members], group_weights[members], num_codewords))
    return torch.stack(tables)


def _check_inputs(normalized: torch.Tensor, importance: torch.Tensor, selection_size: int) -> tuple[int, int]:
    if normalized.ndim != 2:
        raise ValueError("normalized must be 2-D [out, in]")
    out_features, in_features = normalized.shape
    if out_features == 0 or in_features == 0:
        raise ValueError("normalized must not be empty")
    if selection_size < 1 or in_features % selection_size != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by selection_size ({selection_size})")
    if importance.shape != (in_features,):
        raise ValueError(f"importance shape {tuple(importance.shape)} != ({in_features},)")
    if not torch.isfinite(normalized).all() or not torch.isfinite(importance).all():
        raise ValueError("normalized and importance must be finite")
    if (importance < 0).any():
        raise ValueError("importance must be non-negative")
    return out_features, in_features
