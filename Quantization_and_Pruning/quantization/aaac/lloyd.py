"""Weighted Lloyd updates and BF16 finalisation (Member 2, part B).

These two functions complete Algorithm 1 of the AAAC paper:

* :func:`weighted_lloyd_update` is one inner iteration (lines 13-21): every
  weight is assigned to the nearest entry of the table chosen for its selection
  group, and each entry moves to the importance-weighted centroid of its
  Voronoi cell (Eq. 11). Entries whose cell is empty keep their old value.
* :func:`finalize_codebooks` is lines 22 and 25: sort each table ascending and
  round it to BF16. ``fit_aaac_layer`` then re-runs group assignment and index
  encoding on the rounded tables (lines 26-27).

Determinism: the per-cell sums are computed with a stable sort plus a float64
cumulative sum. There are no atomic scatter operations, so the result is
bit-identical across runs on CPU and CUDA, and it does not depend on
``chunk_elements`` (which only affects the nearest-codeword search).
"""

from __future__ import annotations

import torch

from .assignment import encode_indices
from .config import NUM_CODEWORDS, NUM_TABLES


def weighted_lloyd_update(
    normalized: torch.Tensor,
    importance: torch.Tensor,
    codebooks: torch.Tensor,
    selectors: torch.Tensor,
    selection_size: int,
    *,
    chunk_elements: int,
) -> torch.Tensor:
    """One inner iteration of Algorithm 1 (weighted Lloyd step).

    Args:
        normalized:  float32 ``[out, in]`` normalised weights ``u = W / scale``.
        importance:  float32 ``[in]`` non-negative column importance ``I[k]``
                     (same weight for every output row).
        codebooks:   float32 ``[2, 16]`` current tables. Not modified.
        selectors:   bool ``[out, in // S]``; ``True`` means the group uses table 1.
        selection_size: S.
        chunk_elements: memory bound passed on to ``encode_indices``.

    Returns a NEW float32 ``[2, 16]`` tensor (on ``normalized.device``) where,
    for every table ``c`` and entry ``j``::

        cell(c, j) = { (o, k) : group of (o, k) uses table c and
                                encode_indices(...)[o, k] == j }
        new[c, j]  = sum_{cell} I[k] * u[o, k] / sum_{cell} I[k]   if sum_{cell} I[k] > 0
        new[c, j]  = codebooks[c, j]                                otherwise (keep)

    Sums are accumulated in float64 and the result is deterministic.
    """
    if normalized.ndim != 2 or normalized.numel() == 0:
        raise ValueError("normalized must be a non-empty [out, in] tensor")
    out_features, in_features = normalized.shape
    if importance.shape != (in_features,):
        raise ValueError(f"importance shape {tuple(importance.shape)} != ({in_features},)")
    if not torch.isfinite(importance).all() or (importance < 0).any():
        raise ValueError("importance must be finite and non-negative")

    device = normalized.device
    tables = codebooks.to(device=device, dtype=torch.float32)

    # Step 1: assign every weight to a cell (table, index) with the same
    # nearest-codeword rule (lowest index on ties) the fit uses everywhere.
    indices, _ = encode_indices(
        normalized, tables, selectors, selection_size, chunk_elements=chunk_elements
    )
    weight_table = selectors.to(device).repeat_interleave(selection_size, dim=1)
    cell = (weight_table.to(torch.int64) * NUM_CODEWORDS + indices.to(torch.int64)).reshape(-1)

    # Step 2: weighted sums per cell, in float64.
    imp = importance.to(device=device, dtype=torch.float64)
    imp_all = imp.unsqueeze(0).expand(out_features, in_features).reshape(-1)
    num_all = imp_all * normalized.to(torch.float64).reshape(-1)

    denominator = _segment_sums(cell, imp_all)
    numerator = _segment_sums(cell, num_all)

    # Step 3: empty cells (no members, or zero total importance) keep their value.
    keep = denominator <= 0
    centroid = numerator / torch.where(keep, torch.ones_like(denominator), denominator)
    new = torch.where(keep, tables.reshape(-1).to(torch.float64), centroid)
    return new.reshape(NUM_TABLES, NUM_CODEWORDS).to(torch.float32)


def finalize_codebooks(
    codebooks: torch.Tensor,
    codebook_dtype: torch.dtype = torch.bfloat16,
    *,
    sort: bool = True,
) -> torch.Tensor:
    """Sort each fitted table ascending and round it to ``codebook_dtype`` (BF16).

    Returns a new ``[2, 16]`` tensor of dtype ``codebook_dtype``. Sorting only
    changes the order of entries (Algorithm 1, line 22); it is a no-op for tables
    that are already ascending, such as the output of the Lloyd updates, and
    rounding is monotone so sorting before or after it gives the same values.
    ``fit_aaac_layer`` re-runs group assignment and index encoding with these
    rounded tables afterwards, so the stored indices always match the stored
    tables.
    """
    if codebooks.shape != (NUM_TABLES, NUM_CODEWORDS):
        raise ValueError(f"codebooks shape {tuple(codebooks.shape)} != {(NUM_TABLES, NUM_CODEWORDS)}")
    if not codebooks.is_floating_point() or not torch.isfinite(codebooks).all():
        raise ValueError("codebooks must be finite floating point")
    tables = codebooks.detach().to(torch.float32)
    if sort:
        tables = tables.sort(dim=-1).values
    return tables.to(codebook_dtype).clone()


def _segment_sums(cell: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Deterministic ``out[c] = sum(values[cell == c])`` for ``c`` in ``0..31`` (float64).

    A stable sort by cell followed by a cumulative sum avoids atomic scatter
    operations (which are non-deterministic on CUDA). Empty cells give exactly 0.
    """
    num_cells = NUM_TABLES * NUM_CODEWORDS
    sorted_cell, order = torch.sort(cell, stable=True)
    csum = torch.cat([values.new_zeros(1), torch.cumsum(values[order], dim=0)])
    bounds = torch.searchsorted(
        sorted_cell, torch.arange(num_cells + 1, device=cell.device, dtype=cell.dtype)
    )
    return csum[bounds[1:]] - csum[bounds[:-1]]
