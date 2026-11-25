"""Chunked nearest-codeword search, weighted group assignment, and index encoding.

Notation (all in normalised space, ``u = W / scale``):

* ``codebooks``: float tensor ``[2, 16]``; table ``c`` is ``codebooks[c]``.
* ``d_c[o, k] = min_j (u[o, k] - codebooks[c, j])^2``: squared distance to the
  nearest entry of table ``c``.
* Group error of selection group ``(o, s)`` for table ``c``:
  ``E_c[o, s] = sum_{k in group s} I[k] * d_c[o, k]``.

Tie-breaking (fixed so every run and every member gets identical bits):

* nearest codeword: the **lowest index** among equally close entries;
* table selection: table 1 only if ``E_1 < E_0`` strictly, otherwise table 0.

Nearest-neighbour work is done on blocks of whole rows so the temporary
``[rows, in, 16]`` tensor never exceeds ``chunk_elements``. Rows are
independent, so the results are bit-identical for any chunk size.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from .config import NUM_CODEWORDS, NUM_TABLES


def nearest_codeword(values: torch.Tensor, table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest entry of a 1-D ``table`` for every element of ``values`` (no chunking).

    Returns ``(index, squared_distance)`` with the shape of ``values``; ``index``
    is int64 and ties go to the lowest index (``torch.argmin`` returns the first
    minimum).
    """
    table = table.to(device=values.device, dtype=torch.float32)
    sq = (values.to(torch.float32).unsqueeze(-1) - table).square()
    index = sq.argmin(dim=-1)
    return index, sq.gather(-1, index.unsqueeze(-1)).squeeze(-1)


def row_chunks(out_features: int, in_features: int, chunk_elements: int) -> Iterator[slice]:
    """Yield row slices whose ``[rows, in, 16]`` distance tensor fits ``chunk_elements``."""
    rows = max(1, chunk_elements // (in_features * NUM_CODEWORDS))
    for start in range(0, out_features, rows):
        yield slice(start, min(start + rows, out_features))


def assign_selection_groups(
    normalized: torch.Tensor,
    importance: torch.Tensor,
    codebooks: torch.Tensor,
    selection_size: int,
    *,
    chunk_elements: int = 1 << 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick, for every selection group, the table with the lower importance-weighted error.

    Returns ``(selectors, group_error)``:

    * ``selectors``: bool ``[out, in // S]``; ``False`` = table 0, ``True`` = table 1.
    * ``group_error``: float32 ``[out, in // S]``, the error of the chosen table,
      i.e. ``min(E_0, E_1)``. ``group_error.sum()`` is the fit objective.
    """
    out_features, in_features = _check(normalized, importance, codebooks, selection_size)
    num_sel = in_features // selection_size
    weight = importance.to(device=normalized.device, dtype=torch.float32)
    tables = codebooks.to(device=normalized.device, dtype=torch.float32)

    selectors = torch.empty(out_features, num_sel, dtype=torch.bool, device=normalized.device)
    group_error = torch.empty(out_features, num_sel, dtype=torch.float32, device=normalized.device)
    for rows in row_chunks(out_features, in_features, chunk_elements):
        block = normalized[rows].to(torch.float32)
        errors = []
        for c in range(NUM_TABLES):
            _, dist = nearest_codeword(block, tables[c])
            errors.append((dist * weight).reshape(-1, num_sel, selection_size).sum(dim=-1))
        use_table_1 = errors[1] < errors[0]
        selectors[rows] = use_table_1
        group_error[rows] = torch.where(use_table_1, errors[1], errors[0])
    return selectors, group_error


def encode_indices(
    normalized: torch.Tensor,
    codebooks: torch.Tensor,
    selectors: torch.Tensor,
    selection_size: int,
    *,
    chunk_elements: int = 1 << 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-codeword index of every weight inside the table chosen for its group.

    Returns ``(indices, squared_distance)``: uint8 ``[out, in]`` with values in
    ``[0, 16)`` and float32 ``[out, in]`` (unweighted ``(u - table[idx])^2``).
    Members fitting the codebooks can reuse this as the "assign each element to
    a cell" step of a Lloyd iteration.
    """
    if normalized.ndim != 2 or normalized.numel() == 0:
        raise ValueError("normalized must be a non-empty [out, in] tensor")
    out_features, in_features = normalized.shape
    if selection_size < 1 or in_features % selection_size != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by selection_size ({selection_size})")
    _check_codebooks(codebooks)
    num_sel = in_features // selection_size
    if selectors.shape != (out_features, num_sel) or selectors.dtype != torch.bool:
        raise ValueError(f"selectors must be bool with shape {(out_features, num_sel)}")
    tables = codebooks.to(device=normalized.device, dtype=torch.float32)

    indices = torch.empty(out_features, in_features, dtype=torch.uint8, device=normalized.device)
    sq_dist = torch.empty(out_features, in_features, dtype=torch.float32, device=normalized.device)
    for rows in row_chunks(out_features, in_features, chunk_elements):
        block = normalized[rows].to(torch.float32)
        idx0, dist0 = nearest_codeword(block, tables[0])
        idx1, dist1 = nearest_codeword(block, tables[1])
        use_table_1 = selectors[rows].to(normalized.device).repeat_interleave(selection_size, dim=1)
        indices[rows] = torch.where(use_table_1, idx1, idx0).to(torch.uint8)
        sq_dist[rows] = torch.where(use_table_1, dist1, dist0)
    return indices, sq_dist


def _check_codebooks(codebooks: torch.Tensor) -> None:
    if codebooks.shape != (NUM_TABLES, NUM_CODEWORDS):
        raise ValueError(f"codebooks shape {tuple(codebooks.shape)} != {(NUM_TABLES, NUM_CODEWORDS)}")
    if not codebooks.is_floating_point() or not torch.isfinite(codebooks).all():
        raise ValueError("codebooks must be finite floating point")


def _check(normalized, importance, codebooks, selection_size) -> tuple[int, int]:
    if normalized.ndim != 2 or normalized.numel() == 0:
        raise ValueError("normalized must be a non-empty [out, in] tensor")
    out_features, in_features = normalized.shape
    if selection_size < 1 or in_features % selection_size != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by selection_size ({selection_size})")
    if importance.shape != (in_features,):
        raise ValueError(f"importance shape {tuple(importance.shape)} != ({in_features},)")
    if not torch.isfinite(importance).all() or (importance < 0).any():
        raise ValueError("importance must be finite and non-negative")
    _check_codebooks(codebooks)
    return out_features, in_features
