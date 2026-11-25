"""Hyper-parameters and fixed format constants for the AAAC fit."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Fixed by the 4-bit format: every weight stores a 4-bit index into one of two tables.
NUM_TABLES = 2
NUM_CODEWORDS = 16


@dataclass(frozen=True)
class AAACConfig:
    """Settings for :func:`fit_aaac_layer`.

    group_size      g: contiguous weights in one row that share one scale.
    selection_size  S: contiguous weights in one row that share one table choice.
                    Must divide ``group_size`` (the paper settings are S = g and S = 16).
    outer_iters     number of (group assignment -> inner Lloyd loop) rounds (paper: 3).
    inner_iters     weighted Lloyd updates per outer round (paper: 10).
    scale_dtype     storage dtype of the per-group scales. Normalisation uses the
                    *rounded* scale so that encode and decode agree exactly.
    codebook_dtype  storage dtype of the two final tables (paper: BF16).
    chunk_elements  upper bound on the size of the temporary ``[rows, in, 16]``
                    distance tensor used by nearest-neighbour search. Results do
                    not depend on it; it only trades speed for peak memory.
    """

    group_size: int = 128
    selection_size: int = 128
    outer_iters: int = 3
    inner_iters: int = 10
    scale_dtype: torch.dtype = torch.float16
    codebook_dtype: torch.dtype = torch.bfloat16
    chunk_elements: int = 1 << 24

    def __post_init__(self) -> None:
        for name in ("group_size", "selection_size", "chunk_elements"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        for name in ("outer_iters", "inner_iters"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if self.group_size % self.selection_size != 0:
            raise ValueError(
                f"selection_size ({self.selection_size}) must divide group_size ({self.group_size})"
            )
        for name in ("scale_dtype", "codebook_dtype"):
            dtype = getattr(self, name)
            if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
                raise ValueError(f"{name} must be a floating torch.dtype, got {dtype!r}")
