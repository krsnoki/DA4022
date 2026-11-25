"""Deterministic AAAC fit for one linear layer (Algorithm 1) and a whole-model driver.

Pipeline for a weight ``W[out, in]`` and column importance ``I[in]``::

    scales  = absmax of each contiguous g-group, rounded, strictly positive   (scaling.py)
    U       = W / scales                                                      (scaling.py)
    C       = two quantile-initialised 16-entry tables, float32 [2, 16]       (codebook_init.py)
    repeat outer_iters times:
        selectors = per S-group table with lower I-weighted error             (assignment.py)
        repeat inner_iters times:
            C = weighted Lloyd update (empty cells keep their value)          (lloyd.py, Member 2B)
    C = round tables to BF16                                                  (lloyd.py, Member 2B)
    selectors = re-assign with the rounded tables                             (assignment.py)
    indices   = nearest entry in the selected table                           (assignment.py)

Reference decode: ``W_hat[o, k] = scale[o, k // g] * C[selector[o, k // S], indices[o, k]]``.

The fit uses no random numbers and only deterministic tensor ops, so the same
inputs give bit-identical outputs (on a given device and PyTorch build).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch
from torch import nn

from . import lloyd
from .assignment import assign_selection_groups, encode_indices
from .codebook_init import initial_codebooks
from .config import NUM_CODEWORDS, NUM_TABLES, AAACConfig
from .scaling import compute_group_scales, expand_group_scales, normalize_weight

UpdateFn = Callable[..., torch.Tensor]
FinalizeFn = Callable[[torch.Tensor, torch.dtype], torch.Tensor]


@dataclass
class AAACLayerFit:
    """Everything needed to rebuild one quantised layer (handed to Member 3's codec).

    scales      ``[out, in // g]`` in ``config.scale_dtype``, strictly positive.
    codebooks   ``[2, 16]`` in ``config.codebook_dtype`` (BF16): table 0 and table 1.
    selectors   bool ``[out, in // S]``; ``True`` = the group uses table 1.
    indices     uint8 ``[out, in]``, values in ``[0, 16)``.
    objective_history  importance-weighted squared error in normalised space,
                ``sum_{o,k} I[k] (u[o,k] - C[sel][idx])^2``, recorded at every group
                assignment: one value per outer iteration, then the final value
                obtained with the BF16 tables.
    """

    scales: torch.Tensor
    codebooks: torch.Tensor
    selectors: torch.Tensor
    indices: torch.Tensor
    group_size: int
    selection_size: int
    objective_history: list[float] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.indices.shape)

    def to(self, device: str | torch.device) -> "AAACLayerFit":
        return AAACLayerFit(
            scales=self.scales.to(device),
            codebooks=self.codebooks.to(device),
            selectors=self.selectors.to(device),
            indices=self.indices.to(device),
            group_size=self.group_size,
            selection_size=self.selection_size,
            objective_history=list(self.objective_history),
        )

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dense reference reconstruction ``[out, in]``. The packed codec must match this."""
        table_ids = self.selectors.to(torch.long).repeat_interleave(self.selection_size, dim=1)
        flat = self.codebooks.to(torch.float32).reshape(-1)
        values = flat[table_ids * NUM_CODEWORDS + self.indices.to(torch.long)]
        scale = expand_group_scales(self.scales, self.group_size).to(values.device)
        return (values * scale).to(dtype)


def fit_aaac_layer(
    weight: torch.Tensor,
    importance: torch.Tensor,
    config: AAACConfig | None = None,
    *,
    update_fn: UpdateFn | None = None,
    finalize_fn: FinalizeFn | None = None,
) -> AAACLayerFit:
    """Fit AAAC tables, selectors and indices for one ``[out, in]`` weight.

    Args:
        weight:      ``[out, in]`` floating tensor (e.g. ``QuantizationTarget.weight``).
                     It is read only, never modified.
        importance:  ``[in]`` non-negative column energy ``I[k] = sum_t X[t, k]^2``.
                     If it is all zero the fit falls back to uniform importance.
        config:      :class:`AAACConfig`; defaults to g = S = 128, 3 x 10 iterations.
        update_fn:   override for the weighted Lloyd step (defaults to
                     ``lloyd.weighted_lloyd_update``). Mainly for tests.
        finalize_fn: override for the BF16 rounding (defaults to
                     ``lloyd.finalize_codebooks``). Mainly for tests.

    All work happens on ``weight.device``; outputs stay on that device.
    """
    cfg = config or AAACConfig()
    update = update_fn or lloyd.weighted_lloyd_update
    finalize = finalize_fn or lloyd.finalize_codebooks

    with torch.no_grad():
        importance = _prepare_importance(weight, importance)
        g, s = cfg.group_size, cfg.selection_size
        if weight.ndim == 2 and weight.shape[1] % s != 0:
            raise ValueError(f"in_features ({weight.shape[1]}) must be divisible by selection_size ({s})")

        scales = compute_group_scales(weight, g, cfg.scale_dtype)
        normalized = normalize_weight(weight, scales, g)
        codebooks = initial_codebooks(normalized, importance, s)

        history: list[float] = []
        for _ in range(cfg.outer_iters):
            selectors, group_error = assign_selection_groups(
                normalized, importance, codebooks, s, chunk_elements=cfg.chunk_elements
            )
            history.append(_objective(group_error))
            for _ in range(cfg.inner_iters):
                codebooks = update(
                    normalized, importance, codebooks, selectors, s,
                    chunk_elements=cfg.chunk_elements,
                )
                _check_tables(codebooks, torch.float32, "update_fn")

        final_tables = finalize(codebooks, cfg.codebook_dtype)
        _check_tables(final_tables, cfg.codebook_dtype, "finalize_fn")
        final_tables = final_tables.to(normalized.device)

        # Re-run assignment and encoding with the *rounded* tables so the stored
        # selectors/indices are exactly the best choice for what is stored.
        selectors, group_error = assign_selection_groups(
            normalized, importance, final_tables, s, chunk_elements=cfg.chunk_elements
        )
        history.append(_objective(group_error))
        indices, _ = encode_indices(
            normalized, final_tables, selectors, s, chunk_elements=cfg.chunk_elements
        )

    return AAACLayerFit(
        scales=scales,
        codebooks=final_tables,
        selectors=selectors,
        indices=indices,
        group_size=g,
        selection_size=s,
        objective_history=history,
    )


def fit_aaac_model(
    model: nn.Module,
    column_energy: Mapping[str, torch.Tensor],
    config: AAACConfig | None = None,
    *,
    device: str | torch.device = "cpu",
    update_fn: UpdateFn | None = None,
    finalize_fn: FinalizeFn | None = None,
) -> dict[str, AAACLayerFit]:
    """Fit every eligible layer (Member 1's layer set) and return ``{name: fit}`` on CPU.

    The model is not modified. Layers are processed one at a time, so only one
    FP32 weight copy lives on ``device`` at once.
    """
    from Dataloaders.calibration import iter_quantization_targets

    fits: dict[str, AAACLayerFit] = {}
    for target in iter_quantization_targets(model, column_energy, device=device):
        fit = fit_aaac_layer(
            target.weight, target.importance, config,
            update_fn=update_fn, finalize_fn=finalize_fn,
        )
        fits[target.name] = fit.to("cpu")
    return fits


def _prepare_importance(weight: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("weight must be a 2-D tensor with layout [out_features, in_features]")
    if not isinstance(importance, torch.Tensor) or importance.shape != (weight.shape[1],):
        shape = tuple(importance.shape) if isinstance(importance, torch.Tensor) else None
        raise ValueError(f"importance shape {shape} != ({weight.shape[1]},)")
    importance = importance.detach().to(device=weight.device, dtype=torch.float32)
    if not torch.isfinite(importance).all():
        raise ValueError("importance contains non-finite values")
    if (importance < 0).any():
        raise ValueError("importance must be non-negative")
    if not (importance.sum() > 0):
        importance = torch.ones_like(importance)
    return importance


def _objective(group_error: torch.Tensor) -> float:
    return float(group_error.to(torch.float64).sum().item())


def _check_tables(tables: torch.Tensor, dtype: torch.dtype, source: str) -> None:
    if not isinstance(tables, torch.Tensor) or tables.shape != (NUM_TABLES, NUM_CODEWORDS):
        shape = tuple(tables.shape) if isinstance(tables, torch.Tensor) else type(tables).__name__
        raise ValueError(f"{source} must return a [{NUM_TABLES}, {NUM_CODEWORDS}] tensor, got {shape}")
    if tables.dtype != dtype:
        raise ValueError(f"{source} must return dtype {dtype}, got {tables.dtype}")
    if not torch.isfinite(tables).all():
        raise ValueError(f"{source} returned non-finite table entries")
