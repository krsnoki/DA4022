"""Expert scoring and diversity-aware greedy log-determinant selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .calibration import CalibrationStats


def _safe_normalize(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.float64).clamp_min(0)
    total = values.sum()
    return values / total if total > 0 else torch.full_like(values, 1.0 / values.numel())


def importance_weighted_kernel(importance: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    importance = _safe_normalize(importance)
    root = torch.sqrt(importance)
    gram = gram.to(torch.float64)
    return root[:, None] * gram * root[None, :]


def greedy_logdet_select(kernel: torch.Tensor, count: int, lambda_reg: float | None = None) -> list[int]:
    """Greedily maximize ``log det(K_S + lambda I)`` with stable tie-breaking."""
    experts = kernel.shape[0]
    if kernel.shape != (experts, experts):
        raise ValueError("kernel must be square")
    if not 1 <= count <= experts:
        raise ValueError("count must be between one and number of experts")
    if lambda_reg is None:
        # Paper Eq. 4: lambda = (1 / (K E)) sum_e K_ee.
        lambda_reg = float(torch.diag(kernel).sum()) / (count * experts)
    lambda_reg = max(float(lambda_reg), 1e-12)
    selected: list[int] = []
    remaining = list(range(experts))
    for _ in range(count):
        best_index, best_value = -1, float("-inf")
        for candidate in remaining:
            subset = selected + [candidate]
            matrix = kernel[subset][:, subset] + lambda_reg * torch.eye(len(subset), dtype=kernel.dtype)
            sign, logabsdet = torch.linalg.slogdet(matrix)
            value = float(logabsdet) if sign > 0 else float("-inf")
            if value > best_value + 1e-12:
                best_index, best_value = candidate, value
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


@dataclass(frozen=True)
class ExpertScorer:
    stats: CalibrationStats

    def sf(self) -> torch.Tensor:
        return _safe_normalize(self.stats.selection_count / max(self.stats.token_count, 1))

    def cp(self) -> torch.Tensor:
        denom = self.stats.selection_count.clamp_min(1)
        return _safe_normalize(self.stats.selected_probability_sum / denom)

    def acp(self) -> torch.Tensor:
        # Paper Eq. 3 extension: CP × sqrt(E_t[||f_e(t)||²]).
        rms_activation = torch.sqrt(torch.diag(self.stats.output_gram).clamp_min(0) / max(self.stats.token_count, 1))
        return _safe_normalize(self.cp() * rms_activation)

    def scores(self, method: str) -> torch.Tensor:
        method = method.lower().replace("-", "_")
        if method in {"sf", "selection_frequency"}:
            return self.sf()
        if method in {"cp", "conditional_probability"}:
            return self.cp()
        if method in {"acp", "activation_weighted_conditional_probability"}:
            return self.acp()
        if method in {"do_cp", "do_acp"}:
            return self.cp() if method == "do_cp" else self.acp()
        raise ValueError(f"unknown scoring method: {method}")

    def select(self, method: str, count: int, lambda_reg: float | None = None) -> list[int]:
        method = method.lower().replace("-", "_")
        importance = self.scores(method)
        if method in {"do_cp", "do_acp"}:
            gram_expectation = self.stats.output_gram / max(self.stats.token_count, 1)
            kernel = importance_weighted_kernel(importance, gram_expectation)
            return greedy_logdet_select(kernel, count, lambda_reg)
        # Stable ordering: descending score, then ascending expert index.
        return sorted(range(importance.numel()), key=lambda i: (-float(importance[i]), i))[:count]
