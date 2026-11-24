"""Unmodified baselines for the cascade experiments (no confidence signal).

* ``always_model``  - every query answered by one fixed model (always-small,
  always-large, or any single model). This is also the S_1 policy class.
* ``random_router`` - escalate a uniformly random fraction t of queries from
  the cheap model L to the expensive model H. Its expected accuracy is the
  straight line abar(t) = (1 - t) a(0) + t a(1) used in Prop. 3.2, and its
  cost is c_L + t c_H. ``random_router_at_budget`` picks t to spend a budget.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def always_model(U: np.ndarray, C: np.ndarray, model: int, idx: Optional[np.ndarray] = None,
                 latency: Optional[np.ndarray] = None) -> Dict[str, float]:
    U = np.asarray(U, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if idx is None:
        idx = np.arange(U.shape[0])
    out = {"accuracy": float(U[idx, model].mean()), "mean_cost": float(C[idx, model].mean()),
           "escalation_rate": 0.0, "mean_models_called": 1.0}
    if latency is not None:
        lat = np.asarray(latency, dtype=np.float64)[idx, model]
        out.update(latency_mean=float(lat.mean()), latency_median=float(np.median(lat)),
                   latency_p95=float(np.percentile(lat, 95)))
    return out


def random_router(U: np.ndarray, C: np.ndarray, small: int, large: int, frac: float,
                  idx: Optional[np.ndarray] = None, seed: int = 0,
                  latency: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Realised (sampled) random escalation of a fraction ``frac`` of queries.

    The cascade runs the small model first, so escalated queries pay both costs.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError("frac must be in [0, 1]")
    U = np.asarray(U, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if idx is None:
        idx = np.arange(U.shape[0])
    rng = np.random.default_rng(seed)
    n = len(idx)
    esc = np.zeros(n, dtype=bool)
    esc[rng.permutation(n)[: int(round(frac * n))]] = True
    q = np.where(esc, U[idx, large], U[idx, small])
    c = C[idx, small] + esc * C[idx, large]
    out = {"accuracy": float(q.mean()), "mean_cost": float(c.mean()),
           "escalation_rate": float(esc.mean()), "mean_models_called": float(1 + esc.mean())}
    if latency is not None:
        L = np.asarray(latency, dtype=np.float64)
        lat = L[idx, small] + esc * L[idx, large]
        out.update(latency_mean=float(lat.mean()), latency_median=float(np.median(lat)),
                   latency_p95=float(np.percentile(lat, 95)))
    return out


def random_router_expected(cal_U: np.ndarray, cal_C: np.ndarray, small: int, large: int,
                           budget: float) -> float:
    """Escalation fraction t that spends ``budget`` in expectation (clipped to [0, 1])."""
    cL = float(np.mean(cal_C[:, small]))
    cH = float(np.mean(cal_C[:, large]))
    if cH <= 0:
        return 1.0
    return float(np.clip((budget - cL) / cH, 0.0, 1.0))
