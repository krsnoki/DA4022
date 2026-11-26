"""Score providers, including the paper's calibration-trained scores.

Bouchard (2026) App. B.3 evaluates, besides the five raw log-prob scores:

* **correctness-trained token-prob score** - logistic regression on the five
  base scores of a model, predicting that model's own correctness. The score
  is the predicted P(correct).
* **benefit-trained (Diff-01) token-prob score** - ridge regression on the
  five base scores of model i predicting U_{next} - U_i, where ``next`` is the
  model the query would escalate to. The score is the *negated* prediction,
  so a large predicted benefit gives a low score and triggers escalation.

Both are fitted on calibration rows only. The paper's embedding-based
variants need a sentence-embedding model and are not implemented here (see
paper card). Implementations are plain NumPy (no scikit-learn dependency):
L2-regularised logistic regression by Newton/IRLS (C = 1, like scikit-learn's
default) and closed-form ridge (alpha = 1) on standardised features.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .scores import SCORE_NAMES


def _standardise(X: np.ndarray, mu=None, sd=None):
    if mu is None:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd, mu, sd


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0, iters: int = 50):
    """L2 logistic regression (penalty 1/(2C)||w||^2, intercept unpenalised)."""
    Xs, mu, sd = _standardise(np.asarray(X, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    n, p = Xs.shape
    A = np.hstack([np.ones((n, 1)), Xs])
    w = np.zeros(p + 1)
    reg = np.full(p + 1, 1.0 / C)
    reg[0] = 0.0
    for _ in range(iters):
        z = np.clip(A @ w, -30, 30)
        q = 1.0 / (1.0 + np.exp(-z))
        g = A.T @ (q - y) + reg * w
        H = (A * (q * (1 - q))[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(p + 1)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break

    def predict(Xn: np.ndarray) -> np.ndarray:
        Z, _, _ = _standardise(np.asarray(Xn, dtype=np.float64), mu, sd)
        return 1.0 / (1.0 + np.exp(-np.clip(w[0] + Z @ w[1:], -30, 30)))

    return predict


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    Xs, mu, sd = _standardise(np.asarray(X, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    ym = y.mean()
    p = Xs.shape[1]
    w = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(p), Xs.T @ (y - ym))

    def predict(Xn: np.ndarray) -> np.ndarray:
        Z, _, _ = _standardise(np.asarray(Xn, dtype=np.float64), mu, sd)
        return ym + Z @ w

    return predict


class ScoreProvider:
    """Callable ``(model, next_model) -> scores[N]`` used by the cascade search.

    Args:
        base: dict score_name -> [N, M] array of raw scores.
        kind: one of the raw names in SCORE_NAMES, ``"correctness_tokenprob"``,
            ``"benefit_tokenprob"`` or ``"random"``.
        U: [N, M] correctness, required for learned kinds.
        cal_idx: calibration rows used to fit learned kinds.
        seed: seed for the ``"random"`` (uninformative) score.
    """

    LEARNED = ("correctness_tokenprob", "benefit_tokenprob")

    def __init__(
        self,
        base: Dict[str, np.ndarray],
        kind: str,
        U: Optional[np.ndarray] = None,
        cal_idx: Optional[np.ndarray] = None,
        seed: int = 0,
    ) -> None:
        self.base = base
        self.kind = kind
        self.U = None if U is None else np.asarray(U, dtype=np.float64)
        self.cal_idx = cal_idx
        self._cache: Dict = {}
        any_arr = next(iter(base.values()))
        self.N, self.M = any_arr.shape
        if kind == "random":
            self._random = np.random.default_rng(seed).random((self.N, self.M))
        elif kind in self.LEARNED:
            if self.U is None or cal_idx is None:
                raise ValueError(f"{kind} needs U and cal_idx")
        elif kind not in SCORE_NAMES:
            raise KeyError(f"unknown score kind {kind!r}")

    def features(self, model: int) -> np.ndarray:
        return np.stack([self.base[name][:, model] for name in SCORE_NAMES], axis=1)

    def __call__(self, model: int, next_model: Optional[int] = None) -> np.ndarray:
        if self.kind in SCORE_NAMES:
            return self.base[self.kind][:, model]
        if self.kind == "random":
            return self._random[:, model]
        key = (model, next_model if self.kind == "benefit_tokenprob" else None)
        if key not in self._cache:
            X = self.features(model)
            cal = self.cal_idx
            if self.kind == "correctness_tokenprob":
                y = self.U[cal, model]
                if y.min() == y.max():  # degenerate: constant correctness
                    self._cache[key] = np.full(self.N, float(y[0]))
                else:
                    self._cache[key] = fit_logistic(X[cal], y)(X)
            else:
                if next_model is None:
                    raise ValueError("benefit score needs the next model")
                y = self.U[cal, next_model] - self.U[cal, model]
                self._cache[key] = -fit_ridge(X[cal], y)(X)
        return self._cache[key]


ALL_SCORE_KINDS: Sequence[str] = tuple(SCORE_NAMES) + ScoreProvider.LEARNED
