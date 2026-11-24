"""Per-query outcome tables and calibration/test splits.

A :class:`CascadeTable` holds, for one dataset and M models, aligned arrays
over the same N queries: correctness ``U``, costs ``C[cost_name]``, raw scores
``S[score_name]`` and (optionally) latency. It is built from the JSONL files
written by ``experiments/cascade_escalation/collect_generations.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .scores import SCORE_NAMES


@dataclass
class CascadeTable:
    dataset: str
    models: List[str]
    uids: List[str]
    U: np.ndarray                      # [N, M]
    C: Dict[str, np.ndarray]           # cost name -> [N, M]
    S: Dict[str, np.ndarray]           # score name -> [N, M]
    extra: Dict[str, np.ndarray] = field(default_factory=dict)  # e.g. n_in, n_out

    @property
    def n(self) -> int:
        return self.U.shape[0]

    @classmethod
    def from_rows(cls, dataset: str, rows_by_model: Dict[str, List[Dict]], num_parameters: Dict[str, int]) -> "CascadeTable":
        models = list(rows_by_model)
        common = None
        for m in models:
            ids = {r["uid"] for r in rows_by_model[m]}
            common = ids if common is None else common & ids
        uids = sorted(common or [])
        if not uids:
            raise ValueError(f"no common queries across models for {dataset}")
        idx = {m: {r["uid"]: r for r in rows_by_model[m]} for m in models}
        N, M = len(uids), len(models)
        U = np.zeros((N, M))
        flops = np.zeros((N, M))
        gpu = np.zeros((N, M))
        n_in = np.zeros((N, M))
        n_out = np.zeros((N, M))
        S = {s: np.zeros((N, M)) for s in SCORE_NAMES}
        for j, m in enumerate(models):
            P = float(num_parameters[m])
            for i, u in enumerate(uids):
                r = idx[m][u]
                U[i, j] = float(r["correct"])
                n_in[i, j] = r["n_in"]
                n_out[i, j] = r["n_out"]
                flops[i, j] = 2.0 * P * (r["n_in"] + r["n_out"])
                gpu[i, j] = float(r.get("gpu_s", np.nan))
                for s in SCORE_NAMES:
                    S[s][i, j] = float(r["scores"][s])
        C = {"flops": flops / 1e12, "gpu_time": gpu}  # TFLOPs and seconds per query
        return cls(dataset, models, uids, U, C, S, {"n_in": n_in, "n_out": n_out})


def stratified_split(strata: np.ndarray, seed: int, cal_fraction: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Paper's split: stratified on the first (cheapest) model's correctness,
    calibration count rounded down per stratum. Returns sorted index arrays."""
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    cal: List[np.ndarray] = []
    test: List[np.ndarray] = []
    for v in np.unique(strata):
        ids = np.flatnonzero(strata == v)
        ids = ids[rng.permutation(len(ids))]
        k = int(np.floor(cal_fraction * len(ids)))
        cal.append(ids[:k])
        test.append(ids[k:])
    return np.sort(np.concatenate(cal)), np.sort(np.concatenate(test))


def cheapest_model(C: np.ndarray) -> int:
    return int(np.argmin(np.asarray(C).mean(axis=0)))
