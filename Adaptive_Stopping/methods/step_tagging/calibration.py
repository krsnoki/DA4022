"""Calibration of ST-ES on a training split (paper Sec. 5).

For every candidate configuration the replay gives accuracy and mean token
count on the calibration split. For each accuracy-retention target r (85%,
90%, 95% of the standard-inference accuracy), the paper keeps the "best"
configuration; we make that precise as: among configurations whose accuracy
is at least ``r * baseline_accuracy``, take the one with the fewest mean tokens
(ties: higher accuracy, then smaller delta). The same rule is applied to the
step-count baseline so both are calibrated identically.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Sequence

from Adaptive_Stopping.methods.step_tagging.taxonomy import TAG_IDS

PAPER_TARGETS = (0.85, 0.90, 0.95)
PAPER_DELTAS = tuple(range(0, 21))  # delta in [0, 20] (Fig. 3)
DEFAULT_STEP_COUNTS = tuple(list(range(1, 11)) + [12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120,
                                                    150, 200, 250, 300, 400, 500])


@dataclasses.dataclass(frozen=True)
class Config:
    kind: str  # "st_es" | "step_count"
    tag: str | None = None
    delta: int | None = None
    k: int | None = None

    @property
    def key(self) -> str:
        return f"st_es:{self.tag}:{self.delta}" if self.kind == "st_es" else f"step_count:{self.k}"

    def to_dict(self) -> dict:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}

    @staticmethod
    def from_dict(d: dict) -> "Config":
        return Config(kind=d["kind"], tag=d.get("tag"), delta=d.get("delta"), k=d.get("k"))


def build_grid(tags: Iterable[str] | None = None, deltas: Sequence[int] = PAPER_DELTAS,
               step_counts: Sequence[int] = DEFAULT_STEP_COUNTS) -> list[Config]:
    tags = [t for t in (tags or TAG_IDS) if t != "other"]
    grid = [Config("st_es", tag=t, delta=d) for t in tags for d in deltas]
    grid += [Config("step_count", k=k) for k in step_counts]
    return grid


def select_configs(rows: Sequence[dict], baseline_accuracy: float, kind: str,
                   targets: Sequence[float] = PAPER_TARGETS) -> dict[float, dict | None]:
    """Pick, per target, the cheapest config of ``kind`` keeping ``target * baseline_accuracy``.

    ``rows`` are sweep rows with keys ``kind``, ``accuracy``, ``mean_tokens``
    (and ``delta``/``k``). Returns ``{target: row or None}``; None means no
    configuration of that kind reached the target.
    """
    cands = [r for r in rows if r["kind"] == kind]
    out: dict[float, dict | None] = {}
    for t in targets:
        ok = [r for r in cands if r["accuracy"] >= t * baseline_accuracy - 1e-12]
        if not ok:
            out[t] = None
            continue
        out[t] = min(ok, key=lambda r: (r["mean_tokens"], -r["accuracy"], r.get("delta") or 0, r.get("k") or 0))
    return out
