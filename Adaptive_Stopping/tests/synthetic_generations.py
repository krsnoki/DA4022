"""Write synthetic generation files (same format as collect_generations.py).

Used by tests and for timing the offline sweep without a GPU. Correctness and
scores follow a shared-difficulty model; nothing here is a real result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Adaptive_Stopping" / "experiments" / "cascade_escalation"))
from common import generation_path, load_config, write_jsonl  # noqa: E402

from Adaptive_Stopping.methods.cascade_escalation.scores import SCORE_NAMES  # noqa: E402


def write_synthetic(cfg: dict, out: Path, n: int = 400, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    models = cfg["models"]
    for di, ds in enumerate(cfg["datasets"]):
        diff = rng.normal(size=n)
        for j, m in enumerate(models):
            skill = -0.8 + 2.0 * j / max(len(models) - 1, 1)
            logit = skill - diff + 0.6 * rng.normal(size=n)
            correct = rng.random(n) < 1 / (1 + np.exp(-2 * logit))
            base = 1 / (1 + np.exp(-(logit + rng.normal(scale=1.0, size=n))))
            n_in = rng.integers(40, 120, size=n)
            n_out = rng.integers(1, 30, size=n) * (1 + 10 * (ds == "math_hard"))
            rows = [{"uid": f"{ds}-{i}", "response": "x", "n_in": int(n_in[i]), "n_out": int(n_out[i]),
                     "finish": "stop", "correct": int(correct[i]),
                     "gpu_s": float(1e-4 * m["num_parameters"] ** 0.5 * (n_out[i] + 1) / 1000),
                     "scores": {s: float(np.clip(base[i] + 0.02 * k * rng.normal(), 0, 1))
                                for k, s in enumerate(SCORE_NAMES)}}
                    for i in range(n)]
            write_jsonl(generation_path(out, ds, m["slug"]), rows)


if __name__ == "__main__":
    write_synthetic(load_config(sys.argv[1]), Path(sys.argv[2]), n=int(sys.argv[3]) if len(sys.argv) > 3 else 400)
