"""Synthetic multiple-choice items for smoke tests and controller microbenchmarks.

Each item carries a hidden ``_mock`` block that the mock backbone
(``Backbones/Generative_Pretrained/mock_mcq_lm.py``) uses as its latent
answer distribution. Results on these items test controller logic only and
must never be reported as task quality.
"""

from __future__ import annotations

import random
import string
from typing import Dict, List


def _dirichlet(rng: random.Random, k: int, alpha: float) -> List[float]:
    xs = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    s = sum(xs) or 1.0
    return [x / s for x in xs]


def make_synthetic_mcq(
    num_items: int,
    num_options: int = 10,
    seed: int = 0,
    concentration: float = 0.3,
    p_gold_is_mode: float = 0.5,
    steer_range=(0.3, 0.8),
    null_rate: float = 0.05,
    reasoning_words=(2, 40),
) -> List[Dict]:
    """Deterministic synthetic items in the MMLU-Pro record format plus ``_mock``."""
    if num_items < 0 or not 2 <= num_options <= 26:
        raise ValueError("num_items >= 0 and 2 <= num_options <= 26 required")
    rng = random.Random(seed)
    labels = string.ascii_uppercase[:num_options]
    items = []
    for i in range(num_items):
        dist = _dirichlet(rng, num_options, concentration)
        mode = max(range(num_options), key=dist.__getitem__)
        if rng.random() < p_gold_is_mode:
            gold = mode
        else:
            others = [j for j in range(num_options) if j != mode]
            w = [dist[j] + 1e-3 for j in others]
            gold = rng.choices(others, weights=w)[0]
        items.append({
            "question_id": f"syn-{seed}-{i}",
            "question": f"Synthetic question {seed}-{i}?",
            "options": [f"synthetic option {labels[j]} for item {i}" for j in range(num_options)],
            "answer": labels[gold],
            "answer_index": gold,
            "category": "synthetic",
            "_mock": {
                "dist": dist,
                "steer": rng.uniform(*steer_range),
                "null_rate": null_rate,
                "reasoning_words": list(reasoning_words),
            },
        })
    return items


__all__ = ["make_synthetic_mcq"]
