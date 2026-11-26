"""Deterministic mock generative backbone for CPU smoke tests.

Registry ID: ``mock.mcq.latent_dist`` (synthetic, no weights).

The mock reads the question text and the optional targeted hint from the
prompt, then samples answers from the item's hidden ``_mock['dist']``.
A targeted prompt lands on its target with probability ``_mock['steer']``.
Tokens are whitespace-separated words. Option scores are the log of the
latent distribution, so the mock is a well-calibrated model by design.
"""

from __future__ import annotations

import math
import random
import re
import string
from typing import Dict, Iterable, List, Optional, Sequence

from .interface import Generation, OptionScores, log_softmax

ASSISTANT_MARK = "\n<|assistant|>\n"
_Q = re.compile(r"Question: (.*?)\nOptions:", re.DOTALL)
_T = re.compile(r"check whether option ([A-Z]) is correct")
_SCORING = "The answer is ("


class MockMCQBackbone:
    def __init__(self, items: Iterable[dict]):
        self._items = {it["question"].strip(): it for it in items}
        self.calls = {"generate": 0, "score": 0}

    # ------------------------------------------------------------ interface
    def render_prompt(self, user_message: str, assistant_prefix: str = "") -> str:
        return user_message + ASSISTANT_MARK + assistant_prefix

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_model_metadata(self) -> dict:
        return {"id": "mock.mcq.latent_dist", "revision": "n/a", "dtype": "n/a", "device": "cpu",
                "pretraining_regime": "unknown", "notes": "synthetic mock backbone"}

    def _parse(self, prompt: str):
        user, _, prefix = prompt.partition(ASSISTANT_MARK)
        m = _Q.search(user)
        if not m:
            raise KeyError("mock backbone could not find 'Question:' in prompt")
        item = self._items[m.group(1).strip()]
        t = _T.search(user)
        return item, (t.group(1) if t else None), prefix

    def generate(self, prompt: str, n: int, max_tokens: int, temperature: float, seed: Optional[int] = None) -> List[Generation]:
        self.calls["generate"] += 1
        item, target, prefix = self._parse(prompt)
        mk = item["_mock"]
        labels = string.ascii_uppercase[: len(item["options"])]
        dist = list(mk["dist"])
        if temperature and temperature > 0:
            w = [p ** (1.0 / temperature) for p in dist]
            z = sum(w)
            dist = [x / z for x in w]
        out = []
        for i in range(n):
            rng = random.Random(f"{seed}|{i}|{len(prefix)}")
            lo, hi = mk["reasoning_words"]
            words = ["step"] * rng.randint(lo, hi)
            if rng.random() < mk["null_rate"]:
                words += ["unsure", "no", "final", "choice"]
            else:
                if target is not None and rng.random() < mk["steer"]:
                    ans = target
                else:
                    ans = rng.choices(labels, weights=dist)[0]
                words += ["The", "answer", "is", f"({ans})."]
            take = words[:max_tokens]
            text = (" " if prefix and not prefix.endswith((" ", "\n")) else "") + " ".join(take)
            out.append(Generation(text=text, num_tokens=len(take), finished=len(words) <= max_tokens))
        return out

    def score_next_options(self, prompt: str, labels: Sequence[str]) -> OptionScores:
        self.calls["score"] += 1
        item, _, _ = self._parse(prompt)
        dist = item["_mock"]["dist"]
        raw = {lab: math.log(max(dist[i], 1e-12)) for i, lab in enumerate(labels)}
        return OptionScores(log_softmax(raw), aux_tokens=self.count_tokens(_SCORING) + 1)


class NoScoreMockBackbone:
    """Same mock without ``score_next_options`` (missing-telemetry tests)."""

    def __init__(self, items):
        self._inner = MockMCQBackbone(items)

    def render_prompt(self, user_message, assistant_prefix=""):
        return self._inner.render_prompt(user_message, assistant_prefix)

    def count_tokens(self, text):
        return self._inner.count_tokens(text)

    def get_model_metadata(self):
        return self._inner.get_model_metadata()

    def generate(self, *a, **k):
        return self._inner.generate(*a, **k)


__all__ = ["MockMCQBackbone", "NoScoreMockBackbone", "ASSISTANT_MARK"]
