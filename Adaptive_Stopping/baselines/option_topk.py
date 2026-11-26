"""Option top-K control: list the model's K most probable options without sampling.

This is a sanity control for exact-answer portfolio metrics on multiple
choice. A portfolio that contains K distinct options passes Pass@N
whenever the gold option is among them, so a sampler can raise Pass@N
just by covering more options. This control asks how often the gold
option is among the model's own top-K options by next-token probability
after the root prompt (one forward pass, no generation).

The full root option distribution is stored in ``metadata['record_extra']``
so the analysis can evaluate any K afterwards, including a K matched per
question to the number of distinct answers another method found.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

from Adaptive_Stopping.common.budget import Budget, Ledger
from Adaptive_Stopping.common.mcq import build_mcq_prompt, option_labels
from Adaptive_Stopping.common.portfolio import Completion, PortfolioResult
from Adaptive_Stopping.common.rollout import score_options


@dataclass(frozen=True)
class OptionTopKConfig:
    k: int = 8  # default matches SCPS's answer-class quota

    def __post_init__(self):
        if self.k < 1:
            raise ValueError("k must be >= 1")


class OptionTopK:
    method_name = "option_topk"

    def __init__(self, backbone, budget: Budget, config: OptionTopKConfig = OptionTopKConfig(), run_seed: int = 13):
        self.backbone, self.budget, self.cfg, self.run_seed = backbone, budget, config, run_seed

    def run(self, item: Dict[str, Any]) -> PortfolioResult:
        t0 = time.perf_counter()
        labels = option_labels(len(item["options"]))
        ledger = Ledger(self.budget)
        user = build_mcq_prompt(item["question"], item["options"])
        sc = score_options(self.backbone, user, labels, ledger)
        if sc is None:
            return PortfolioResult(self.method_name, [], ledger.as_dict(), 0, "no_option_scores",
                                   wall_time_s=time.perf_counter() - t0,
                                   metadata={"violations": ledger.violations(), "record_extra": {"root_probs": None}})
        probs = sc.probs()
        ranked = sorted(labels, key=lambda a: -probs[a])
        comps = [Completion(a, 0, "option_topk") for a in ranked[: self.cfg.k]]
        return PortfolioResult(
            self.method_name, comps, ledger.as_dict(), 1, "done", wall_time_s=time.perf_counter() - t0,
            metadata={"violations": ledger.violations(),
                      "record_extra": {"root_probs": {a: round(probs[a], 6) for a in labels}}},
        )


__all__ = ["OptionTopK", "OptionTopKConfig"]
