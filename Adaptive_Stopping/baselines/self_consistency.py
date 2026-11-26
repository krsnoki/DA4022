"""Self-consistency baseline: independent root rollouts until the hard budget is spent.

This is the no-stopping, full-compute reference for SCPS (paper Table 13,
"Self-consistency r16"): every question receives ``max_rollouts`` samples
at ``max_new_tokens`` regardless of redundancy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

from Adaptive_Stopping.common.budget import Budget, Ledger
from Adaptive_Stopping.common.mcq import build_mcq_prompt, option_labels
from Adaptive_Stopping.common.policy import NeverStop
from Adaptive_Stopping.common.portfolio import Completion, PortfolioResult
from Adaptive_Stopping.common.rollout import derive_seed, execute_samples


@dataclass(frozen=True)
class SelfConsistencyConfig:
    temperature: float = 0.6
    samples_per_call: int = 16  # one batched call by default

    def __post_init__(self):
        if self.temperature < 0 or self.samples_per_call < 1:
            raise ValueError("invalid SelfConsistencyConfig")


class SelfConsistency:
    method_name = "self_consistency"

    def __init__(self, backbone, budget: Budget, config: SelfConsistencyConfig = SelfConsistencyConfig(), run_seed: int = 13):
        self.backbone, self.budget, self.cfg, self.run_seed = backbone, budget, config, run_seed

    def run(self, item: Dict[str, Any]) -> PortfolioResult:
        t0 = time.perf_counter()
        labels = option_labels(len(item["options"]))
        qid = item.get("question_id", "q")
        user = build_mcq_prompt(item["question"], item["options"])
        ledger = Ledger(self.budget)
        policy = NeverStop()
        policy.reset({"question_id": qid})
        portfolio, call = [], 0
        while not policy.should_stop({}, {}, self.budget):
            n = min(self.cfg.samples_per_call, ledger.max_affordable_samples())
            if n <= 0:
                break
            seed = derive_seed(self.run_seed, qid, "sc", call)
            out = execute_samples(self.backbone, user, labels, ledger, n, self.cfg.temperature, seed)
            portfolio += [Completion(a, t, "root", txt, call) for a, t, txt in out.completed]
            call += 1
        return PortfolioResult(
            self.method_name, portfolio, ledger.as_dict(), call, "budget_exhausted",
            wall_time_s=time.perf_counter() - t0,
            metadata={"policy": policy.report(), "violations": ledger.violations()},
        )


__all__ = ["SelfConsistency", "SelfConsistencyConfig"]
