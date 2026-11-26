"""Tree-prefix expansion and semantic-pruning baselines (paper Table 7).

Tree-prefix: a Tree-of-Thoughts-style frontier search. Each round expands
the ``beam`` highest-ranked frontier nodes by sampling ``branch``
continuations of up to ``horizon`` tokens. Nodes are ranked by the
maximum terminal-option probability at the node (model confidence). It
uses no residual-support or overlap scoring. When the frontier is empty
the root is expanded again.

Semantic pruning: same search, but while walking the ranked list it skips
nodes whose predicted answer (argmax option) is already in the portfolio
or whose predicted answer distribution has cosine overlap above
``overlap_threshold`` (paper: 0.75) with a node already chosen this round.
If every node would be skipped, the top node is expanded anyway.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from Adaptive_Stopping.common.budget import Budget, Ledger
from Adaptive_Stopping.common.mcq import build_mcq_prompt, option_labels
from Adaptive_Stopping.common.portfolio import Completion, PortfolioResult
from Adaptive_Stopping.common.rollout import derive_seed, execute_samples, score_options
from Adaptive_Stopping.methods.budget_policy.scps.surrogate import softmax_from_logprobs, uniform


@dataclass(frozen=True)
class TreePrefixConfig:
    temperature: float = 0.6
    beam: int = 2
    branch: int = 2
    horizon: Optional[int] = 64
    semantic_pruning: bool = False
    overlap_threshold: float = 0.75

    def __post_init__(self):
        if self.beam < 1 or self.branch < 1 or (self.horizon is not None and self.horizon < 1):
            raise ValueError("invalid TreePrefixConfig")
        if not 0.0 <= self.overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be in [0, 1]")


@dataclass
class Node:
    key: str
    prefix_text: str
    prefix_tokens: int
    unattributed: int
    probs: Dict[str, float]

    @property
    def confidence(self) -> float:
        return max(self.probs.values())

    @property
    def predicted(self) -> str:
        return max(self.probs, key=self.probs.get)


def cosine(p: Dict[str, float], q: Dict[str, float]) -> float:
    num = sum(p[k] * q.get(k, 0.0) for k in p)
    den = math.sqrt(sum(v * v for v in p.values())) * math.sqrt(sum(v * v for v in q.values()))
    return 0.0 if den == 0 else num / den


class TreePrefix:
    method_name = "tree_prefix"

    def __init__(self, backbone, budget: Budget, config: TreePrefixConfig = TreePrefixConfig(), run_seed: int = 13):
        self.backbone, self.budget, self.cfg, self.run_seed = backbone, budget, config, run_seed
        if config.semantic_pruning:
            self.method_name = "semantic_pruning"

    def _probs(self, user: str, labels, ledger: Ledger, prefix: str) -> Dict[str, float]:
        sc = score_options(self.backbone, user, labels, ledger, prefix_text=prefix)
        return uniform(labels) if sc is None else softmax_from_logprobs(sc.logprobs, 1.0)

    def _select(self, frontier: List[Node], seen: set) -> List[Node]:
        ranked = sorted(frontier, key=lambda n: -n.confidence)
        if not self.cfg.semantic_pruning:
            return ranked[: self.cfg.beam]
        chosen: List[Node] = []
        for node in ranked:
            if len(chosen) == self.cfg.beam:
                break
            if node.predicted in seen:
                continue
            if any(cosine(node.probs, c.probs) > self.cfg.overlap_threshold for c in chosen):
                continue
            chosen.append(node)
        return chosen or ranked[:1]

    def run(self, item: Dict[str, Any]) -> PortfolioResult:
        t0 = time.perf_counter()
        cfg = self.cfg
        labels = option_labels(len(item["options"]))
        qid = item.get("question_id", "q")
        user = build_mcq_prompt(item["question"], item["options"])
        ledger = Ledger(self.budget)
        root = Node("root", "", 0, 0, self._probs(user, labels, ledger, ""))
        frontier: List[Node] = []
        portfolio: List[Completion] = []
        seen: set = set()
        rnd, counter, stop = 0, 0, "budget_exhausted"
        while True:
            pool = frontier if frontier else [root]
            chosen = self._select(pool, seen)
            progressed = False
            for node in chosen:
                room = self.budget.max_new_tokens - node.prefix_tokens
                step = room if cfg.horizon is None else min(cfg.horizon, room)
                is_root = node.key == "root"
                new_r = cfg.branch if is_root else cfg.branch - 1
                n = cfg.branch
                while n > 0 and not ledger.can_sample(n, step, new_rollouts=new_r):
                    n -= 1
                    new_r = n if is_root else n - 1
                if n <= 0:
                    continue
                seed = derive_seed(self.run_seed, qid, self.method_name, rnd, node.key)
                out = execute_samples(self.backbone, user, labels, ledger, n, cfg.temperature, seed, cfg.horizon,
                                      node.prefix_text, node.prefix_tokens, node.unattributed)
                progressed = True
                for ans, tok, txt in out.completed:
                    portfolio.append(Completion(ans, tok, node.key, txt, rnd))
                    if ans is not None:
                        seen.add(ans)
                if not is_root:
                    frontier = [f for f in frontier if f.key != node.key]
                for txt, plen, unattr in out.unfinished:
                    counter += 1
                    frontier.append(Node(f"prefix:{counter}", txt, plen, unattr, self._probs(user, labels, ledger, txt)))
            rnd += 1
            if not progressed:
                break
            if rnd >= self.budget.decode_token_limit + self.budget.max_rollouts + 1:
                stop = "round_cap"  # safety net against zero-token continuations
                break
        return PortfolioResult(
            self.method_name, portfolio, ledger.as_dict(), rnd, stop,
            wall_time_s=time.perf_counter() - t0,
            metadata={"config": cfg.__dict__, "violations": ledger.violations(), "frontier_left": len(frontier)},
        )


__all__ = ["TreePrefix", "TreePrefixConfig", "cosine"]
