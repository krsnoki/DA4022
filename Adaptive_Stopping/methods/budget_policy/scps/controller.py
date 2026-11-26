"""Semantic-Coverage Portfolio Search (SCPS), exact-answer instance.

Implements Algorithm 1 of the paper with the plug-in surrogate from
``surrogate.py``. Per planning round t:

1. build completion-level actions ``(source, n)`` from the root prompt,
   targeted (answer-diverse) root prompts and unfinished frontier prefixes;
2. estimate residual support ``v``, pairwise overlap ``omega`` and cost;
3. greedily select a feasible batch maximizing the Equation (7) score;
4. stop if the batch is empty or its score is not positive;
5. execute, add completed artifacts to the portfolio, update the frontier,
   the option-bank memory and the surrogate posteriors.

SCPS returns a portfolio; it does not choose a final answer.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Adaptive_Stopping.common.budget import Budget, Ledger
from Adaptive_Stopping.common.mcq import build_mcq_prompt, option_labels
from Adaptive_Stopping.common.policy import StoppingPolicy
from Adaptive_Stopping.common.portfolio import Completion, PortfolioResult
from Adaptive_Stopping.common.rollout import derive_seed, execute_samples, score_options, supports_option_scores

from .surrogate import PluginSurrogate, Source, softmax_from_logprobs, uniform


@dataclass(frozen=True)
class SCPSConfig:
    """Controller settings. Defaults mirror paper Table 8 where it gives a value."""

    temperature: float = 0.6
    max_rounds: int = 10  # Table 8: 10 planning rounds
    actions_per_round: int = 1  # Table 8: batch size 1
    sample_sizes: Tuple[int, ...] = (1, 2, 4)  # macro-action sizes; our choice
    root_proposal_batch: int = 16  # Table 8: max candidate actions scored per round
    distinct_source_target: int = 12  # Table 8: distinct-source-prefix target
    answer_class_quota: int = 8  # Table 8: terminal answer-class quota
    horizon: Optional[int] = None  # None = completion-level (max_new_tokens)
    discovery_weight: float = 1.0  # Table 8
    overlap_penalty: float = 0.5  # Table 8 (beta)
    cost_weight: float = 1.0  # lambda; cost is normalized by decode_token_limit
    seen_answer_penalty: float = 1.0  # Table 8; our definition, see score()
    repeat_penalty: float = 0.25  # Table 8; our definition, see score()
    posterior_mix: float = 0.05  # pi = (1 - eps) * root option probs + eps * uniform
    prior_strength: float = 4.0  # Dirichlet pseudo-count for source answer distributions
    null_prior: Tuple[float, float] = (1.0, 9.0)
    steer_prior: Tuple[float, float] = (1.0, 1.0)
    length_prior: float = 48.0
    targeted_actions: bool = True  # ablation: "no targeted action generation"
    score_prefixes: bool = True  # terminal-option scores for frontier prefixes

    def __post_init__(self):
        checks = [
            (self.temperature >= 0, "temperature >= 0"),
            (self.max_rounds >= 0, "max_rounds >= 0"),
            (self.actions_per_round >= 1, "actions_per_round >= 1"),
            (len(self.sample_sizes) > 0 and all(isinstance(k, int) and k >= 1 for k in self.sample_sizes), "sample_sizes >= 1"),
            (self.root_proposal_batch >= 1, "root_proposal_batch >= 1"),
            (self.distinct_source_target >= 1, "distinct_source_target >= 1"),
            (self.answer_class_quota >= 1, "answer_class_quota >= 1"),
            (self.horizon is None or self.horizon >= 1, "horizon >= 1 or None"),
            (0.0 <= self.posterior_mix <= 1.0, "posterior_mix in [0, 1]"),
            (self.prior_strength > 0, "prior_strength > 0"),
            (min(self.overlap_penalty, self.cost_weight, self.seen_answer_penalty, self.repeat_penalty) >= 0, "penalties >= 0"),
            (self.length_prior > 0, "length_prior > 0"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(f"invalid SCPSConfig: {msg}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SCPSConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown SCPSConfig keys: {sorted(unknown)}")
        d = dict(d)
        for k in ("sample_sizes", "null_prior", "steer_prior"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


@dataclass
class Action:
    source: Source
    n: int
    new_rollouts: int
    p: Dict[str, float]  # predicted per-sample answer distribution


class SCPSStoppingPolicy(StoppingPolicy):
    """SCPS stop rule expressed in the shared contract.

    Reads ``state['best_score']`` (score of the selected batch, computed from
    surrogate estimates already in memory, so no extra forward pass),
    ``state['round']``, ``state['n_seen']`` and ``state['feasible']``.
    """

    def __init__(self, max_rounds: int, answer_class_quota: int):
        self.max_rounds = max_rounds
        self.quota = answer_class_quota
        self.reset()

    def reset(self, request_metadata=None):
        self.reason: Optional[str] = None
        self.checks = 0

    def should_stop(self, state, metrics, budget) -> bool:
        self.checks += 1
        if state.get("round", 0) >= self.max_rounds:
            self.reason = "max_rounds"
        elif state.get("n_seen", 0) >= self.quota:
            self.reason = "answer_class_quota"
        elif not state.get("feasible", True):
            self.reason = "budget_exhausted"
        elif state.get("best_score") is not None and state["best_score"] <= 0.0:
            self.reason = "no_positive_action"
        else:
            return False
        return True

    def report(self):
        return {"policy": "scps", "stop_reason": self.reason, "checks": self.checks}


class SCPSController:
    method_name = "scps"

    def __init__(self, backbone, budget: Budget, config: SCPSConfig = SCPSConfig(), run_seed: int = 13):
        self.backbone = backbone
        self.budget = budget
        self.cfg = config
        self.run_seed = run_seed

    # ------------------------------------------------------------ scoring
    def score(self, batch: Sequence[Action], sur: PluginSurrogate, seen: set) -> Dict[str, float]:
        """Equation (7) plug-in score with two extra penalty terms.

        L(B) = w_d * sum_e v(e) - beta * sum_{e<f} omega(e, f) - lambda * sum_e c(e) / D
               - gamma_seen * sum_e n_e * seen_mass(e) * Lbar / D
               - gamma_repeat * sum_e executions(source_e) * n_e * Lbar / D

        D is the decode-token limit and Lbar the running mean completion
        length, so the last three terms are in "fraction of decode budget"
        units. The paper lists seen-answer and repeat penalties in Table 8
        without formulas; these definitions are ours.
        """
        cfg = self.cfg
        D = max(1, self.budget.decode_token_limit)
        Lbar = sur.mean_length
        v = sum(sur.residual_support(a.p, a.n, seen) for a in batch)
        om = 0.0
        for i, a in enumerate(batch):
            for b in batch[i + 1:]:
                om += sur.overlap(a.p, a.n, b.p, b.n, seen)
        cost = sum(sur.expected_cost(a.source, a.n, self.budget.max_new_tokens) for a in batch)
        seen_pen = sum(a.n * sur.seen_mass(a.p, seen) for a in batch) * Lbar / D
        rep = sum(sur.stats_for(a.source.key).executions * a.n for a in batch) * Lbar / D
        total = (
            cfg.discovery_weight * v
            - cfg.overlap_penalty * om
            - cfg.cost_weight * cost / D
            - cfg.seen_answer_penalty * seen_pen
            - cfg.repeat_penalty * rep
        )
        return {"score": total, "v": v, "omega": om, "cost": cost, "seen_pen": seen_pen, "repeat_pen": rep}

    def _feasible(self, ledger: Ledger, batch: Sequence[Action]) -> bool:
        n = sum(a.n for a in batch)
        new_r = sum(a.new_rollouts for a in batch)
        mt = max(self.budget.max_new_tokens - min(a.source.prefix_tokens for a in batch), 0) if batch else 0
        step = mt if self.cfg.horizon is None else min(self.cfg.horizon, mt)
        return ledger.can_sample(n, step, new_rollouts=new_r)

    def select_batch(self, actions: List[Action], sur: PluginSurrogate, seen: set, ledger: Ledger):
        batch: List[Action] = []
        cur = 0.0
        used_sources = set()
        while len(batch) < self.cfg.actions_per_round:
            best, best_gain, best_terms = None, 0.0, None
            for a in actions:
                if a.source.kind == "prefix" and a.source.key in used_sources:
                    continue  # a prefix can be continued once per round
                cand = batch + [a]
                if not self._feasible(ledger, cand):
                    continue
                terms = self.score(cand, sur, seen)
                gain = terms["score"] - cur
                if best is None or gain > best_gain:
                    best, best_gain, best_terms = a, gain, terms
            if best is None or best_gain <= 0.0:
                if not batch:
                    return [], (best_terms["score"] if best_terms else None), best is not None
                break
            batch.append(best)
            used_sources.add(best.source.key)
            cur = best_terms["score"]
        return batch, cur, True

    # ------------------------------------------------------------ actions
    def build_actions(self, sources: List[Source], sur: PluginSurrogate, seen: set, executed: set, ledger: Ledger) -> List[Action]:
        allow_new = len(executed) < self.cfg.distinct_source_target
        acts: List[Action] = []
        for src in sources:
            if src.key not in executed and not allow_new:
                continue
            if src.kind == "target" and src.target in seen:
                continue
            p = sur.answer_distribution(src)
            for n in self.cfg.sample_sizes:
                new_r = n - 1 if src.kind == "prefix" else n
                if ledger.rollouts + new_r > self.budget.max_rollouts:
                    continue
                acts.append(Action(src, n, new_r, p))
        # cap the candidate set by single-action residual support (root_proposal_batch)
        acts.sort(key=lambda a: -sur.residual_support(a.p, a.n, seen) / a.n)
        return acts[: self.cfg.root_proposal_batch * len(self.cfg.sample_sizes)]

    # ------------------------------------------------------------ main loop
    def run(self, item: Dict[str, Any]) -> PortfolioResult:
        t0 = time.perf_counter()
        cfg, bb = self.cfg, self.backbone
        qid = item.get("question_id", "q")
        labels = option_labels(len(item["options"]))
        ledger = Ledger(self.budget)
        user_root = build_mcq_prompt(item["question"], item["options"])
        telemetry = "option_scores" if supports_option_scores(bb) else "no_option_scores"

        root_scores = score_options(bb, user_root, labels, ledger)
        if root_scores is not None:
            p_pi = softmax_from_logprobs(root_scores.logprobs, 1.0)
            root_prior = softmax_from_logprobs(root_scores.logprobs, cfg.temperature or 1e-6)
        else:
            p_pi, root_prior = uniform(labels), uniform(labels)
        eps = cfg.posterior_mix
        pi = {a: (1 - eps) * p_pi[a] + eps / len(labels) for a in labels}
        sur = PluginSurrogate(
            labels, pi, root_prior,
            prior_strength=cfg.prior_strength, null_prior=cfg.null_prior,
            steer_prior=cfg.steer_prior, length_prior=min(cfg.length_prior, self.budget.max_new_tokens),
        )

        sources: List[Source] = [Source("root", "root", user_root, root_prior)]
        if cfg.targeted_actions:
            for a in sorted(labels, key=lambda x: -pi[x]):
                sources.append(Source(f"target:{a}", "target", build_mcq_prompt(item["question"], item["options"], target=a), root_prior, target=a))

        policy = SCPSStoppingPolicy(cfg.max_rounds, cfg.answer_class_quota)
        policy.reset({"question_id": qid})
        portfolio: List[Completion] = []
        seen: set = set()
        executed: set = set()
        prefix_meta: Dict[str, int] = {}  # prefix key -> unattributed tokens
        trace: List[Dict[str, Any]] = []
        rnd, prefix_counter = 0, 0

        while True:
            state = {"round": rnd, "n_seen": len(seen), "feasible": ledger.max_affordable_samples(1) > 0 or any(
                s.kind == "prefix" for s in sources)}
            if policy.should_stop(state, {}, self.budget):
                break
            actions = self.build_actions(sources, sur, seen, executed, ledger)
            batch, best_score, any_feasible = self.select_batch(actions, sur, seen, ledger)
            state.update({"best_score": best_score, "feasible": any_feasible})
            if not batch:
                if not policy.should_stop(state, {}, self.budget):
                    policy.reason = "no_positive_action" if any_feasible else "budget_exhausted"
                break
            terms = self.score(batch, sur, seen)
            round_log = {"round": rnd, "actions": [], "score": round(terms["score"], 6),
                         "v": round(terms["v"], 6), "omega": round(terms["omega"], 6), "cost": round(terms["cost"], 2)}
            for act in batch:
                src = act.source
                seed = derive_seed(self.run_seed, qid, rnd, src.key, act.n)
                out = execute_samples(
                    bb, src.prompt, labels, ledger, act.n, cfg.temperature, seed, cfg.horizon,
                    prefix_text=src.prefix_text, prefix_tokens=src.prefix_tokens,
                    prefix_unattributed=prefix_meta.get(src.key, 0),
                )
                executed.add(src.key)
                answers = [a for a, _, _ in out.completed]
                sur.observe(src, answers, [t for _, t, _ in out.completed])
                for ans, tok, text in out.completed:
                    portfolio.append(Completion(ans, tok, src.key, text, rnd))
                    if ans is not None:
                        seen.add(ans)
                if src.kind == "prefix":
                    sources = [s for s in sources if s.key != src.key]
                for text, plen, unattr in out.unfinished:
                    prefix_counter += 1
                    key = f"prefix:{prefix_counter}"
                    user = src.prompt
                    prior = root_prior
                    if cfg.score_prefixes:
                        sc = score_options(bb, user, labels, ledger, prefix_text=text)
                        if sc is not None:
                            prior = softmax_from_logprobs(sc.logprobs, cfg.temperature or 1e-6)
                    sources.append(Source(key, "prefix", user, prior, prefix_text=text, prefix_tokens=plen))
                    prefix_meta[key] = unattr
                round_log["actions"].append({"source": src.key, "n": act.n, "answers": answers,
                                             "unfinished": len(out.unfinished)})
            round_log["seen"] = sorted(seen)
            round_log["surrogate"] = sur.snapshot()
            trace.append(round_log)
            rnd += 1

        return PortfolioResult(
            method=self.method_name,
            completions=portfolio,
            ledger=ledger.as_dict(),
            rounds=rnd,
            stop_reason=policy.reason or "unknown",
            trace=trace,
            wall_time_s=time.perf_counter() - t0,
            metadata={"telemetry": telemetry, "config": asdict(cfg), "policy": policy.report(),
                      "violations": ledger.violations()},
        )


__all__ = ["SCPSConfig", "SCPSController", "SCPSStoppingPolicy", "Action"]
