"""Hard budgets and realized-cost accounting shared by SCPS and its baselines.

Budget fields follow the SCPS paper protocol (Table 8): maximum rollouts,
maximum new tokens per completion, a decode-token limit and a total-token
limit, where total = decode + auxiliary scoring tokens. Feasibility is
checked against the worst case (every sampled completion uses
``max_new_tokens``), so a run can never exceed a hard limit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Budget:
    max_rollouts: int = 16
    max_new_tokens: int = 192
    decode_token_limit: int = 3072
    total_token_limit: int = 30000

    def __post_init__(self):
        for name in ("max_rollouts", "max_new_tokens", "decode_token_limit", "total_token_limit"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"Budget.{name} must be a non-negative int, got {v!r}")


@dataclass
class Ledger:
    """Realized resource use for one question."""

    budget: Budget
    rollouts: int = 0
    decode_tokens: int = 0
    aux_tokens: int = 0
    prompt_tokens: int = 0  # prefill, reported separately (not budgeted)
    generate_calls: int = 0
    scoring_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.decode_tokens + self.aux_tokens

    def remaining_rollouts(self) -> int:
        return self.budget.max_rollouts - self.rollouts

    def can_sample(
        self, n: int, max_tokens: int | None = None, aux_estimate: int = 0, new_rollouts: int | None = None
    ) -> bool:
        """Worst-case check that ``n`` more samples fit every hard limit.

        ``new_rollouts`` defaults to ``n``. Continuing an unfinished prefix
        with ``n`` samples creates ``n - 1`` new rollouts, because the first
        sample extends the existing trace.
        """
        if n <= 0:
            return False
        mt = self.budget.max_new_tokens if max_tokens is None else min(max_tokens, self.budget.max_new_tokens)
        if mt <= 0:
            return False
        nr = n if new_rollouts is None else new_rollouts
        if self.rollouts + nr > self.budget.max_rollouts:
            return False
        if self.decode_tokens + n * mt > self.budget.decode_token_limit:
            return False
        if self.total_tokens + n * mt + aux_estimate > self.budget.total_token_limit:
            return False
        return True

    def can_score(self, aux_estimate: int) -> bool:
        return self.total_tokens + aux_estimate <= self.budget.total_token_limit

    def max_affordable_samples(self, max_tokens: int | None = None) -> int:
        n = 0
        while self.can_sample(n + 1, max_tokens):
            n += 1
        return n

    def charge_generation(self, n_samples: int, decode_tokens: int, prompt_tokens: int = 0) -> None:
        self.rollouts += n_samples
        self.decode_tokens += decode_tokens
        self.prompt_tokens += prompt_tokens
        self.generate_calls += 1

    def charge_scoring(self, aux_tokens: int) -> None:
        self.aux_tokens += aux_tokens
        self.scoring_calls += 1

    def violations(self) -> list:
        b, out = self.budget, []
        if self.rollouts > b.max_rollouts:
            out.append("max_rollouts")
        if self.decode_tokens > b.decode_token_limit:
            out.append("decode_token_limit")
        if self.total_tokens > b.total_token_limit:
            out.append("total_token_limit")
        return out

    def as_dict(self) -> dict:
        d = asdict(self)
        d["budget"] = asdict(self.budget)
        d["total_tokens"] = self.total_tokens
        return d


__all__ = ["Budget", "Ledger"]
