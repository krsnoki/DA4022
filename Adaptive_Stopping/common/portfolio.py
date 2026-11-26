"""Portfolio records and exact-answer portfolio metrics.

Metric definitions follow SCPS paper Appendix H.2:

* ``pass_at_n``: 1 if any completion canonicalizes to the gold label.
* ``unique_answers``: distinct non-null answer classes.
* ``duplicate_token_fraction``: decode tokens spent on completions whose
  answer class was already in the portfolio when they arrived, divided by
  all decode tokens of completed artifacts. Null completions are counted
  in the denominator but never as duplicates (the paper does not say how
  nulls are treated; this choice is ours).
* ``majority_answer``: most frequent non-null class, ties broken by first
  arrival. A diagnostic only; SCPS itself does not select an answer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Completion:
    answer: Optional[str]
    num_tokens: int
    source: str  # action/source key that produced it
    text: str = ""
    round: int = 0


@dataclass
class PortfolioResult:
    method: str
    completions: List[Completion]
    ledger: Dict[str, Any]
    rounds: int
    stop_reason: str
    trace: List[Dict[str, Any]] = field(default_factory=list)
    wall_time_s: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def answers(self) -> List[Optional[str]]:
        return [c.answer for c in self.completions]

    @property
    def answer_set(self) -> set:
        return {a for a in self.answers if a is not None}

    def unique_answers(self) -> int:
        return len(self.answer_set)

    def pass_at_n(self, gold: str) -> int:
        return int(gold in self.answer_set)

    def duplicate_token_fraction(self) -> Optional[float]:
        seen, dup, total = set(), 0, 0
        for c in self.completions:
            total += c.num_tokens
            if c.answer is not None:
                if c.answer in seen:
                    dup += c.num_tokens
                seen.add(c.answer)
        return None if total == 0 else dup / total

    def majority_answer(self) -> Optional[str]:
        counts = Counter(a for a in self.answers if a is not None)
        if not counts:
            return None
        best = max(counts.values())
        for a in self.answers:
            if a is not None and counts[a] == best:
                return a
        return None

    def to_dict(self, include_text: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if not include_text:
            for c in d["completions"]:
                c.pop("text", None)
        return d


__all__ = ["Completion", "PortfolioResult"]
