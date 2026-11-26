"""Execute bounded rollout actions against a generative backbone.

Shared by SCPS and the baselines so that every method uses the same
prompt rendering, canonicalizer, completion rule and token accounting.

Completion rule: a sample is a completed artifact when it contains a
parseable answer, ends on EOS/stop or reaches ``max_new_tokens`` in total.
Otherwise it is an unfinished prefix that returns to the frontier and adds
no portfolio value yet (paper Section 4).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from Backbones.Generative_Pretrained.interface import GenerativeBackbone, OptionScores

from .budget import Ledger
from .mcq import canonicalize_answer

SCORING_SUFFIX = "The answer is ("


def derive_seed(*parts) -> int:
    """Deterministic 31-bit seed from arbitrary parts."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def supports_option_scores(backbone) -> bool:
    return callable(getattr(backbone, "score_next_options", None))


def score_options(
    backbone: GenerativeBackbone,
    user_message: str,
    labels: Sequence[str],
    ledger: Ledger,
    prefix_text: str = "",
) -> Optional[OptionScores]:
    """Terminal-option log-probabilities after ``prefix_text`` (or None if unavailable/unaffordable)."""
    if not supports_option_scores(backbone):
        return None
    sep = "\n" if prefix_text and not prefix_text.endswith("\n") else ""
    prompt = backbone.render_prompt(user_message, assistant_prefix=prefix_text + sep + SCORING_SUFFIX)
    est = backbone.count_tokens(sep + SCORING_SUFFIX) + 1
    if not ledger.can_score(est):
        return None
    scores = backbone.score_next_options(prompt, labels)
    # Paper-protocol accounting: with prefix caching only the scoring suffix
    # and the scored position are new work, whatever the backbone reports.
    ledger.charge_scoring(est)
    return scores


@dataclass
class RolloutOutcome:
    completed: List[Tuple[Optional[str], int, str]]  # (answer, tokens attributed, full text)
    unfinished: List[Tuple[str, int, int]]  # (prefix text, prefix length in tokens, unattributed tokens)
    decode_tokens: int


def execute_samples(
    backbone: GenerativeBackbone,
    user_message: str,
    labels: Sequence[str],
    ledger: Ledger,
    n: int,
    temperature: float,
    seed: int,
    horizon: Optional[int] = None,
    prefix_text: str = "",
    prefix_tokens: int = 0,
    prefix_unattributed: int = 0,
) -> RolloutOutcome:
    """Sample ``n`` continuations of ``prefix_text`` for up to ``horizon`` new tokens.

    The caller must have checked feasibility with ``ledger.can_sample``.

    Token attribution: decode tokens of a prefix are attributed to the
    first continuation of that prefix (index 0), so attributed tokens over
    all completed artifacts never double count. Tokens of prefixes that are
    never continued stay unattributed.
    """
    budget = ledger.budget
    room = budget.max_new_tokens - prefix_tokens
    step = room if horizon is None else min(horizon, room)
    if n <= 0 or step <= 0:
        return RolloutOutcome([], [], 0)
    prompt = backbone.render_prompt(user_message, assistant_prefix=prefix_text)
    gens = backbone.generate(prompt, n=n, max_tokens=step, temperature=temperature, seed=seed)
    new_rollouts = n - 1 if prefix_text else n
    decode = sum(g.num_tokens for g in gens)
    prompt_tokens = backbone.count_tokens(prompt)
    ledger.charge_generation(new_rollouts, decode, prompt_tokens)
    completed, unfinished = [], []
    for i, g in enumerate(gens):
        text = prefix_text + g.text
        total = prefix_tokens + g.num_tokens
        attributed = g.num_tokens + (prefix_unattributed if i == 0 else 0)
        answer = canonicalize_answer(text, labels)
        if answer is not None or g.finished or total >= budget.max_new_tokens:
            completed.append((answer, attributed, text))
        else:
            unfinished.append((text, total, attributed))
    return RolloutOutcome(completed, unfinished, decode)


__all__ = ["SCORING_SUFFIX", "derive_seed", "supports_option_scores", "score_options", "execute_samples", "RolloutOutcome"]
