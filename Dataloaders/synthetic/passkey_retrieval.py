"""Synthetic passkey-retrieval prompts for long-context evaluation.

A 5-digit key is hidden at a random depth inside repeated filler text and the
model is asked for it at the end. This is the classic "needle in a haystack"
setup (the simplest task family in RULER), generated locally so that no
dataset download is needed.

Role: benchmark / synthetic. Deterministic for a fixed seed.
The loader knows nothing about the method that consumes it.
"""

import random
from dataclasses import dataclass

FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)
NEEDLE = "The pass key is {key}. Remember it. {key} is the pass key. "
INSTRUCTION = (
    "There is important information hidden inside a lot of irrelevant text. "
    "Find it and memorize it. I will quiz you about it afterwards.\n\n"
)
QUESTION = "\n\nWhat is the pass key? Answer with the number only."


@dataclass
class PasskeySample:
    prompt: str        # user message, before any chat template
    answer: str        # the 5-digit key
    depth: float       # relative position of the needle, 0 = start, 1 = end
    n_filler: int      # number of filler repetitions used


def make_sample(n_filler, rng):
    key = str(rng.randint(10000, 99999))
    depth = rng.random()
    before = int(n_filler * depth)
    text = FILLER * before + NEEDLE.format(key=key) + FILLER * (n_filler - before)
    return PasskeySample(INSTRUCTION + text + QUESTION, key, depth, n_filler)


def fillers_for_length(tokenizer, target_tokens):
    """Number of filler repetitions that gives roughly `target_tokens` tokens."""
    per_filler = len(tokenizer(FILLER * 20, add_special_tokens=False).input_ids) / 20
    fixed = len(tokenizer(INSTRUCTION + NEEDLE.format(key="12345") + QUESTION).input_ids)
    return max(1, int((target_tokens - fixed) / per_filler))


def build_passkey_dataset(tokenizer, target_tokens=2000, num_samples=20, seed=0):
    """Returns a list of PasskeySample whose prompts are about `target_tokens`
    tokens long under `tokenizer`."""
    rng = random.Random(seed)
    n_filler = fillers_for_length(tokenizer, target_tokens)
    return [make_sample(n_filler, rng) for _ in range(num_samples)]


def is_correct(generated_text, answer):
    return answer in generated_text
