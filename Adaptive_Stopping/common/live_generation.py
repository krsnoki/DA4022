"""Online generation with a step-level stopping policy (batch size 1).

Implements Algorithm 1 of the paper as a live loop: the policy is evaluated
inside ``model.generate`` as a stopping criterion, at the token that closes
each reasoning step (``"\\n\\n"``). When it fires, generation stops, the exit
prompt is appended, and up to ``answer_budget`` answer tokens are decoded
greedily, reusing the KV cache of the trace.

The segmentation and policy calls are the same as in offline replay
(``common.steps`` / ``common.replay``), so with identical tokens both paths stop
at the same step. The time spent in the policy (tagging included) is measured
separately as its overhead.
"""

from __future__ import annotations

import time
from typing import Sequence

import torch

from Adaptive_Stopping.common.forced_answer import ANSWER_BUDGET, EXIT_PROMPTS, BoxCloseCriteria
from Adaptive_Stopping.common.policy import StepState, StoppingPolicy
from Adaptive_Stopping.common.steps import IncrementalDetokenizer, StepSegmenter


class StepPolicyCriteria:
    """HF stopping criterion that feeds completed steps to a ``StoppingPolicy`` (batch size 1)."""

    def __init__(self, policy: StoppingPolicy, tokenizer, prompt_len: int, budget: dict | None = None):
        self.policy = policy
        self.policy.reset()
        self.detok = IncrementalDetokenizer(tokenizer)
        self.seg = StepSegmenter()
        self.prompt_len = prompt_len
        self.budget = budget
        self.n = 0
        self.stopped = False
        self.stop_step: int | None = None
        self.cut: int | None = None
        self.policy_time_s = 0.0
        self.n_steps = 0

    def consume(self, token_ids: Sequence[int]) -> bool:
        for t in token_ids:
            self.n += 1
            for step in self.seg.feed(self.detok.push(t), self.n):
                self.n_steps += 1
                if self.stopped:
                    continue
                t0 = time.perf_counter()
                stop = self.policy.should_stop(
                    StepState(step.index, step.text, step.end_token, step.in_thinking), None, self.budget)
                self.policy_time_s += time.perf_counter() - t0
                if stop:
                    self.stopped, self.stop_step, self.cut = True, step.index, self.n
        return self.stopped

    def __call__(self, input_ids: torch.Tensor, scores=None, **kwargs) -> torch.Tensor:
        if input_ids.shape[0] != 1:
            raise ValueError("StepPolicyCriteria supports batch size 1")
        new = input_ids[0, self.prompt_len + self.n:].tolist()
        stop = self.consume(new)
        return torch.tensor([stop], device=input_ids.device)


class _FirstTokenTimer:
    """Minimal streamer: records the time of the first generated token (time to first token)."""

    def __init__(self):
        self.calls = 0
        self.t_first: float | None = None

    def put(self, value):
        self.calls += 1
        if self.calls == 2 and self.t_first is None:  # first call carries the prompt
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.t_first = time.perf_counter()

    def end(self):
        pass


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def continue_with_exit_prompt(model, tokenizer, out, *, eos_ids: Sequence[int], pad_id: int,
                              exit_prompt_id: str = "exit_boxed.v1",
                              answer_budget: int = ANSWER_BUDGET) -> list[int]:
    """Forced answer after a stop, reusing the KV cache of ``out`` (a ``generate`` result with
    ``return_dict_in_generate=True``). Falls back to a fresh prefill when no cache is returned."""
    device = out.sequences.device
    exit_ids = tokenizer(EXIT_PROMPTS[exit_prompt_id], add_special_tokens=False)["input_ids"]
    full = torch.cat([out.sequences, torch.tensor([exit_ids], device=device)], dim=1)
    cache = getattr(out, "past_key_values", None)
    box = BoxCloseCriteria(tokenizer, eos_ids, full.shape[1], 1)
    kwargs = {"cache": cache} if cache is not None else {}
    out2 = model.generate(input_ids=full, attention_mask=torch.ones_like(full), max_new_tokens=answer_budget,
                          do_sample=False, temperature=None, top_p=None, top_k=None, stopping_criteria=[box],
                          eos_token_id=list(eos_ids), pad_token_id=pad_id, **kwargs)
    new = out2[0, full.shape[1]:].tolist()
    stop = box.stop_len[0]
    return new[:stop] if stop is not None else new[:answer_budget]


@torch.inference_mode()
def generate_with_policy(model, tokenizer, prompt_ids: Sequence[int], policy: StoppingPolicy | None, *,
                         max_new_tokens: int, sampling: dict, seed: int, eos_ids: Sequence[int], pad_id: int,
                         exit_prompt_id: str = "exit_boxed.v1", answer_budget: int = ANSWER_BUDGET,
                         budget: dict | None = None) -> dict:
    """One request: generate under ``policy``; force the answer if it stopped. Returns tokens and timings.

    ``policy=None`` is plain standard generation with no monitoring at all (the baseline).
    """
    device = next(model.parameters()).device
    eos = set(eos_ids)
    ids = torch.tensor([list(prompt_ids)], device=device)
    crit = StepPolicyCriteria(policy, tokenizer, len(prompt_ids), budget) if policy is not None else None
    timer = _FirstTokenTimer()
    torch.manual_seed(seed)
    _sync()
    t0 = time.perf_counter()
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         stopping_criteria=[crit] if crit else None, streamer=timer, return_dict_in_generate=True,
                         eos_token_id=list(eos_ids), pad_token_id=pad_id, **sampling)
    _sync()
    t1 = time.perf_counter()
    seq = out.sequences
    gen = seq[0, len(prompt_ids):].tolist()
    finish = "length"
    if gen and gen[-1] in eos:
        gen, finish = gen[:-1], "eos"
    answer_ids: list[int] = []
    t2 = t1
    stopped = bool(crit and crit.stopped)
    if stopped:
        finish = "policy"
        answer_ids = continue_with_exit_prompt(model, tokenizer, out, eos_ids=eos_ids, pad_id=pad_id,
                                               exit_prompt_id=exit_prompt_id, answer_budget=answer_budget)
        _sync()
        t2 = time.perf_counter()
    return {
        "gen_ids": gen, "finish_reason": finish, "stopped": stopped, "stop_step": crit.stop_step if crit else None,
        "cut": crit.cut if crit else None, "answer_ids": answer_ids, "n_generated": len(gen) + len(answer_ids),
        "latency_s": t2 - t0, "ttft_s": (timer.t_first - t0) if timer.t_first else None,
        "reasoning_s": t1 - t0, "answer_s": t2 - t1, "policy_time_s": crit.policy_time_s if crit else 0.0,
        "steps_seen": crit.n_steps if crit else None,
        "policy_report": policy.report() if policy is not None else {"policy": "standard"},
    }
