"""Prompt construction and token-by-token generation with TERMINATOR early exit.

The prompt follows the released HF inference script: chat template with
add_generation_prompt=True, then an explicit <think> token. Thinking tokens
are everything the model writes after <think> and before </think>.

`generate_with_exit` is a plain HF decoding loop with a KV cache for the base
model and a second KV cache for the probe layer, so each new token costs one
base step plus one probe-layer step. The probe prediction for thinking token
i is computed in the same forward that consumes token i, before token i+1 is
sampled, as in the released code.
"""

from __future__ import annotations

import time
from typing import List, Optional

import torch

# Exit message used by the released code before forcing </think>.
EXIT_MESSAGE = "\nI've run out of thinking tokens. I need to commit to a final answer."

PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def special_ids(tokenizer):
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_id is None or think_end_id is None or think_id == tokenizer.unk_token_id:
        raise ValueError("tokenizer has no <think>/</think> tokens")
    return think_id, think_end_id


def build_prompt_ids(tokenizer, problem: str) -> List[int]:
    """Token ids of the chat prompt, ending with the <think> token."""
    messages = [{"role": "user", "content": problem + PROMPT_SUFFIX}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    think_id, _ = special_ids(tokenizer)
    return list(ids) + [think_id]


def exit_ids(tokenizer, message: str = EXIT_MESSAGE) -> List[int]:
    """Tokens forced after an early exit: the exit message, then </think>."""
    _, think_end_id = special_ids(tokenizer)
    msg = tokenizer(message, add_special_tokens=False)["input_ids"] if message else []
    return list(msg) + [think_end_id]


def sample_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int,
                 generator: Optional[torch.Generator]) -> int:
    """Sample one id from logits [V] with temperature, top-k and top-p.
    temperature <= 0 means greedy."""
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits.float() / temperature
    if top_k and top_k > 0:
        vals, idx = torch.topk(logits, min(top_k, logits.numel()))
    else:
        vals, idx = torch.sort(logits, descending=True)
    probs = torch.softmax(vals, dim=-1)
    if top_p < 1.0:
        cum = torch.cumsum(probs, dim=-1)
        keep = cum - probs < top_p
        keep[0] = True
        probs = probs * keep
        probs = probs / probs.sum()
    j = torch.multinomial(probs, 1, generator=generator)
    return int(idx[j])


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _base_step(model, ids, cache, device):
    x = torch.tensor([ids], device=device)
    out = model(input_ids=x, past_key_values=cache, use_cache=True)
    return out.logits[0, -1]


@torch.no_grad()
def generate_with_exit(
    model,
    tokenizer,
    prompt_ids: List[int],
    probe=None,
    policy=None,
    max_think: int = 6000,
    max_answer: int = 1024,
    budget: Optional[int] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int = 0,
    exit_message: str = EXIT_MESSAGE,
    time_probe: bool = False,
) -> dict:
    """Generate thinking then answer for one prompt (batch size 1).

    probe: TerminatorProbe or None. policy: object with reset/should_stop/report,
    or None for no early stopping. budget: max thinking tokens passed to the
    policy; 0 skips thinking entirely. Thinking is always capped at max_think.
    """
    from transformers import DynamicCache

    from .probe import LastLayerHook

    device = next(model.parameters()).device
    _, think_end_id = special_ids(tokenizer)
    eos_ids = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id}
    gen = torch.Generator(device=device).manual_seed(seed)
    if policy is not None:
        policy.reset({"seed": seed})

    hook = LastLayerHook(model)
    base_cache = DynamicCache()
    probe_cache = DynamicCache() if probe is not None else None
    probe_dtype = next(probe.parameters()).dtype if probe is not None else None
    probe_s = 0.0
    probs: List[float] = []

    def think_step(ids: List[int], pos: int):
        nonlocal probe_s
        logits = _base_step(model, ids, base_cache, device)
        if probe is None:
            return logits, None
        if time_probe:
            _sync(device)
            t = time.perf_counter()
        logit = probe(hook.hidden.to(probe_dtype), past_key_values=probe_cache,
                      position_offset=pos)
        p = float(torch.sigmoid(logit[0, -1].float()))
        if time_probe:
            _sync(device)
            probe_s += time.perf_counter() - t
        return logits, p

    try:
        _sync(device)
        t0 = time.perf_counter()
        logits, _ = think_step(list(prompt_ids), 0)
        pos = len(prompt_ids)
        think: List[int] = []
        end = "cap"
        if budget is not None and budget <= 0:
            end = "budget"
        while end == "cap" and len(think) < max_think:
            tok = sample_token(logits, temperature, top_p, top_k, gen)
            if tok == think_end_id or tok in eos_ids:
                end = "natural"
                break
            think.append(tok)
            logits, p = think_step([tok], pos)
            pos += 1
            if p is not None:
                probs.append(p)
            if policy is not None:
                metrics = {} if p is None else {"terminator_prob": p}
                if policy.should_stop({"thinking_tokens": len(think)}, metrics, budget):
                    end = policy.report().get("reason") or "policy"
        _sync(device)
        t_think = time.perf_counter() - t0

        tail = [think_end_id] if end == "natural" else exit_ids(tokenizer, exit_message)
        logits = _base_step(model, tail, base_cache, device)
        answer: List[int] = []
        while len(answer) < max_answer:
            tok = sample_token(logits, temperature, top_p, top_k, gen)
            if tok in eos_ids:
                break
            answer.append(tok)
            logits = _base_step(model, [tok], base_cache, device)
        _sync(device)
        t_total = time.perf_counter() - t0
    finally:
        hook.remove()

    early = end not in ("natural", "cap")
    return {
        "think_ids": think,
        "answer_ids": answer,
        "answer_text": tokenizer.decode(answer, skip_special_tokens=True),
        "think_end": end,
        "stopped_early": early,
        "exit_pos": len(think) if early else None,
        "probs": probs,
        "think_s": t_think,
        "total_s": t_total,
        "probe_s": probe_s,
        "policy_report": policy.report() if policy is not None else None,
    }
