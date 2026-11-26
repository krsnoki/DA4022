"""Answer forcing after an early stop.

Paper, Sec. 3: "we prompted the models right after early-stopping, and allowed
an additional budget of 100 tokens", following s1 (Muennighoff et al., 2025).
The exact exit prompt is not given in the paper; ``exit_boxed.v1`` is our
choice: close the thinking block and open a boxed final answer. Decoding of
the answer is greedy and stops at EOS, when the ``\\boxed{`` brace closes, or
after ``ANSWER_BUDGET`` tokens.
"""

from __future__ import annotations

from typing import Sequence

import torch

EXIT_PROMPTS = {
    "exit_boxed.v1": "</think>\n\n**Final Answer**\n\\boxed{",
}
ANSWER_BUDGET = 100


def brace_delta(piece: str) -> int:
    piece = piece.replace("\\{", "").replace("\\}", "")
    return piece.count("{") - piece.count("}")


class BoxCloseCriteria:
    """Per-row stop once the brace opened by the exit prompt is closed (or EOS)."""

    def __init__(self, tokenizer, eos_ids: Sequence[int], start_len: int, batch_size: int):
        self.tok = tokenizer
        self.eos = set(eos_ids)
        self.start_len = start_len
        self.depth = [1] * batch_size
        self.stop_len: list[int | None] = [None] * batch_size

    def __call__(self, input_ids: torch.Tensor, scores=None, **kwargs) -> torch.Tensor:
        n_new = input_ids.shape[1] - self.start_len
        last = input_ids[:, -1].tolist()
        for i, t in enumerate(last):
            if self.stop_len[i] is not None:
                continue
            if t in self.eos:
                self.stop_len[i] = n_new - 1  # EOS itself is not part of the answer
                continue
            self.depth[i] += brace_delta(self.tok.decode([t]))
            if self.depth[i] <= 0:
                self.stop_len[i] = n_new
        return torch.tensor([s is not None for s in self.stop_len], device=input_ids.device)


def _batches(order: list[int], lengths: list[int], max_batch: int, kv_budget: int, prefill_budget: int,
             extra: int) -> list[list[int]]:
    out, cur, longest = [], [], 0
    for i in order:
        ln = lengths[i]
        n = len(cur) + 1
        lw = max(longest, ln)
        if cur and (n > max_batch or n * (lw + extra) > kv_budget or n * lw > prefill_budget):
            out.append(cur)
            cur, longest = [], 0
            lw = ln
        cur.append(i)
        longest = lw
    if cur:
        out.append(cur)
    return out


@torch.inference_mode()
def force_answers(model, tokenizer, contexts: Sequence[Sequence[int]], *, exit_prompt_id: str = "exit_boxed.v1",
                  eos_ids: Sequence[int], pad_id: int, max_new_tokens: int = ANSWER_BUDGET, max_batch: int = 32,
                  kv_token_budget: int = 110_000, prefill_token_budget: int = 40_000, log=None) -> list[list[int]]:
    """For each context (prompt + truncated trace token IDs), append the exit prompt and decode the answer.

    Returns the generated answer token IDs per context (exit prompt not included).
    """
    exit_ids = tokenizer(EXIT_PROMPTS[exit_prompt_id], add_special_tokens=False)["input_ids"]
    device = next(model.parameters()).device
    full = [list(c) + exit_ids for c in contexts]
    lengths = [len(f) for f in full]
    order = sorted(range(len(full)), key=lambda i: lengths[i], reverse=True)
    results: list[list[int] | None] = [None] * len(full)
    batches = _batches(order, lengths, max_batch, kv_token_budget, prefill_token_budget, max_new_tokens)
    for bi, idx in enumerate(batches):
        width = max(lengths[i] for i in idx)
        ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        att = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, width - lengths[i]:] = torch.tensor(full[i])
            att[r, width - lengths[i]:] = 1
        crit = BoxCloseCriteria(tokenizer, eos_ids, width, len(idx))
        out = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None, top_k=None, eos_token_id=list(eos_ids),
                             pad_token_id=pad_id, stopping_criteria=[crit])
        new = out[:, width:].tolist()
        for r, i in enumerate(idx):
            stop = crit.stop_len[r]
            toks = new[r][:stop] if stop is not None else new[r][:max_new_tokens]
            results[i] = [t for t in toks]
        if log:
            log(f"forced answers: batch {bi + 1}/{len(batches)} size={len(idx)} ctx<={width}")
    return results  # type: ignore[return-value]


@torch.inference_mode()
def force_answers_shared_prefix(model, tokenizer, full_ids: Sequence[int], cut_lens: Sequence[int], *,
                                exit_prompt_id: str = "exit_boxed.v1", eos_ids: Sequence[int], pad_id: int,
                                max_new_tokens: int = ANSWER_BUDGET, kv_token_budget: int = 40_000,
                                max_batch: int = 32, prefill_chunk: int = 1024) -> dict[int, list[int]]:
    """Forced answers for many cut points of *one* trace, reusing a single prefill.

    ``full_ids`` is prompt + full generated trace; each ``cut_len`` is a context
    length (prompt + kept trace tokens). With causal attention the KV cache of a
    prefix does not depend on later tokens, so one forward pass over the longest
    needed prefix gives the cache of every shorter one; each row of a batch gets
    its prefix slice, left-padded. Greedy decoding then matches ``force_answers``
    up to floating-point differences (checked in the tests on a tiny model).
    Each batch decodes into a preallocated ``StaticCache`` (no per-step
    concatenation, so long prefixes do not fragment GPU memory).
    """
    from transformers import StaticCache

    exit_ids = tokenizer(EXIT_PROMPTS[exit_prompt_id], add_special_tokens=False)["input_ids"]
    device = next(model.parameters()).device
    cuts = sorted(set(int(c) for c in cut_lens), reverse=True)
    if not cuts:
        return {}
    from transformers import DynamicCache

    longest = cuts[0]
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    ids = torch.tensor([list(full_ids[:longest])], device=device)
    prefix_cache = DynamicCache()
    for s in range(0, ids.shape[1], prefill_chunk):
        # chunked prefill: without flash attention (Windows) SDPA may materialise the full attention matrix,
        # which for an 8k-token trace no longer fits next to the model on an 8 GB GPU
        base(input_ids=ids[:, s:s + prefill_chunk], past_key_values=prefix_cache, use_cache=True)
    layers = [(layer.keys, layer.values) for layer in prefix_cache.layers]
    del prefix_cache, ids
    results: dict[int, list[int]] = {}
    i = 0
    while i < len(cuts):
        width = cuts[i]
        n = max(1, min(max_batch, kv_token_budget // (width + len(exit_ids) + max_new_tokens)))
        chunk = cuts[i:i + n]
        i += n
        total = width + len(exit_ids)
        cache = StaticCache(config=model.config, max_cache_len=total + max_new_tokens)
        for li, (k, v) in enumerate(layers):
            kb = k.new_zeros((len(chunk), k.shape[1], width, k.shape[3]))
            vb = v.new_zeros((len(chunk), v.shape[1], width, v.shape[3]))
            for r, ln in enumerate(chunk):
                kb[r, :, width - ln:] = k[0, :, :ln]
                vb[r, :, width - ln:] = v[0, :, :ln]
            cache.update(kb, vb, li)
            del kb, vb
        ids = torch.full((len(chunk), total), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), total), dtype=torch.long)
        for r, ln in enumerate(chunk):
            row = list(full_ids[:ln]) + exit_ids
            ids[r, total - len(row):] = torch.tensor(row)
            att[r, total - len(row):] = 1
        crit = BoxCloseCriteria(tokenizer, eos_ids, total, len(chunk))
        gen = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), cache=cache,
                             max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                             eos_token_id=list(eos_ids), pad_token_id=pad_id, stopping_criteria=[crit],
                             disable_compile=True)
        new = gen[:, total:].tolist()
        for r, ln in enumerate(chunk):
            stop = crit.stop_len[r]
            results[ln] = new[r][:stop] if stop is not None else new[r][:max_new_tokens]
        del cache, gen
        if torch.cuda.is_available():
            # Every batch allocates differently sized buffers; without this the caching allocator keeps them all
            # reserved (observed 7.8 GB dedicated + 5 GB spilled to shared memory on Windows).
            torch.cuda.empty_cache()
    del layers
    return results


def answer_text(tokenizer, answer_ids: Sequence[int], exit_prompt_id: str = "exit_boxed.v1") -> str:
    """Text scored for a forced answer: the exit prompt after ``</think>`` plus the generated answer."""
    prompt = EXIT_PROMPTS[exit_prompt_id].split("</think>", 1)[-1]
    return prompt + tokenizer.decode(list(answer_ids), skip_special_tokens=True)
