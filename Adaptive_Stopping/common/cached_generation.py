"""Batched sampling of full (unstopped) reasoning traces, cached for offline replay.

The paper applies ST-ES offline: it runs standard inference once and replays
the stopping rule on the saved traces. This module produces those traces with
a cheap form of continuous batching that fits a single 8 GB GPU:

* jobs are grouped by current context length, and each round generates at most
  ``chunk_tokens`` new tokens for a batch;
* a round ends early once ``min_done_frac`` of its rows have emitted EOS, and
  unfinished rows are re-queued with their partial output as context, so short
  traces do not wait for long ones.

Continuing a sampled sequence from its own prefix draws from the same
distribution as sampling it in one call, so this changes throughput, not the
model's output distribution. Exact token streams depend on the batch schedule
(seeds are set per round); the schedule is deterministic for fixed inputs.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Iterable

import torch


@dataclasses.dataclass
class GenJob:
    uid: str
    prompt_ids: list[int]
    gen_ids: list[int] = dataclasses.field(default_factory=list)
    finish_reason: str | None = None  # "eos" | "length"
    rounds: int = 0

    @property
    def context_len(self) -> int:
        return len(self.prompt_ids) + len(self.gen_ids)


class _RoundStop:
    """Per-row EOS tracking; ends the whole round once enough rows are done."""

    def __init__(self, eos_ids: Iterable[int], min_done_frac: float, start_len: int):
        self.eos = torch.tensor(sorted(set(eos_ids)))
        self.min_done_frac = min_done_frac
        self.start_len = start_len
        self.done = None

    def __call__(self, input_ids: torch.Tensor, scores=None, **kwargs) -> torch.Tensor:
        last = input_ids[:, -1]
        hit = torch.isin(last, self.eos.to(last.device))
        self.done = hit if self.done is None else (self.done | hit)
        if input_ids.shape[1] > self.start_len and self.done.float().mean().item() >= self.min_done_frac:
            return torch.ones_like(self.done)
        return self.done.clone()


def _take_batch(pending: list[GenJob], max_batch: int, kv_budget: int, chunk: int) -> list[GenJob]:
    batch: list[GenJob] = []
    longest = 0
    for job in pending:
        longest_new = max(longest, job.context_len)
        n = len(batch) + 1
        if batch and (n > max_batch or n * (longest_new + chunk) > kv_budget):
            break
        batch.append(job)
        longest = longest_new
    return batch


def _prefill_rows(model, ctxs: list[list[int]], width: int, device, max_new: int, prefill_chunk: int = 2048):
    """KV cache for a left-padded batch, built by prefilling each row on its own.

    Row r occupies cache slots ``[width - len(ctx_r), width - 1)`` and holds the
    KV of all but its last token, which ``generate`` feeds as the first
    uncached input. Prefilling one row at a time keeps peak memory independent
    of the decode batch size. Each row is prefilled without padding, so its
    RoPE positions are 0..len-2, matching the positions ``generate`` derives
    from the attention mask for the following tokens.

    The result is a ``StaticCache`` sized for the whole round
    (``width - 1 + max_new + 1`` slots), so decoding writes in place instead of
    concatenating a larger tensor every step (which fragments the allocator
    and, on Windows, spills into shared system memory).
    """
    from transformers import StaticCache

    cfg = model.config
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    cache = StaticCache(config=cfg, max_cache_len=width + max_new)
    # Allocate each layer's static buffer (zeros) and mark the first width-1 slots as filled; rows are then
    # written into those buffers in place, so no batch-sized temporary copy is ever held next to the cache.
    for li in range(cfg.num_hidden_layers):
        z = torch.zeros((len(ctxs), n_kv, width - 1, head_dim), dtype=model.dtype, device=device)
        cache.update(z, z, li)
        del z
    from transformers import DynamicCache

    decoder = model.get_decoder()
    for r, c in enumerate(ctxs):
        ids = torch.tensor([c[:-1]], device=device)
        n = ids.shape[1]
        row_cache = DynamicCache()
        for s in range(0, n, prefill_chunk):  # chunked prefill bounds activation memory for long rows
            decoder(input_ids=ids[:, s:s + prefill_chunk], past_key_values=row_cache, use_cache=True)
        for li, layer in enumerate(row_cache.layers):
            cache.layers[li].keys[r, :, width - 1 - n:width - 1] = layer.keys[0]
            cache.layers[li].values[r, :, width - 1 - n:width - 1] = layer.values[0]
        del row_cache, ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # drop the per-row prefill temporaries before decoding (fragmentation)
    return cache


@torch.inference_mode()
def generate_full_traces(
    model,
    jobs: list[GenJob],
    *,
    eos_ids: list[int],
    pad_id: int,
    max_new_tokens: int,
    seed: int,
    sampling: dict,
    chunk_tokens: int = 1024,
    min_done_frac: float = 0.75,
    max_batch: int = 32,
    kv_token_budget: int = 40_000,
    on_finish: Callable[[GenJob], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Generate every job to EOS or ``max_new_tokens``. Returns run statistics.

    Jobs are served shortest-context first so fresh prompts run in full
    batches; partially generated jobs are re-queued and grouped by length.
    """
    device = next(model.parameters()).device
    eos_set = set(eos_ids)
    pending = [j for j in jobs if j.finish_reason is None]
    stats = {"rounds": 0, "generated_tokens": 0, "decode_steps": 0, "wall_s": 0.0, "max_batch_used": 0}
    t0 = time.perf_counter()
    while pending:
        pending.sort(key=lambda j: j.context_len)
        batch = _take_batch(pending, max_batch, kv_token_budget, chunk_tokens)
        pending = pending[len(batch):]
        ctx = [j.prompt_ids + j.gen_ids for j in batch]
        width = max(len(c) for c in ctx)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), width), dtype=torch.long)
        for i, c in enumerate(ctx):
            input_ids[i, width - len(c):] = torch.tensor(c)
            attn[i, width - len(c):] = 1
        remaining = [max_new_tokens - len(j.gen_ids) for j in batch]
        step_new = min(chunk_tokens, max(remaining))
        stopper = _RoundStop(eos_ids, min_done_frac, width)
        cache = _prefill_rows(model, ctx, width, device, step_new)
        torch.manual_seed(seed * 1_000_003 + stats["rounds"])
        out = model.generate(
            input_ids=input_ids.to(device), attention_mask=attn.to(device), max_new_tokens=step_new, cache=cache,
            eos_token_id=list(eos_ids), pad_token_id=pad_id, stopping_criteria=[stopper], disable_compile=True,
            **sampling,
        )
        del cache
        new = out[:, width:].tolist()
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # release fragments of the grown KV cache between rounds
        stats["rounds"] += 1
        stats["decode_steps"] += len(new[0]) if new else 0
        stats["max_batch_used"] = max(stats["max_batch_used"], len(batch))
        for job, toks, rem in zip(batch, new, remaining):
            job.rounds += 1
            cut = next((k for k, t in enumerate(toks) if t in eos_set), None)
            if cut is not None:
                job.gen_ids.extend(toks[:cut])
                job.finish_reason = "eos"
                stats["generated_tokens"] += cut + 1
            else:
                toks = toks[:rem]
                job.gen_ids.extend(toks)
                stats["generated_tokens"] += len(toks)
                if len(job.gen_ids) >= max_new_tokens:
                    job.finish_reason = "length"
            if job.finish_reason is None:
                pending.append(job)
            elif on_finish is not None:
                on_finish(job)
        elapsed = time.perf_counter() - t0
        done = sum(j.finish_reason is not None for j in jobs)
        mem = ""
        if torch.cuda.is_available():
            mem = (f" | peak alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GB"
                   f" reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
        log(f"round {stats['rounds']}: batch={len(batch)} ctx<={width} +{len(new[0]) if new else 0} tok | "
            f"done {done}/{len(jobs)} | {stats['generated_tokens'] / max(elapsed, 1e-9):.0f} tok/s | {elapsed:.0f}s{mem}")
    stats["wall_s"] = time.perf_counter() - t0
    stats["tokens_per_s"] = stats["generated_tokens"] / max(stats["wall_s"], 1e-9)
    return stats
