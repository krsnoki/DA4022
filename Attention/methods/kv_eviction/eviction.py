"""KV-cache eviction for Hugging Face decoder-only models (Llama / Qwen2 style).

Flow:
    1. run the prompt once (prefill) while recording, for every layer, the
       queries of the last `window` tokens,
    2. score every cached position per head and keep the best `budget` fraction,
    3. decode greedily on the shortened cache.

The cache tensors are [B, H_kv, N, D] and rotary position information is
already inside the keys, so positions can be dropped per head by a plain
gather. The only thing to take care of afterwards is that new tokens still
get their original position ids, because the cache is now shorter than the
prompt.

Limitations: batch size 1 (no padding handling), models whose attention
layers receive `position_embeddings=(cos, sin)` and expose `q_proj`.
"""

import torch

from .scoring import METHODS, gather_positions, select_keep_indices


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class QueryRecorder:
    """Records the rotated queries of the last `window` positions in each
    attention layer during one forward pass."""

    def __init__(self, model, window):
        self.window = window
        self.queries = {}
        self.handles = []
        layers = model.model.layers
        for i, layer in enumerate(layers):
            hook = layer.self_attn.register_forward_pre_hook(
                self._make_hook(i, model.config.num_attention_heads), with_kwargs=True
            )
            self.handles.append(hook)

    def _make_hook(self, layer_idx, n_heads):
        def hook(module, args, kwargs):
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            pos_emb = kwargs.get("position_embeddings")
            if pos_emb is None:
                raise RuntimeError(
                    "attention layer did not receive position_embeddings; "
                    "this transformers version / model is not supported"
                )
            cos, sin = pos_emb
            w = min(self.window, hidden.shape[1])
            b = hidden.shape[0]
            q = module.q_proj(hidden[:, -w:])
            q = q.view(b, w, n_heads, -1).transpose(1, 2)           # [B, Hq, W, D]
            cos, sin = cos[:, -w:].unsqueeze(1), sin[:, -w:].unsqueeze(1)
            q = q * cos + _rotate_half(q) * sin
            self.queries[layer_idx] = q.detach()

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def group_queries(q, n_kv_heads):
    """With grouped-query attention several query heads share one KV head.
    Fold them into extra window queries of that KV head.

    q : [B, Hq, W, D] -> [B, Hkv, G*W, D], plus the window offset of every
    folded query (needed for the causal mask).
    """
    b, hq, w, d = q.shape
    g = hq // n_kv_heads
    q = q.reshape(b, n_kv_heads, g * w, d)
    offsets = torch.arange(w, device=q.device).repeat(g)
    return q, offsets


def _get_kv(cache, i):
    if hasattr(cache, "layers"):
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


def _set_kv(cache, i, k, v):
    if hasattr(cache, "layers"):
        cache.layers[i].keys, cache.layers[i].values = k, v
    else:
        cache.key_cache[i], cache.value_cache[i] = k, v


def cache_num_layers(cache):
    return len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)


def cache_nbytes(cache):
    total = 0
    for i in range(cache_num_layers(cache)):
        k, v = _get_kv(cache, i)
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


def compress_cache(cache, queries, method="dropkv", budget=0.1, window=8, kernel_size=11, generator=None):
    """Evict positions from every layer of `cache` in place.

    queries : {layer index: [B, Hq, W, D]} from QueryRecorder.
    budget  : fraction of positions kept per head.
    Returns the number of positions kept per head.
    """
    score_fn = METHODS[method]
    n_keep = None
    for i in range(cache_num_layers(cache)):
        k, v = _get_kv(cache, i)
        n = k.shape[2]
        n_keep = max(int(round(n * budget)), min(window, n))
        if n_keep >= n:
            return n
        q, offsets = group_queries(queries[i], k.shape[1])
        w = queries[i].shape[2]
        q_pos = n - w + offsets
        if method == "random":
            scores = score_fn(q, k, v, generator=generator)
        else:
            scores = score_fn(q, k, v, q_pos=q_pos)
        keep = select_keep_indices(scores, n_keep, window, kernel_size, pool=(method != "random"))
        _set_kv(cache, i, gather_positions(k, keep), gather_positions(v, keep))
    return n_keep


@torch.no_grad()
def prefill(model, input_ids, method="full", budget=1.0, window=8, kernel_size=11, generator=None):
    """Run the prompt and return (logits of the last position, cache, info).

    method="full" keeps the whole cache and is the unmodified baseline.
    """
    recorder = QueryRecorder(model, window) if method != "full" else None
    try:
        out = model(input_ids=input_ids, use_cache=True)
    finally:
        if recorder is not None:
            recorder.remove()
    cache = out.past_key_values
    n = input_ids.shape[1]
    info = {"prompt_len": n, "kv_bytes_full": cache_nbytes(cache), "kept": n}
    if recorder is not None:
        info["kept"] = compress_cache(cache, recorder.queries, method, budget, window, kernel_size, generator)
    info["kv_bytes"] = cache_nbytes(cache)
    return out.logits[:, -1], cache, info


@torch.no_grad()
def decode_step(model, token, cache, position, cache_len):
    """One greedy step. `position` is the true position of `token` in the
    sequence, `cache_len` the current (possibly shortened) cache length."""
    device = token.device
    out = model(
        input_ids=token,
        past_key_values=cache,
        position_ids=torch.tensor([[position]], device=device),
        attention_mask=torch.ones(1, cache_len + 1, dtype=torch.long, device=device),
        use_cache=True,
    )
    return out.logits[:, -1], out.past_key_values


@torch.no_grad()
def generate(model, input_ids, max_new_tokens=16, eos_token_id=None, **prefill_kwargs):
    """Greedy generation with an optionally compressed prompt cache.
    Returns (new token ids [T], info)."""
    logits, cache, info = prefill(model, input_ids, **prefill_kwargs)
    position, cache_len = info["prompt_len"], info["kept"]
    tokens = []
    for _ in range(max_new_tokens):
        token = logits.argmax(dim=-1, keepdim=True)
        tokens.append(token.item())
        if eos_token_id is not None and token.item() == eos_token_id:
            break
        logits, cache = decode_step(model, token, cache, position, cache_len)
        position += 1
        cache_len += 1
    return tokens, info
