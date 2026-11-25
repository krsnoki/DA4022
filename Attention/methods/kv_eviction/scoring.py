"""Token scoring for KV-cache eviction.

Shapes used everywhere in this file:
    q : [B, H, W, D]  queries of the observation window (last W prompt tokens)
    k : [B, H, N, D]  cached keys
    v : [B, H, N, D]  cached values
    returned scores : [B, H, N]

All scores are returned so that a larger value means the token matters more.
For DropKV the score is the cost of evicting the token, so evicting the k
smallest costs is the same as keeping the largest ones.
"""

import torch
import torch.nn.functional as F


def attention_probs(q, k, q_pos=None):
    """Softmax attention of the window queries over the cached keys.

    q_pos[t] is the position in the prompt of query t. A query cannot attend
    to keys after itself, so those logits are masked. By default the queries
    are assumed to be the last W positions.
    """
    n, w, d = k.shape[-2], q.shape[-2], q.shape[-1]
    if q_pos is None:
        q_pos = torch.arange(n - w, n, device=q.device)
    logits = torch.matmul(q, k.transpose(-1, -2)) / d**0.5
    key_pos = torch.arange(n, device=q.device)
    future = key_pos[None, :] > q_pos[:, None]
    logits = logits.masked_fill(future, float("-inf"))
    return torch.softmax(logits, dim=-1)


def dropkv_scores(q, k, v, q_pos=None, eps=1e-6):
    """DropKV score: sum over window queries of ||Delta({j})||^2.

        score_j = sum_t (p_tj / (1 - p_tj))^2 * ||a_t - v_j||^2

    The squared distance is expanded as ||a||^2 + ||v||^2 - 2<a, v>, so the
    largest temporary is [B, H, W, N] and the [W, N, D] residual tensor is
    never built.
    """
    q, k, v = q.float(), k.float(), v.float()
    p = attention_probs(q, k, q_pos)                        # [B, H, W, N]
    a = torch.matmul(p, v)                                  # [B, H, W, D]
    a_sq = (a * a).sum(-1, keepdim=True)                    # [B, H, W, 1]
    v_sq = (v * v).sum(-1).unsqueeze(-2)                    # [B, H, 1, N]
    dist_sq = a_sq + v_sq - 2.0 * torch.matmul(a, v.transpose(-1, -2))
    dist_sq = dist_sq.clamp_min(0.0)
    weight = (p / (1.0 - p + eps)) ** 2
    return (weight * dist_sq).sum(dim=-2)


def dropkv_scores_naive(q, k, v, q_pos=None, eps=1e-6):
    """Same score written directly from the formula. Builds the full
    [B, H, W, N, D] residual tensor, so only use it for small inputs or as a
    reference in tests and in the scoring-cost comparison."""
    q, k, v = q.float(), k.float(), v.float()
    p = attention_probs(q, k, q_pos)
    a = torch.matmul(p, v)
    residual = a.unsqueeze(-2) - v.unsqueeze(-3)            # [B, H, W, N, D]
    dist_sq = (residual * residual).sum(-1)
    weight = (p / (1.0 - p + eps)) ** 2
    return (weight * dist_sq).sum(dim=-2)


def snapkv_scores(q, k, v=None, q_pos=None):
    """SnapKV: attention mass a token receives from the window. Values unused."""
    p = attention_probs(q.float(), k.float(), q_pos)
    return p.sum(dim=-2)


def random_scores(q, k, v=None, q_pos=None, generator=None):
    b, h, n, _ = k.shape
    return torch.rand(b, h, n, generator=generator).to(k.device)


METHODS = {
    "dropkv": dropkv_scores,
    "snapkv": snapkv_scores,
    "random": random_scores,
}


def select_keep_indices(scores, n_keep, window, kernel_size=11, pool=True):
    """Pick which positions stay in the cache, separately for each head.

    scores : [B, H, N]. Scores are max-pooled over neighbouring positions (as
    in SnapKV) so that tokens around an important one survive with it. The
    last `window` positions are always kept. Returns sorted indices
    [B, H, n_keep].
    """
    b, h, n = scores.shape
    n_keep = min(n, max(n_keep, min(window, n)))
    if pool and kernel_size > 1:
        scores = F.max_pool1d(
            scores.reshape(b * h, 1, n), kernel_size, stride=1, padding=kernel_size // 2
        ).reshape(b, h, n)
    scores = scores.clone()
    scores[..., n - min(window, n):] = float("inf")
    keep = scores.topk(n_keep, dim=-1).indices
    return keep.sort(dim=-1).values


def gather_positions(x, keep):
    """x : [B, H, N, D], keep : [B, H, n_keep] -> [B, H, n_keep, D]."""
    return x.gather(2, keep.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))


def attention_output_shift(q, k, v, keep, q_pos=None, eps=1e-8):
    """Relative change of the attention output of every window query when only
    the positions in `keep` stay in the cache:

        ||a_kept - a_full|| / ||a_full||

    a_kept renormalises the attention over the kept positions, which is what
    the model does once the other positions are gone (Lemma 1).
    keep : [B, H, n_keep]. Returns [B, H, W].
    """
    q, k, v = q.float(), k.float(), v.float()
    p = attention_probs(q, k, q_pos)                                  # [B, H, W, N]
    a_full = torch.matmul(p, v)
    p_kept = p.gather(-1, keep.unsqueeze(-2).expand(-1, -1, p.shape[-2], -1))
    p_kept = p_kept / p_kept.sum(dim=-1, keepdim=True)
    a_kept = torch.matmul(p_kept, gather_positions(v, keep))
    return (a_kept - a_full).norm(dim=-1) / (a_full.norm(dim=-1) + eps)
