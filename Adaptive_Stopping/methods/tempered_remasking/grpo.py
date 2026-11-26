"""diffu-GRPO (d1; Zhao et al., 2025) with a pluggable rollout sampler (Sec. 4.3).

The paper post-trains LLaDA-8B-Instruct with the d1 framework and compares
three rollout samplers at a *fixed wall-clock budget* (72 h on 4×A100):

* TCT (T_pos = 0.1, λ = 0.6)        — this paper
* CT  (λ = 0.6)                      — Fast-dLLM
* AR  (left-to-right)                — Ni et al. (2026)

d1's policy log-likelihood estimator needs a single NFE: the completion is
fully masked, each prompt token is independently masked with probability
``p_mask`` (d1: 0.15), and the per-token log-probabilities of the completion
are read off one forward pass. A fresh prompt mask is drawn for each of the
``μ`` inner policy-gradient iterations; the same mask is used for the current,
old and reference policies within an iteration. The objective is the GRPO
clipped surrogate with a k3 KL penalty to the reference policy.

The rollout sampler is any :class:`~remasking.RemaskingPolicy`, so the only
difference between the three runs is the sampler (and therefore rollout
speed and diversity). Logged per step: wall time, group mean reward, group
reward std (Fig. 5 middle), fraction of zero-variance groups, rollout NFEs.

Hyper-parameter defaults below are illustrative; take the exact values from
the d1 configuration you reproduce (they are not restated in this paper).
For an 8B model, wrap ``policy`` with LoRA adapters (see ``LoRA/``).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import torch

from .metrics import group_reward_stats
from .remasking import RemaskingPolicy
from .sampler import SamplerConfig, generate

__all__ = [
    "DiffuGRPOConfig",
    "random_prompt_mask",
    "one_step_logprobs",
    "completion_mask",
    "group_advantages",
    "diffu_grpo_loss",
    "DiffuGRPOTrainer",
]


@dataclass
class DiffuGRPOConfig:
    num_generations: int = 6          # group size G
    prompts_per_step: int = 1
    p_mask_prompt: float = 0.15       # d1 prompt-masking probability
    num_iterations: int = 1           # μ inner updates per rollout batch
    clip_eps: float = 0.2
    beta: float = 0.04                # KL coefficient
    scale_rewards: bool = False       # divide advantages by group std (standard GRPO) or not
    max_grad_norm: Optional[float] = 1.0
    time_budget_s: float = 72 * 3600.0  # paper: 72 h fixed training duration
    max_steps: Optional[int] = None
    rollout_batch_size: int = 8


def random_prompt_mask(prompt_ids: torch.Tensor, p_mask: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Bernoulli(p_mask) mask over prompt positions, ``[B, P]`` bool."""
    u = torch.rand(prompt_ids.shape, device=prompt_ids.device, generator=generator)
    return u < p_mask


def one_step_logprobs(
    model: Callable[[torch.Tensor], torch.Tensor],
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    mask_token_id: int,
    prompt_mask: torch.Tensor,
) -> torch.Tensor:
    """d1 one-step estimator of per-token ``log π(o_k | q)``.

    Input to the model: ``[prompt with prompt_mask positions -> MASK, MASK * L]``.
    Returns ``[B, L]`` log-probabilities of ``completion_ids`` (differentiable
    if called with grad enabled).
    """
    B, P = prompt_ids.shape
    L = completion_ids.shape[1]
    x = torch.cat(
        [prompt_ids.masked_fill(prompt_mask, mask_token_id), torch.full((B, L), mask_token_id, device=prompt_ids.device, dtype=torch.long)],
        dim=1,
    )
    logits = model(x)[:, P:].float()
    return torch.log_softmax(logits, dim=-1).gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)


def completion_mask(completion_ids: torch.Tensor, eos_token_ids: Sequence[int]) -> torch.Tensor:
    """1 up to and including the first EOS token, 0 afterwards (``[B, L]`` float)."""
    if not eos_token_ids:
        return torch.ones_like(completion_ids, dtype=torch.float32)
    is_eos = torch.isin(completion_ids, torch.as_tensor(list(eos_token_ids), device=completion_ids.device))
    L = completion_ids.shape[1]
    first = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full_like(is_eos[:, 0], L, dtype=torch.long))
    return (torch.arange(L, device=completion_ids.device).unsqueeze(0) <= first.unsqueeze(1)).float()


def group_advantages(rewards: torch.Tensor, scale: bool = False, eps: float = 1e-4) -> torch.Tensor:
    """``A_i = r_i − mean(r)`` (optionally ``/ (std(r) + eps)``) within a group."""
    adv = rewards - rewards.mean()
    if scale:
        adv = adv / (rewards.std(unbiased=True) + eps) if rewards.numel() > 1 else adv
    return adv


def diffu_grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: Optional[torch.Tensor],
    advantages: torch.Tensor,
    cmask: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> tuple:
    """GRPO clipped surrogate + β·KL (k3 estimator), averaged per sequence then over the batch."""
    ratio = torch.exp(logp - old_logp)
    A = advantages.unsqueeze(1)
    surrogate = torch.minimum(ratio * A, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A)
    per_tok = -surrogate
    kl = torch.zeros_like(logp)
    if ref_logp is not None and beta != 0:
        d = ref_logp - logp
        kl = torch.exp(d) - d - 1
        per_tok = per_tok + beta * kl
    denom = cmask.sum(1).clamp_min(1.0)
    loss = ((per_tok * cmask).sum(1) / denom).mean()
    with torch.no_grad():
        stats = {
            "kl": float(((kl * cmask).sum(1) / denom).mean()),
            "clip_frac": float((((ratio - 1).abs() > clip_eps).float() * cmask).sum() / cmask.sum().clamp_min(1)),
        }
    return loss, stats


class DiffuGRPOTrainer:
    """Time-budgeted diffu-GRPO with any remasking policy for rollouts.

    Args:
        policy: trainable ``nn.Module`` mapping ids to logits (``ModelFn``).
        reference: frozen reference model (``None`` -> deep copy of ``policy``
            at construction; pass ``None`` together with ``beta=0`` to skip).
        optimizer: optimiser over ``policy``'s trainable parameters.
        rollout_policy: remasking policy used to generate rollouts.
        sampler_config: :class:`SamplerConfig` for rollouts.
        reward_fn: ``reward_fn(completion_ids: list[int], example: dict) -> float``.
        mask_token_id / eos_token_ids: special tokens.
        config: :class:`DiffuGRPOConfig`.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        reference: Optional[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        rollout_policy: RemaskingPolicy,
        sampler_config: SamplerConfig,
        reward_fn: Callable[[List[int], dict], float],
        mask_token_id: int,
        eos_token_ids: Sequence[int] = (),
        config: DiffuGRPOConfig = DiffuGRPOConfig(),
        generator: Optional[torch.Generator] = None,
        log_fn: Optional[Callable[[dict], None]] = None,
    ):
        self.policy = policy
        if reference is None and config.beta != 0:
            reference = copy.deepcopy(policy)
        if reference is not None:
            reference.eval()
            for p in reference.parameters():
                p.requires_grad_(False)
        self.reference = reference
        self.optimizer = optimizer
        self.rollout_policy = rollout_policy
        self.sampler_config = sampler_config
        self.reward_fn = reward_fn
        self.mask_token_id = mask_token_id
        self.eos_token_ids = list(eos_token_ids)
        self.config = config
        self.generator = generator
        self.log_fn = log_fn
        self.history: List[dict] = []

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def rollout(self, example: dict) -> dict:
        """Generate G completions for one prompt and score them."""
        cfg = self.config
        prompt = example["prompt_ids"]
        was_training = self.policy.training
        self.policy.eval()
        completions, nfes = [], []
        remaining = cfg.num_generations
        while remaining > 0:
            b = min(cfg.rollout_batch_size, remaining)
            res = generate(
                self.policy, prompt.unsqueeze(0).expand(b, -1), self.rollout_policy, self.sampler_config,
                self.mask_token_id, self.eos_token_ids, self.generator,
            )
            completions.append(res.generated)
            nfes += res.nfe.tolist()
            remaining -= b
        self.policy.train(was_training)
        comp = torch.cat(completions)
        rewards = torch.tensor([float(self.reward_fn(c.tolist(), example)) for c in comp], dtype=torch.float32)
        return {"prompt_ids": prompt, "completions": comp, "rewards": rewards, "nfe": nfes}

    def update(self, groups: List[dict]) -> dict:
        """μ policy-gradient iterations on a batch of rollout groups."""
        cfg = self.config
        dev = groups[0]["completions"].device
        prompts = [g["prompt_ids"].unsqueeze(0).expand(g["completions"].shape[0], -1) for g in groups]
        adv = [group_advantages(g["rewards"], cfg.scale_rewards).to(dev) for g in groups]
        cmasks = [completion_mask(g["completions"], self.eos_token_ids) for g in groups]
        # Fix prompt masks and old/ref log-probs for every inner iteration up front.
        plan = []
        with torch.no_grad():
            for _ in range(cfg.num_iterations):
                it = []
                for g, pr in zip(groups, prompts):
                    m = random_prompt_mask(pr, cfg.p_mask_prompt, self.generator)
                    old = one_step_logprobs(self.policy, pr, g["completions"], self.mask_token_id, m)
                    ref = (
                        one_step_logprobs(self.reference, pr, g["completions"], self.mask_token_id, m)
                        if self.reference is not None and cfg.beta != 0
                        else None
                    )
                    it.append((m, old, ref))
                plan.append(it)
        self.policy.train()
        losses, kls, clips = [], [], []
        for it in plan:
            self.optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for (m, old, ref), g, pr, a, cm in zip(it, groups, prompts, adv, cmasks):
                logp = one_step_logprobs(self.policy, pr, g["completions"], self.mask_token_id, m)
                loss, st = diffu_grpo_loss(logp, old, ref, a, cm, cfg.clip_eps, cfg.beta)
                (loss / len(groups)).backward()
                total += float(loss.detach()) / len(groups)
                kls.append(st["kl"])
                clips.append(st["clip_frac"])
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            self.optimizer.step()
            losses.append(total)
        return {"loss": sum(losses) / len(losses), "kl": sum(kls) / len(kls), "clip_frac": sum(clips) / len(clips)}

    def train(self, examples: Iterator[dict]) -> List[dict]:
        """Run until the wall-clock budget (or ``max_steps``) is exhausted.

        ``examples`` yields dicts with ``prompt_ids`` (``[P]`` long tensor) and
        whatever the reward function needs (e.g. the reference answer).
        """
        cfg = self.config
        start = time.monotonic()
        step = 0
        it = iter(examples)
        while True:
            if time.monotonic() - start >= cfg.time_budget_s:
                break
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
            batch = []
            for _ in range(cfg.prompts_per_step):
                try:
                    batch.append(next(it))
                except StopIteration:
                    break
            if not batch:
                break
            t0 = time.monotonic()
            groups = [self.rollout(ex) for ex in batch]
            t_roll = time.monotonic() - t0
            upd = self.update(groups)
            step += 1
            rs = group_reward_stats([g["rewards"].tolist() for g in groups])
            nfe = [n for g in groups for n in g["nfe"]]
            rec = {
                "step": step,
                "wall_time_s": time.monotonic() - start,
                "rollout_time_s": t_roll,
                "rollout_nfe_mean": sum(nfe) / len(nfe),
                **rs,
                **upd,
            }
            self.history.append(rec)
            if self.log_fn is not None:
                self.log_fn(rec)
        return self.history

    def config_dict(self) -> dict:
        return {
            "grpo": asdict(self.config),
            "sampler": asdict(self.sampler_config),
            "rollout_policy": self.rollout_policy.config(),
        }
