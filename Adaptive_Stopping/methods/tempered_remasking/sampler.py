"""Generic sampling loop for masked diffusion LMs with pluggable remasking.

The loop is the "first-hit" sampler of Sec. 2.2 combined with semi-autoregressive
(block-wise) decoding (Arriola et al. 2025; Nie et al. 2025b), as used in the
paper's experiments (L = 256 generated tokens, blocks of 32, T_token = 0.8):

    x <- [prompt, MASK * L]
    for each block (left to right):
        while the block still contains masks:
            logits = model(x)                       # one NFE
            x0, c  = sample_tokens(logits, T_token) # Alg. 1-4, lines 4-5
            U      = policy.select(c, candidates)   # Alg. 1-4, line 6(-7)
            x[U]   = x0[U]

NFE accounting follows footnote 3 of the paper: for confidence-based policies
(LC/TLC/CT/TCT/random) NFEs are counted until all L positions are unmasked; for
autoregressive generation, generation stops at the first EOS token
(``nfe_paper_convention`` reports "tokens generated before <eos>").

The loop also implements the module's shared ``StoppingPolicy`` contract
(:class:`DecodingStopPolicy`), including an optional hard NFE budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch

from .remasking import CONFIDENCE_MODES, RemaskingPolicy, get_num_transfer_tokens, sample_tokens

__all__ = [
    "ModelFn",
    "SamplerConfig",
    "DecodingStopPolicy",
    "GenerationResult",
    "generate",
    "generate_samples",
]

#: A masked diffusion LM: maps token ids ``[B, N]`` to logits ``[B, N, V]`` such
#: that ``logits[b, k]`` parameterises ``q_θ^k(· | x_t)``. Model adapters (see
#: :mod:`.models`) must already apply any family-specific logit shift.
ModelFn = Callable[[torch.Tensor], torch.Tensor]

EMPTY_SELECTION_MODES = ("resample", "argmax", "skip")
BUDGET_MODES = ("commit_all", "leave_masked")


@dataclass
class SamplerConfig:
    """Sampling hyper-parameters (paper defaults from Sec. 4.1).

    Attributes:
        gen_length: number of generated positions L (paper: 256).
        block_length: semi-AR block size (paper: 32). Set equal to
            ``gen_length`` to disable block-wise decoding.
        steps: total denoising steps for fixed-K policies (LC/TLC/random);
            ``K = gen_length / steps`` tokens per step. Must be a multiple of the
            number of blocks. ``None`` -> ``gen_length`` (K = 1, the setting
            analysed in Appendix D). Ignored by CT/TCT/AR.
        token_temperature: T_token (paper: 0.8 for sampling; 0 for greedy eval).
        confidence_mode: see :data:`remasking.CONFIDENCE_MODES`.
        empty_selection: what to do when a threshold policy (CT/TCT) selects
            no position in a step:
            ``"resample"`` (default) – redraw tokens/Bernoullis from the *same*
            logits until non-empty (exactly the distribution of the literal
            Algorithm 4 conditioned on progress, without wasting NFEs, because
            the model is deterministic given x_t), falling back to ``argmax``
            after ``max_resample`` tries or for deterministic CT;
            ``"argmax"`` – unmask the single most confident candidate (the
            Fast-dLLM reference behaviour);
            ``"skip"`` – commit nothing and spend another NFE (literal Alg. 4;
            only allowed for stochastic policies).
        max_resample: resampling attempts before the argmax fallback.
        max_nfe: optional per-sample NFE budget (``None`` = unlimited).
        on_budget_exhausted: ``"commit_all"`` fills every remaining mask with
            the last proposal ``x0`` (one-shot finish); ``"leave_masked"``
            returns the partially masked sequence. With ``max_nfe = 0`` no
            forward pass happens and the output stays masked either way.
        stop_at_eos: stop when an EOS token is committed. ``None`` -> enabled
            for the autoregressive policy only (paper footnote 3). Only valid
            for the autoregressive policy.
        record_entropy: record the untempered predictive entropy of each
            position at the step it was unmasked (used for fork analyses).
    """

    gen_length: int = 256
    block_length: int = 32
    steps: Optional[int] = None
    token_temperature: float = 0.8
    confidence_mode: str = "sampled"
    empty_selection: str = "resample"
    max_resample: int = 64
    max_nfe: Optional[int] = None
    on_budget_exhausted: str = "commit_all"
    stop_at_eos: Optional[bool] = None
    record_entropy: bool = False

    def resolved_steps(self) -> int:
        return self.gen_length if self.steps is None else int(self.steps)

    def num_blocks(self) -> int:
        return self.gen_length // self.block_length

    def validate(self, policy: RemaskingPolicy) -> None:
        if self.gen_length < 1:
            raise ValueError("gen_length must be >= 1")
        if self.block_length < 1 or self.gen_length % self.block_length != 0:
            raise ValueError("gen_length must be a positive multiple of block_length")
        if self.token_temperature < 0:
            raise ValueError("token_temperature must be >= 0")
        if self.confidence_mode not in CONFIDENCE_MODES:
            raise ValueError(f"confidence_mode must be one of {CONFIDENCE_MODES}")
        if self.empty_selection not in EMPTY_SELECTION_MODES:
            raise ValueError(f"empty_selection must be one of {EMPTY_SELECTION_MODES}")
        if self.on_budget_exhausted not in BUDGET_MODES:
            raise ValueError(f"on_budget_exhausted must be one of {BUDGET_MODES}")
        if self.max_nfe is not None and self.max_nfe < 0:
            raise ValueError("max_nfe must be >= 0 or None")
        if self.max_resample < 0:
            raise ValueError("max_resample must be >= 0")
        if policy.fixed_k and policy.uses_blocks:
            steps = self.resolved_steps()
            if steps < self.num_blocks() or steps % self.num_blocks() != 0:
                raise ValueError("steps must be a positive multiple of gen_length // block_length")
        if policy.adaptive and self.empty_selection == "skip" and not policy.stochastic:
            raise ValueError("empty_selection='skip' would never make progress for a deterministic threshold policy")
        if self.stop_at_eos and policy.name != "autoregressive":
            raise ValueError("stop_at_eos is only defined for the autoregressive policy")


class DecodingStopPolicy:
    """Termination rule of the sampler, in the module's ``StoppingPolicy`` form.

    Vectorised over the batch. ``should_stop`` returns ``True`` for a sample when

    * ``stop_at_eos`` and an EOS token has been committed (``"eos"``; takes
      precedence, since the sampler pads the rest of the sequence with EOS), or
    * no masked position remains (``"complete"``), or
    * an NFE budget is set and ``nfe >= budget`` (``"budget"``).

    The required telemetry (``state['masked_remaining']``, ``state['eos_generated']``
    when ``stop_at_eos``, ``metrics['nfe']`` when a budget is set) is produced
    by the sampler itself, without any extra forward pass. Missing telemetry
    raises ``KeyError`` instead of silently never stopping.
    """

    def __init__(self, stop_at_eos: bool = False):
        self.stop_at_eos = bool(stop_at_eos)
        self.reset()

    def reset(self, request_metadata: Optional[dict] = None) -> None:
        self.request_metadata = request_metadata or {}
        self._reasons: Dict[str, int] = {}
        self._nfes: List[int] = []

    @staticmethod
    def _get(d: Optional[dict], key: str, what: str) -> torch.Tensor:
        if d is None or key not in d:
            raise KeyError(f"missing telemetry {what}[{key!r}] required by DecodingStopPolicy")
        return torch.as_tensor(d[key])

    def stop_reasons(self, state: dict, metrics: dict, budget: Optional[int]) -> List[Optional[str]]:
        masked = self._get(state, "masked_remaining", "state").reshape(-1)
        complete = masked == 0
        eos = torch.zeros_like(complete)
        if self.stop_at_eos:
            eos = self._get(state, "eos_generated", "state").reshape(-1).bool()
        over = torch.zeros_like(complete)
        if budget is not None:
            over = self._get(metrics, "nfe", "metrics").reshape(-1) >= int(budget)
        reasons: List[Optional[str]] = []
        for c, e, o in zip(complete.tolist(), eos.tolist(), over.tolist()):
            reasons.append("eos" if e else "complete" if c else "budget" if o else None)
        return reasons

    def should_stop(self, state: dict, metrics: dict, budget: Optional[int]) -> torch.Tensor:
        return torch.tensor([r is not None for r in self.stop_reasons(state, metrics, budget)])

    def record(self, reason: str, nfe: int) -> None:
        self._reasons[reason] = self._reasons.get(reason, 0) + 1
        self._nfes.append(int(nfe))

    def report(self) -> dict:
        n = len(self._nfes)
        return {
            "num_samples": n,
            "stop_reasons": dict(self._reasons),
            "stop_rate_eos": self._reasons.get("eos", 0) / n if n else None,
            "stop_rate_budget": self._reasons.get("budget", 0) / n if n else None,
            "mean_nfe": sum(self._nfes) / n if n else None,
            "max_nfe": max(self._nfes) if n else None,
            "deterministic": None,  # filled in by the sampler (depends on policy & T_token)
        }


@dataclass
class GenerationResult:
    """Output of :func:`generate`.

    Attributes:
        sequences: ``[B, P + L]`` prompt followed by generated tokens.
        prompt_length: ``P``.
        nfe: ``[B]`` forward passes actually spent on each sample.
        nfe_paper_convention: ``[B]`` NFEs as counted in the paper (footnote 3):
            equal to ``nfe`` except for AR samples that stopped at EOS, where
            the forward pass that produced EOS is not counted.
        stop_reasons: per-sample ``"complete" | "eos" | "budget"``.
        unmask_step: ``[B, P + L]`` global step at which each position was
            committed (``-1`` for prompt / never). Gives ``T(j)`` of App. D.
        confidence_at_unmask: ``[B, P + L]`` confidence of the committed token.
        entropy_at_unmask: ``[B, P + L]`` predictive entropy at commit time, or
            ``None`` unless ``record_entropy``.
        tokens_per_step: ``[S, B]`` positions committed at each global step.
        stats: counters (resampling / fallbacks / skipped steps) and the
            stopping-policy report.
        config: sampler and policy configuration (for experiment metadata).
    """

    sequences: torch.Tensor
    prompt_length: int
    nfe: torch.Tensor
    nfe_paper_convention: torch.Tensor
    stop_reasons: List[str]
    unmask_step: torch.Tensor
    confidence_at_unmask: torch.Tensor
    entropy_at_unmask: Optional[torch.Tensor]
    tokens_per_step: torch.Tensor
    stats: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def generated(self) -> torch.Tensor:
        return self.sequences[:, self.prompt_length:]


def _handle_empty_selection(
    sel: torch.Tensor,
    cand: torch.Tensor,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    confidence: torch.Tensor,
    policy: RemaskingPolicy,
    cfg: SamplerConfig,
    generator: Optional[torch.Generator],
    stats: dict,
):
    """Apply ``cfg.empty_selection`` to rows where a threshold policy chose nothing."""
    empty = ~sel.any(dim=-1) & cand.any(dim=-1)
    if not empty.any() or cfg.empty_selection == "skip":
        if empty.any():
            stats["skipped_steps"] += int(empty.sum())
        return sel, tokens, confidence
    if cfg.empty_selection == "resample" and policy.stochastic:
        for _ in range(cfg.max_resample):
            idx = torch.nonzero(empty, as_tuple=False).squeeze(-1)
            prop = sample_tokens(logits[idx], cfg.token_temperature, generator, cfg.confidence_mode)
            new_sel = policy.select(prop.confidence, cand[idx], None, generator)
            tokens[idx] = prop.tokens
            confidence[idx] = prop.confidence
            sel[idx] = new_sel
            stats["resamples"] += int(idx.numel())
            empty = ~sel.any(dim=-1) & cand.any(dim=-1)
            if not empty.any():
                return sel, tokens, confidence
    # argmax fallback (Fast-dLLM behaviour): commit the most confident candidate.
    idx = torch.nonzero(empty, as_tuple=False).squeeze(-1)
    best = confidence[idx].masked_fill(~cand[idx], float("-inf")).argmax(dim=-1)
    sel[idx, best] = True
    stats["argmax_fallbacks"] += int(idx.numel())
    return sel, tokens, confidence


@torch.no_grad()
def generate(
    model: ModelFn,
    prompt_ids: torch.Tensor,
    policy: RemaskingPolicy,
    config: SamplerConfig,
    mask_token_id: int,
    eos_token_ids: Optional[Sequence[int]] = None,
    generator: Optional[torch.Generator] = None,
) -> GenerationResult:
    """Sample from a masked diffusion LM with the given remasking policy.

    Args:
        model: callable mapping ``[B, N]`` token ids to ``[B, N, V]`` logits.
        prompt_ids: ``[P]`` or ``[B, P]`` prompt token ids (``P`` may be 0).
            Rows must be equally long (repeat a prompt to draw several samples).
        policy: a :class:`~remasking.RemaskingPolicy`.
        config: :class:`SamplerConfig`.
        mask_token_id: id of the ``[MASK]`` token.
        eos_token_ids: EOS ids (needed for AR early stopping).
        generator: optional ``torch.Generator`` (on the model's device).

    Returns:
        :class:`GenerationResult`.
    """
    config.validate(policy)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if prompt_ids.dim() != 2:
        raise ValueError("prompt_ids must be [P] or [B, P]")
    device = prompt_ids.device
    B, P = prompt_ids.shape
    L = config.gen_length
    N = P + L
    stop_at_eos = (policy.name == "autoregressive") if config.stop_at_eos is None else bool(config.stop_at_eos)
    eos_ids = torch.as_tensor(list(eos_token_ids or []), dtype=torch.long, device=device)
    if stop_at_eos and eos_ids.numel() == 0:
        stop_at_eos = False  # nothing to detect; run to completion

    x = torch.full((B, N), mask_token_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    nfe = torch.zeros(B, dtype=torch.long, device=device)
    unmask_step = torch.full((B, N), -1, dtype=torch.long, device=device)
    conf_at = torch.full((B, N), float("nan"), dtype=torch.float64, device=device)
    ent_at = torch.full((B, N), float("nan"), dtype=torch.float64, device=device) if config.record_entropy else None
    last_tokens = torch.full((B, N), mask_token_id, dtype=torch.long, device=device)
    has_proposal = torch.zeros(B, dtype=torch.bool, device=device)
    eos_generated = torch.zeros(B, dtype=torch.bool, device=device)
    stopped = torch.zeros(B, dtype=torch.bool, device=device)
    reasons: List[str] = [""] * B
    tokens_per_step: List[torch.Tensor] = []
    stats = {"resamples": 0, "argmax_fallbacks": 0, "skipped_steps": 0}

    stopper = DecodingStopPolicy(stop_at_eos=stop_at_eos)
    stopper.reset({"batch_size": B, "prompt_length": P})

    def update_stops() -> None:
        state = {"masked_remaining": (x[:, P:] == mask_token_id).sum(dim=-1), "eos_generated": eos_generated}
        for b, r in enumerate(stopper.stop_reasons(state, {"nfe": nfe}, config.max_nfe)):
            if r is None or stopped[b]:
                continue
            if r == "budget" and config.on_budget_exhausted == "commit_all" and has_proposal[b]:
                still = x[b] == mask_token_id
                still[:P] = False
                x[b] = torch.where(still, last_tokens[b], x[b])
                unmask_step[b] = torch.where(still, torch.full_like(unmask_step[b], max(len(tokens_per_step) - 1, 0)), unmask_step[b])
            stopped[b] = True
            reasons[b] = r
            stopper.record(r, int(nfe[b]))

    update_stops()  # handles max_nfe == 0 and L == 0 edge cases

    if policy.uses_blocks:
        bl = config.block_length
        blocks = [(P + i * bl, P + (i + 1) * bl) for i in range(config.num_blocks())]
        steps_per_block = config.resolved_steps() // config.num_blocks()
    else:
        blocks = [(P, N)]
        steps_per_block = L

    global_step = 0
    for lo, hi in blocks:
        schedule = None
        if policy.fixed_k:
            schedule = get_num_transfer_tokens(x[:, lo:hi] == mask_token_id, steps_per_block)
        i = 0
        while True:
            block_masked = x[:, lo:hi] == mask_token_id
            active = block_masked.any(dim=-1) & ~stopped
            if not active.any():
                break
            rows = torch.nonzero(active, as_tuple=False).squeeze(-1)
            logits = model(x[rows])
            if logits.shape[:2] != (rows.numel(), N):
                raise ValueError(f"model returned logits of shape {tuple(logits.shape)}; expected [{rows.numel()}, {N}, V]")
            nfe[rows] += 1
            prop = sample_tokens(
                logits, config.token_temperature, generator, config.confidence_mode, compute_entropy=config.record_entropy
            )
            tokens, confidence = prop.tokens, prop.confidence
            cand = torch.zeros((rows.numel(), N), dtype=torch.bool, device=device)
            cand[:, lo:hi] = block_masked[rows]

            k = None
            if policy.fixed_k:
                k = schedule[rows, i] if i < steps_per_block else torch.ones_like(rows)
                # Guard: a row must always make progress. A scheduled K of 0 only occurs
                # when steps > gen_length (K < 1); LLaDA's reference code would spend an
                # NFE committing nothing, here such steps commit one token instead.
                k = torch.where(k > 0, k, torch.ones_like(k))
            sel = policy.select(confidence, cand, k, generator)
            if policy.adaptive:
                sel, tokens, confidence = _handle_empty_selection(
                    sel, cand, logits, tokens, confidence, policy, config, generator, stats
                )
            del logits

            sub_x = x[rows]
            x[rows] = torch.where(sel, tokens, sub_x)
            unmask_step[rows] = torch.where(sel, torch.full_like(sel, global_step, dtype=torch.long), unmask_step[rows])
            conf_at[rows] = torch.where(sel, confidence.to(torch.float64), conf_at[rows])
            if ent_at is not None:
                ent_at[rows] = torch.where(sel, prop.entropy, ent_at[rows])
            last_tokens[rows] = tokens
            has_proposal[rows] = True
            step_counts = torch.zeros(B, dtype=torch.long, device=device)
            step_counts[rows] = sel.sum(dim=-1)
            tokens_per_step.append(step_counts)

            if stop_at_eos:
                committed_eos = sel & torch.isin(tokens, eos_ids)
                hit = committed_eos.any(dim=-1)
                for j in torch.nonzero(hit, as_tuple=False).squeeze(-1).tolist():
                    b = int(rows[j])
                    eos_tok = tokens[j][committed_eos[j]][0]
                    rest = x[b] == mask_token_id
                    rest[:P] = False
                    x[b] = torch.where(rest, eos_tok, x[b])  # pad with EOS, as LLaDA does
                    eos_generated[b] = True

            update_stops()
            global_step += 1
            i += 1

    for b in range(B):  # rows that never triggered a stop condition (cannot happen, defensive)
        if not stopped[b]:
            reasons[b] = "complete"
            stopper.record("complete", int(nfe[b]))

    nfe_paper = nfe.clone()
    if policy.name == "autoregressive":
        for b, r in enumerate(reasons):
            if r == "eos":
                nfe_paper[b] -= 1

    report = stopper.report()
    report["deterministic"] = (not policy.stochastic) and config.token_temperature == 0
    stats["stop_policy"] = report
    tps = torch.stack(tokens_per_step) if tokens_per_step else torch.zeros((0, B), dtype=torch.long, device=device)
    return GenerationResult(
        sequences=x,
        prompt_length=P,
        nfe=nfe,
        nfe_paper_convention=nfe_paper,
        stop_reasons=reasons,
        unmask_step=unmask_step,
        confidence_at_unmask=conf_at,
        entropy_at_unmask=ent_at,
        tokens_per_step=tps,
        stats=stats,
        config={"sampler": asdict(config), "policy": policy.config(), "stop_at_eos": stop_at_eos},
    )


def generate_samples(
    model: ModelFn,
    prompt_ids: torch.Tensor,
    num_samples: int,
    policy: RemaskingPolicy,
    config: SamplerConfig,
    mask_token_id: int,
    eos_token_ids: Optional[Sequence[int]] = None,
    batch_size: int = 8,
    generator: Optional[torch.Generator] = None,
) -> List[dict]:
    """Draw ``num_samples`` independent samples for one prompt (for pass@k).

    Returns one dict per sample with keys ``tokens`` (generated ids, list),
    ``nfe``, ``nfe_paper_convention`` and ``stop_reason``.
    """
    if prompt_ids.dim() != 1:
        raise ValueError("prompt_ids must be 1-D for generate_samples")
    out: List[dict] = []
    remaining = num_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        res = generate(
            model, prompt_ids.unsqueeze(0).expand(b, -1), policy, config, mask_token_id, eos_token_ids, generator
        )
        for j in range(b):
            out.append(
                {
                    "tokens": res.generated[j].tolist(),
                    "nfe": int(res.nfe[j]),
                    "nfe_paper_convention": int(res.nfe_paper_convention[j]),
                    "stop_reason": res.stop_reasons[j],
                }
            )
        remaining -= b
    return out
