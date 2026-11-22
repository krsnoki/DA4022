"""Training-free input and generation difficulty proxies for DIPA."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Any

from Adaptive_Stopping.methods.dipa.types import GenerationAttempt, ProblemInstance
from Adaptive_Stopping.methods.dipa.verifier import normalize_math_answer


@dataclass(frozen=True)
class ProxySpec:
    """Registry metadata for a single difficulty proxy."""

    name: str
    fn: Callable[..., float]
    negatively_correlated: bool
    requires_model: bool = False


def to_difficulty_score(
    raw_value: float,
    negatively_correlated: bool,
    min_difficulty: float = 1e-6,
) -> float:
    """Map a raw proxy signal to a positive score where higher means harder."""
    if not math.isfinite(raw_value):
        return max(1.0, min_difficulty)
    val = max(float(raw_value), min_difficulty)
    if negatively_correlated:
        return max(1.0 / val, min_difficulty)
    return val


def _import_torch() -> tuple[Any, Any]:
    """Lazy-import PyTorch so zero-overhead proxies stay torch-free."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for model-based difficulty proxies (ment, mgn, mvog)."
        ) from exc
    return torch, F


def _model_device(model: Any, torch_mod: Any) -> Any:
    """Infer model device without assuming HF-specific wrapper attributes."""
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch_mod.device("cpu")


def _tokenize_text(
    tokenizer: Any,
    text: str,
    torch_mod: Any,
    device: Any,
    add_special_tokens: bool = True,
) -> Any:
    """Convert text to a [1, seq_len] token-ID tensor on ``device``."""
    if callable(tokenizer):
        try:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=add_special_tokens,
            )
            if isinstance(encoded, dict) or hasattr(encoded, "__getitem__"):
                return encoded["input_ids"].to(device)
        except TypeError:
            pass
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        token_ids = tokenizer.encode(text)
    if not isinstance(token_ids, torch_mod.Tensor):
        token_ids = torch_mod.tensor(token_ids, dtype=torch_mod.long)
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)
    return token_ids.to(device)


def _encode_sequence(
    tokenizer: Any,
    prompt: str,
    attempt: GenerationAttempt | None,
    torch_mod: Any,
    device: Any,
) -> tuple[Any, int]:
    """Build [1, S] or [1, S+L] token-ID tensor and return prompt length S."""
    prompt_ids = _tokenize_text(tokenizer, prompt, torch_mod, device, True)
    prompt_len = int(prompt_ids.size(1))
    if attempt is None:
        return prompt_ids, prompt_len

    if attempt.token_ids is not None and len(attempt.token_ids) > 0:
        gen_ids = torch_mod.tensor(
            attempt.token_ids, dtype=torch_mod.long, device=device
        )
        if gen_ids.ndim == 1:
            gen_ids = gen_ids.unsqueeze(0)
    else:
        gen_ids = _tokenize_text(
            tokenizer, attempt.text, torch_mod, device, add_special_tokens=False
        )
    if gen_ids.numel() == 0:
        return prompt_ids, prompt_len
    return torch_mod.cat([prompt_ids, gen_ids], dim=1), prompt_len


def _sequence_cross_entropy(
    model: Any,
    input_ids: Any,
    functional: Any,
    inputs_embeds: Any = None,
) -> Any:
    """Compute scalar next-token cross-entropy loss across ``input_ids``."""
    if input_ids.size(1) < 2:
        return None
    outputs = (
        model(inputs_embeds=inputs_embeds)
        if inputs_embeds is not None
        else model(input_ids=input_ids)
    )
    logits = (
        outputs.logits
        if hasattr(outputs, "logits")
        else (outputs[0] if isinstance(outputs, tuple) else outputs)
    )
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def _input_gradient_norms(
    model: Any,
    input_ids: Any,
    prompt_len: int,
    torch_mod: Any,
    functional: Any,
) -> Any:
    """Return L2 gradient norms of CE loss w.r.t. input token embeddings."""
    if input_ids.size(1) < 2:
        return None
    if not hasattr(model, "get_input_embeddings"):
        raise ValueError("Model must implement get_input_embeddings() for mgn/mvog.")
    embed_layer = model.get_input_embeddings()
    with torch_mod.enable_grad():
        inputs_embeds = embed_layer(input_ids).detach().requires_grad_(True)
        loss = _sequence_cross_entropy(
            model, input_ids, functional, inputs_embeds=inputs_embeds
        )
        if loss is None:
            return None
        grads = torch_mod.autograd.grad(loss, inputs_embeds)[0]

    # When no generation tokens follow x, the final input token is not shifted into logits.
    grad_len = (
        max(1, prompt_len - 1) if input_ids.size(1) == prompt_len else prompt_len
    )
    input_grads = grads[:, :grad_len, :]
    return input_grads.squeeze(0).norm(p=2, dim=-1)


def _mean_over_attempts(
    attempts: Sequence[GenerationAttempt],
    per_attempt_fn: Callable[[GenerationAttempt], float],
) -> float:
    """Average a scalar proxy across generation attempts, defaulting to 1.0."""
    if not attempts:
        return 1.0
    values = [per_attempt_fn(attempt) for attempt in attempts]
    return float(sum(values) / len(values))


def _input_const(
    instance: ProblemInstance, model: Any = None, tokenizer: Any = None
) -> float:
    return 1.0


def _input_mql(
    instance: ProblemInstance, model: Any = None, tokenizer: Any = None
) -> float:
    if tokenizer is not None:
        length = len(tokenizer.encode(instance.prompt))
    else:
        length = len(instance.prompt.split())
    return float(max(1, length))


def _input_ment(
    instance: ProblemInstance, model: Any = None, tokenizer: Any = None
) -> float:
    torch_mod, functional = _import_torch()
    device = _model_device(model, torch_mod)
    input_ids, _ = _encode_sequence(tokenizer, instance.prompt, None, torch_mod, device)
    with torch_mod.no_grad():
        loss = _sequence_cross_entropy(model, input_ids, functional)
    return float(loss.item()) if loss is not None else 1.0


def _input_mgn(
    instance: ProblemInstance, model: Any = None, tokenizer: Any = None
) -> float:
    torch_mod, functional = _import_torch()
    device = _model_device(model, torch_mod)
    input_ids, prompt_len = _encode_sequence(
        tokenizer, instance.prompt, None, torch_mod, device
    )
    norms = _input_gradient_norms(model, input_ids, prompt_len, torch_mod, functional)
    return float(norms.mean().item()) if norms is not None else 1.0


def _input_mvog(
    instance: ProblemInstance, model: Any = None, tokenizer: Any = None
) -> float:
    torch_mod, functional = _import_torch()
    device = _model_device(model, torch_mod)
    input_ids, prompt_len = _encode_sequence(
        tokenizer, instance.prompt, None, torch_mod, device
    )
    norms = _input_gradient_norms(model, input_ids, prompt_len, torch_mod, functional)
    return float(norms.var(unbiased=False).item()) if norms is not None else 1.0


def _gen_const(
    instance: ProblemInstance,
    attempts: Sequence[GenerationAttempt],
    model: Any = None,
    tokenizer: Any = None,
) -> float:
    return 1.0


def _gen_mgl(
    instance: ProblemInstance,
    attempts: Sequence[GenerationAttempt],
    model: Any = None,
    tokenizer: Any = None,
) -> float:
    if not attempts:
        return 1.0
    lengths = [
        a.num_tokens if a.num_tokens > 0 else max(1, len(a.text.split()))
        for a in attempts
    ]
    return float(sum(lengths) / len(lengths))


def _gen_mgc(
    instance: ProblemInstance,
    attempts: Sequence[GenerationAttempt],
    model: Any = None,
    tokenizer: Any = None,
) -> float:
    if not attempts:
        return 1.0
    answers = [
        normalize_math_answer(a.extracted_answer)
        if a.extracted_answer is not None
        else a.text.strip()
        for a in attempts
    ]
    max_count = max(Counter(answers).values())
    return float(max_count / len(attempts))


def _gen_ment(
    instance: ProblemInstance,
    attempts: Sequence[GenerationAttempt],
    model: Any = None,
    tokenizer: Any = None,
) -> float:
    torch_mod, functional = _import_torch()
    device = _model_device(model, torch_mod)

    def _per_attempt(attempt: GenerationAttempt) -> float:
        full_ids, _ = _encode_sequence(
            tokenizer, instance.prompt, attempt, torch_mod, device
        )
        with torch_mod.no_grad():
            loss = _sequence_cross_entropy(model, full_ids, functional)
        return float(loss.item()) if loss is not None else 1.0

    return _mean_over_attempts(attempts, _per_attempt)


def _gen_mgn(
    instance: ProblemInstance,
    attempts: Sequence[GenerationAttempt],
    model: Any = None,
    tokenizer: Any = None,
) -> float:
    torch_mod, functional = _import_torch()
    device = _model_device(model, torch_mod)

    def _per_attempt(attempt: GenerationAttempt) -> float:
        full_ids, prompt_len = _encode_sequence(
            tokenizer, instance.prompt, attempt, torch_mod, device
        )
        norms = _input_gradient_norms(
            model, full_ids, prompt_len, torch_mod, functional
        )
        return float(norms.mean().item()) if norms is not None else 1.0

    return _mean_over_attempts(attempts, _per_attempt)


def _gen_mvog(
    instance: ProblemInstance,
    attempts: Sequence[GenerationAttempt],
    model: Any = None,
    tokenizer: Any = None,
) -> float:
    torch_mod, functional = _import_torch()
    device = _model_device(model, torch_mod)

    def _per_attempt(attempt: GenerationAttempt) -> float:
        full_ids, prompt_len = _encode_sequence(
            tokenizer, instance.prompt, attempt, torch_mod, device
        )
        norms = _input_gradient_norms(
            model, full_ids, prompt_len, torch_mod, functional
        )
        return float(norms.var(unbiased=False).item()) if norms is not None else 1.0

    return _mean_over_attempts(attempts, _per_attempt)


INPUT_PROXIES: dict[str, ProxySpec] = {
    "const": ProxySpec("const", _input_const, negatively_correlated=False),
    "mql": ProxySpec("mql", _input_mql, negatively_correlated=False),
    "ment": ProxySpec(
        "ment", _input_ment, negatively_correlated=False, requires_model=True
    ),
    "mgn": ProxySpec(
        "mgn", _input_mgn, negatively_correlated=True, requires_model=True
    ),
    "mvog": ProxySpec(
        "mvog", _input_mvog, negatively_correlated=True, requires_model=True
    ),
}

GEN_PROXIES: dict[str, ProxySpec] = {
    "const": ProxySpec("const", _gen_const, negatively_correlated=False),
    "mgl": ProxySpec("mgl", _gen_mgl, negatively_correlated=False),
    "mgc": ProxySpec("mgc", _gen_mgc, negatively_correlated=True),
    "ment": ProxySpec(
        "ment", _gen_ment, negatively_correlated=False, requires_model=True
    ),
    "mgn": ProxySpec(
        "mgn", _gen_mgn, negatively_correlated=True, requires_model=True
    ),
    "mvog": ProxySpec(
        "mvog", _gen_mvog, negatively_correlated=True, requires_model=True
    ),
}


def _validate_proxy_spec(
    registry: dict[str, ProxySpec],
    proxy_name: str,
    kind: str,
    model: Any,
    tokenizer: Any,
) -> ProxySpec:
    spec = registry.get(proxy_name)
    if spec is None:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown {kind} proxy {proxy_name!r}. Available: [{available}]"
        )
    if spec.requires_model and (model is None or tokenizer is None):
        raise ValueError(
            f"Proxy {proxy_name!r} requires both model and tokenizer to be provided."
        )
    return spec


def compute_input_difficulty(
    proxy_name: str,
    instance: ProblemInstance,
    min_difficulty: float = 1e-6,
    model: Any = None,
    tokenizer: Any = None,
) -> float:
    """Compute initial difficulty score M_input(x) for an instance."""
    spec = _validate_proxy_spec(INPUT_PROXIES, proxy_name, "input", model, tokenizer)
    raw = spec.fn(instance=instance, model=model, tokenizer=tokenizer)
    return to_difficulty_score(
        raw, spec.negatively_correlated, min_difficulty=min_difficulty
    )


def compute_gen_difficulty(
    proxy_name: str,
    instance: ProblemInstance,
    attempts: Sequence[GenerationAttempt],
    min_difficulty: float = 1e-6,
    model: Any = None,
    tokenizer: Any = None,
) -> float:
    """Compute updated difficulty score M_gen(O | x) from accumulated attempts."""
    spec = _validate_proxy_spec(GEN_PROXIES, proxy_name, "generation", model, tokenizer)
    raw = spec.fn(
        instance=instance, attempts=attempts, model=model, tokenizer=tokenizer
    )
    return to_difficulty_score(
        raw, spec.negatively_correlated, min_difficulty=min_difficulty
    )
