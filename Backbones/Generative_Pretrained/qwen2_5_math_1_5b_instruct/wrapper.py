"""Wrapper for Qwen/Qwen2.5-Math-1.5B-Instruct with lazy HF/PyTorch loading."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Any


def _validate_generation_params(
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
) -> None:
    """Validate sampling and length bounds for generation."""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if not math.isfinite(top_p) or top_p <= 0 or top_p > 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
        raise ValueError(
            f"repetition_penalty must be > 0, got {repetition_penalty}"
        )
    if isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")


@dataclass(frozen=True)
class BackboneConfig:
    """Loading and default generation configuration for Qwen2.5-Math-1.5B-Instruct."""

    model_id: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    revision: str = "aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"
    device: str = "auto"
    dtype: str = "float16"
    temperature: float = 0.7
    top_p: float = 0.8
    repetition_penalty: float = 1.05
    max_new_tokens: int = 512
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        _validate_generation_params(
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
        )


def get_model_metadata() -> dict[str, Any]:
    """Return static model metadata matching ``Backbones/registry.yaml``."""
    return {
        "id": "llm.qwen2_5_math_1_5b_instruct.generative",
        "name": "Qwen2.5-Math-1.5B-Instruct",
        "modality": "text",
        "architecture": "Qwen2ForCausalLM",
        "pretraining_regime": "mixed",
        "objective": (
            "mathematical reasoning via chain-of-thought and tool-integrated reasoning"
        ),
        "weights_source": "huggingface",
        "weights_uri": "https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct",
        "revision": "aafeb0fc6f22cbf0eaeed126eff8be45b0360a35",
        "license": "apache-2.0",
        "num_parameters": 1540000000,
        "context_length": 32768,
        "max_generation_tokens": 8192,
        "dtype": "float16",
        "parent_backbone": "Qwen/Qwen2.5-Math-1.5B",
        "fine_tuning_dataset": "unknown",
    }


def _resolve_dtype(dtype_name: str, torch_mod: Any) -> Any:
    """Map string precision names to ``torch.dtype`` values."""
    mapping = {
        "float16": torch_mod.float16,
        "fp16": torch_mod.float16,
        "bfloat16": torch_mod.bfloat16,
        "bf16": torch_mod.bfloat16,
        "float32": torch_mod.float32,
        "fp32": torch_mod.float32,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(
            f"Unsupported dtype {dtype_name!r}. Expected float16, bfloat16, or float32."
        )
    return mapping[key]


def _build_ones_mask(input_ids: Any) -> Any:
    """Create a ones attention mask matching ``input_ids`` shape."""
    if hasattr(input_ids, "new_ones") and hasattr(input_ids, "shape"):
        return input_ids.new_ones(input_ids.shape)
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return [[1] * len(row) for row in input_ids]
        return [1] * len(input_ids)
    if hasattr(input_ids, "shape"):
        import numpy as np

        return np.ones(input_ids.shape, dtype=np.int64)
    raise TypeError(
        f"Cannot derive attention_mask for input_ids of type {type(input_ids).__name__}"
    )


def _model_device(model: Any) -> Any:
    """Infer model parameter device without requiring Hugging Face attributes."""
    if not hasattr(model, "parameters"):
        return None
    try:
        first_param = next(model.parameters())
    except (StopIteration, TypeError):
        return None
    return getattr(first_param, "device", None)


def _move_to_model_device(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    """Move tensor-like inputs onto ``model``'s parameter device when known."""
    device = _model_device(model)
    if device is None:
        return inputs
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _prompt_length(input_ids: Any) -> int:
    """Return token count along the sequence axis of ``input_ids``."""
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)
    return len(input_ids)


def _extract_sequences(outputs: Any) -> list[Any]:
    """Normalize ``model.generate()`` output into a list of 1D sequences."""
    raw = outputs.sequences if hasattr(outputs, "sequences") else outputs
    if hasattr(raw, "ndim"):
        return [raw] if raw.ndim == 1 else list(raw)
    if isinstance(raw, list) and raw and not isinstance(raw[0], (list, tuple)):
        return [raw]
    return list(raw)


def _to_int_list(tokens: Any) -> list[int]:
    """Convert a 1D tensor or sequence of token IDs into Python ints."""
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return [int(tok) for tok in tokens]


class Qwen25Math15BInstructBackbone:
    """Reusable wrapper around ``Qwen/Qwen2.5-Math-1.5B-Instruct``."""

    def __init__(
        self,
        config: BackboneConfig | None = None,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.config = config if config is not None else BackboneConfig()
        self.model = model
        self.tokenizer = tokenizer

    def load(
        self,
        config: BackboneConfig | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> Qwen25Math15BInstructBackbone:
        """Lazy-import PyTorch/Transformers, load model & tokenizer, and enter eval mode."""
        active_config = config if config is not None else self.config
        overrides: dict[str, Any] = {}
        if device is not None:
            overrides["device"] = device
        if dtype is not None:
            overrides["dtype"] = dtype
        if overrides:
            active_config = replace(active_config, **overrides)
        self.config = active_config

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to load Qwen2.5-Math-1.5B-Instruct."
            ) from exc
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required to load Qwen2.5-Math-1.5B-Instruct."
            ) from exc

        torch_dtype = _resolve_dtype(self.config.dtype, torch)
        target_device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self.config.device == "auto"
            else self.config.device
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
        )
        if (
            hasattr(tokenizer, "pad_token_id")
            and tokenizer.pad_token_id is None
            and getattr(tokenizer, "eos_token_id", None) is not None
        ):
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            torch_dtype=torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
        )
        if hasattr(model, "to"):
            moved = model.to(target_device)
            if moved is not None:
                model = moved
        if hasattr(model, "eval"):
            model.eval()

        self.tokenizer = tokenizer
        self.model = model
        return self

    def preprocess(self, raw_inputs: str | list[dict[str, str]]) -> dict[str, Any]:
        """Format string or chat-message prompts and tokenize to tensor inputs."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded. Call load() first.")

        if isinstance(raw_inputs, str):
            text = raw_inputs
        elif isinstance(raw_inputs, list):
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    raw_inputs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = "\n".join(msg["content"] for msg in raw_inputs)
        else:
            raise TypeError(
                f"raw_inputs must be str or list[dict[str, str]], got {type(raw_inputs).__name__}"
            )

        encoded = self.tokenizer(text, return_tensors="pt")
        inputs = dict(encoded)
        if "attention_mask" not in inputs or inputs["attention_mask"] is None:
            inputs["attention_mask"] = _build_ones_mask(inputs["input_ids"])
        return inputs

    def forward(self, inputs: dict[str, Any]) -> Any:
        """Run a causal-LM forward pass on preprocessed ``inputs``."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        prepared = _move_to_model_device(inputs, self.model)
        return self.model(**prepared)

    def _prepare_generation_context(self, seed: int | None) -> tuple[Any, Any]:
        """Lazy-import PyTorch for no-grad execution and optional RNG seeding."""
        try:
            import torch
        except ImportError:
            return nullcontext(), None
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        return torch.no_grad(), torch

    def generate(
        self,
        raw_inputs: str | list[dict[str, str]],
        num_return_sequences: int = 1,
        seed: int | None = None,
        **gen_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate candidate completions and return decoded token/latency records."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generate().")
        if num_return_sequences <= 0:
            raise ValueError(
                f"num_return_sequences must be > 0, got {num_return_sequences}"
            )

        inputs = _move_to_model_device(self.preprocess(raw_inputs), self.model)
        prompt_len = _prompt_length(inputs["input_ids"])

        temperature = float(gen_kwargs.pop("temperature", self.config.temperature))
        top_p = float(gen_kwargs.pop("top_p", self.config.top_p))
        repetition_penalty = float(
            gen_kwargs.pop("repetition_penalty", self.config.repetition_penalty)
        )
        max_new_tokens = int(
            gen_kwargs.pop("max_new_tokens", self.config.max_new_tokens)
        )
        _validate_generation_params(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        do_sample = (
            bool(gen_kwargs.pop("do_sample"))
            if "do_sample" in gen_kwargs
            else temperature > 0.0
        )

        generate_kwargs: dict[str, Any] = {
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        generate_kwargs.update(gen_kwargs)

        grad_ctx, torch_mod = self._prepare_generation_context(seed)
        is_cuda = (
            torch_mod is not None
            and torch_mod.cuda.is_available()
            and getattr(_model_device(self.model), "type", None) == "cuda"
        )
        if is_cuda:
            torch_mod.cuda.synchronize()
        start = time.perf_counter()
        with grad_ctx:
            outputs = self.model.generate(**inputs, **generate_kwargs)
        if is_cuda:
            torch_mod.cuda.synchronize()
        total_latency_ms = max(0.0, (time.perf_counter() - start) * 1000.0)

        sequences = _extract_sequences(outputs)
        # Amortize batched wall-clock latency across returned sequences so callers summing
        # per-attempt latencies match total generation wall-clock time.
        per_seq_latency_ms = total_latency_ms / len(sequences) if sequences else 0.0

        results: list[dict[str, Any]] = []
        for seq in sequences:
            gen_ids = _to_int_list(seq[prompt_len:])
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(
                {
                    "text": text,
                    "num_tokens": len(gen_ids),
                    "token_ids": gen_ids,
                    "latency_ms": per_seq_latency_ms,
                }
            )
        return results

    def get_model_metadata(self) -> dict[str, Any]:
        """Combine static registry metadata with active wrapper configuration."""
        metadata = get_model_metadata()
        metadata["config"] = asdict(self.config)
        metadata["loaded"] = self.model is not None and self.tokenizer is not None
        return metadata

    def save_adaptation_state(self, path: str | Path) -> Path:
        """Persist JSON configuration and metadata (never base model weights)."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": get_model_metadata(),
            "config": asdict(self.config),
        }
        out_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return out_path
