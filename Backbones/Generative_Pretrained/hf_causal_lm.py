"""Wrapper for Hugging Face causal language models listed in the registry.

This is the stable backbone interface from ``Backbones/README_backbones.md``:
``load``, ``preprocess``, ``forward``, ``get_model_metadata`` and
``save_adaptation_state``. Experiments refer to a registry ID, for example
``text.olmoe-1b-7b-0125-instruct``, and never hard-code a checkpoint.

    backbone = HFCausalLMBackbone.load("text.olmoe-1b-7b-0125-instruct", device="cuda")
    prompts = backbone.preprocess(["What is 2 + 3?"])

The checkpoint itself is never modified. Methods that change behaviour at run
time (such as XShare routing) attach to ``backbone.model``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import torch

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "registry.yaml"


@dataclass(frozen=True)
class BackboneSpec:
    """The registry fields the loader needs."""

    id: str
    name: str
    hf_path: str
    revision: str
    dtype: str
    license: str


def load_spec(backbone_id: str, registry_path: Path = REGISTRY_PATH) -> BackboneSpec:
    """Read one entry of ``Backbones/registry.yaml``."""
    import yaml

    registry = yaml.safe_load(Path(registry_path).read_text())
    if backbone_id not in registry:
        raise KeyError(f"{backbone_id!r} is not in {registry_path}")
    entry = registry[backbone_id]
    if entry.get("status") == "not_used":
        raise ValueError(f"{backbone_id!r} is marked not_used in the registry; see its card")
    return BackboneSpec(
        id=entry["id"],
        name=entry["name"],
        hf_path=entry["hf_path"],
        revision=str(entry["revision"]),
        dtype=entry["dtype"],
        license=entry["license"],
    )


def _driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


class HFCausalLMBackbone:
    """A registry-pinned Hugging Face causal LM plus its tokenizer."""

    def __init__(self, spec: BackboneSpec, model, tokenizer, load_time_s: Optional[float] = None):
        self.spec = spec
        self.model = model
        self.tokenizer = tokenizer
        self.load_time_s = load_time_s

    @classmethod
    def load(
        cls,
        backbone: Union[str, BackboneSpec],
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "HFCausalLMBackbone":
        """Load weights and tokenizer at the pinned revision.

        ``load_time_s`` covers reading the files and moving the weights to the
        device. It does not include downloading: the files must already be in
        the Hugging Face cache or the time includes the download.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        spec = load_spec(backbone) if isinstance(backbone, str) else backbone
        dtype = dtype or getattr(torch, spec.dtype)
        start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_path, revision=spec.revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(spec.hf_path, revision=spec.revision, dtype=dtype)
        model = model.to(device).eval()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        return cls(spec, model, tokenizer, time.perf_counter() - start)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def preprocess(self, user_messages: list[str]) -> list[str]:
        """Wrap each user message in the model's chat template, ready for generation."""
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": message}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in user_messages
        ]

    def forward(self, **inputs):
        """Plain forward pass; returns the Hugging Face model output (``logits`` etc.)."""
        return self.model(**inputs)

    def eos_token_ids(self) -> set[int]:
        """Token ids that end a reply: the tokenizer's EOS plus the generation config's."""
        ids = set()
        if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        extra = getattr(self.model.generation_config, "eos_token_id", None)
        if extra is not None:
            ids.update(extra if isinstance(extra, (list, tuple)) else [extra])
        return ids

    def get_model_metadata(self) -> dict:
        """Everything a result row needs to identify the model and runtime."""
        config = self.model.config
        device = self.device
        meta = {
            "backbone_id": self.spec.id,
            "hf_path": self.spec.hf_path,
            "revision": self.spec.revision,
            "license": self.spec.license,
            "dtype": str(next(self.model.parameters()).dtype).replace("torch.", ""),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "architecture": type(self.model).__name__,
            "n_layers": getattr(config, "num_hidden_layers", None),
            "n_experts": getattr(config, "num_experts", None),
            "top_k": getattr(config, "num_experts_per_tok", None),
            "norm_topk_prob": getattr(config, "norm_topk_prob", None),
            "attn_implementation": getattr(config, "_attn_implementation", "unknown"),
            "experts_implementation": getattr(config, "_experts_implementation", None) or "eager",
            "load_time_s": self.load_time_s,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "driver": _driver_version() if device.type == "cuda" else None,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "python": platform.python_version(),
        }
        try:
            import transformers

            meta["transformers"] = transformers.__version__
        except ImportError:
            meta["transformers"] = None
        return meta

    def save_adaptation_state(self, path: Union[str, Path], state: dict) -> Path:
        """Save the run-time adaptation applied to this backbone, as JSON.

        Routing methods such as XShare do not change any weight, so their
        adaptation state is just their configuration. No weights are written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"backbone": asdict(self.spec), "adaptation": state}
        path.write_text(json.dumps(payload, indent=1, default=str))
        return path
