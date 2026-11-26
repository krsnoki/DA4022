"""Hugging Face causal-LM wrapper driven by a ``Backbones/registry.yaml`` entry.

Exposes the stable backbone interface from ``README_backbones.md``:
``load``, ``preprocess`` (chat template + tokenisation), ``generate`` (thin
pass-through to ``transformers``) and ``get_model_metadata``. Adaptation
methods use ``.model`` / ``.tokenizer`` directly when they need their own
decoding loop; they never download or modify weights themselves.
"""

from __future__ import annotations

from typing import Sequence

import torch

from Backbones.registry import get_backbone_entry

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class HFCausalLMBackbone:
    def __init__(self, entry: dict):
        self.entry = entry
        self.model = None
        self.tokenizer = None
        self.device = None
        self.dtype = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_registry(cls, backbone_id: str, device: str = "cuda", dtype: str | None = None,
                      attn_implementation: str | None = "sdpa") -> "HFCausalLMBackbone":
        bb = cls(get_backbone_entry(backbone_id))
        bb.load(device=device, dtype=dtype, attn_implementation=attn_implementation)
        return bb

    def load_tokenizer(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.entry["hf_path"], revision=self.entry["revision"])
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok
        return tok

    def load(self, device: str = "cuda", dtype: str | None = None, attn_implementation: str | None = "sdpa"):
        from transformers import AutoModelForCausalLM

        dtype = dtype or self.entry.get("dtype", "bfloat16")
        if device == "cpu" and dtype != "float32":
            dtype = "float32"
        self.load_tokenizer()
        kwargs = {"revision": self.entry["revision"], "dtype": _DTYPES[dtype]}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(self.entry["hf_path"], **kwargs)
        model.to(device).eval()
        self.model, self.device, self.dtype = model, device, dtype
        return self

    # ------------------------------------------------------------------ #
    def apply_chat_template(self, messages: Sequence[dict]) -> str:
        """Prompt text with the assistant turn opened (for R1 distills this ends in ``<think>\\n``)."""
        return self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)

    def preprocess(self, messages: Sequence[dict]) -> list[int]:
        """Token IDs of the chat prompt. The template already contains BOS, so none is added."""
        text = self.apply_chat_template(messages)
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    @property
    def eos_token_ids(self) -> list[int]:
        eos = self.model.generation_config.eos_token_id if self.model is not None else self.tokenizer.eos_token_id
        return list(eos) if isinstance(eos, (list, tuple)) else [eos]

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    # ------------------------------------------------------------------ #
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_model_metadata(self) -> dict:
        import transformers

        meta = {
            "backbone_id": self.entry["id"],
            "hf_path": self.entry["hf_path"],
            "revision": self.entry["revision"],
            "license": self.entry.get("license"),
            "dtype": self.dtype,
            "device": self.device,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        }
        if self.model is not None:
            meta["num_parameters_counted"] = self.count_parameters()
            meta["attn_implementation"] = getattr(self.model.config, "_attn_implementation", None)
        if self.device and self.device.startswith("cuda") and torch.cuda.is_available():
            idx = torch.device(self.device).index or 0
            props = torch.cuda.get_device_properties(idx)
            meta.update({"gpu": props.name, "gpu_total_memory_mb": round(props.total_memory / 2**20),
                         "cuda_runtime": torch.version.cuda})
        return meta
