"""Minimal backbone loader backed by ``Backbones/registry.yaml``.

``load_backbone(backbone_id)`` returns ``(model, preprocessor, metadata)``.
Hugging Face models are loaded at the pinned revision.  ``test.*`` ids build
tiny randomly initialised models from a config (no download) for CPU tests.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import yaml

_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry.yaml")


def load_registry() -> dict:
    with open(_REGISTRY_PATH) as f:
        entries = yaml.safe_load(f)
    return {e["id"]: e for e in entries}


def get_model_metadata(backbone_id: str) -> dict:
    reg = load_registry()
    if backbone_id not in reg:
        raise KeyError(f"unknown backbone id {backbone_id!r}; known: {sorted(reg)}")
    return reg[backbone_id]


def _tiny_qwen2(attn_implementation: str = "eager"):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        attn_implementation=attn_implementation,
    )
    return Qwen2ForCausalLM(cfg)


def _tiny_siglip(attn_implementation: str = "eager"):
    from transformers import SiglipConfig, SiglipModel

    cfg = SiglipConfig(
        text_config=dict(
            vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, max_position_embeddings=16,
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            image_size=32, patch_size=8,
        ),
        attn_implementation=attn_implementation,
    )
    for sub in (cfg.text_config, cfg.vision_config):
        sub._attn_implementation_internal = attn_implementation
    return SiglipModel(cfg)


def load_backbone(
    backbone_id: str,
    dtype: Optional[Any] = None,
    device: Optional[str] = None,
    attn_implementation: str = "eager",
    revision: Optional[str] = None,
) -> Tuple[Any, Any, dict]:
    """Load a registered backbone. Returns ``(model, preprocessor_or_None, metadata)``."""
    import torch

    meta = dict(get_model_metadata(backbone_id))
    if backbone_id == "test.tiny_qwen2.random":
        model, pre = _tiny_qwen2(attn_implementation), None
    elif backbone_id == "test.tiny_siglip.random":
        model, pre = _tiny_siglip(attn_implementation), None
    else:
        hf_id, rev = meta["hf_id"], revision or meta["revision"]
        kwargs = dict(revision=rev, attn_implementation=attn_implementation)
        if dtype is not None:
            kwargs["dtype"] = dtype
        if meta["modality"] == "text":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
            pre = AutoTokenizer.from_pretrained(hf_id, revision=rev)
        else:
            from transformers import AutoModel, AutoProcessor

            model = AutoModel.from_pretrained(hf_id, **kwargs)
            pre = AutoProcessor.from_pretrained(hf_id, revision=rev)
        meta["loaded_revision"] = rev
    if dtype is not None and backbone_id.startswith("test."):
        model = model.to(dtype)
    if device is not None:
        model = model.to(device)
    return model, pre, meta
