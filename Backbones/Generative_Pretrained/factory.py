"""Build a generative backbone from an experiment config block.

```yaml
backbone:
  type: vllm | hf | mock
  id: Qwen/Qwen3.5-9B
  revision: <commit>      # pin for reported results
  dtype: bfloat16
  device: cuda            # hf only
  kwargs: {}              # extra constructor arguments
```
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional


def build_backbone(cfg: Dict, items: Optional[Iterable[dict]] = None):
    kind = cfg.get("type")
    kwargs = dict(cfg.get("kwargs") or {})
    if kind == "mock":
        from .mock_mcq_lm import MockMCQBackbone, NoScoreMockBackbone

        if items is None:
            raise ValueError("mock backbone needs the synthetic items")
        return NoScoreMockBackbone(items) if kwargs.get("no_option_scores") else MockMCQBackbone(items)
    if kind == "hf":
        from .hf_causal_lm import HFCausalLM

        return HFCausalLM(cfg["id"], revision=cfg.get("revision"), device=cfg.get("device", "cpu"),
                          dtype=cfg.get("dtype", "float32"), **kwargs)
    if kind == "vllm":
        from .vllm_causal_lm import VLLMCausalLM

        return VLLMCausalLM(cfg["id"], revision=cfg.get("revision"), dtype=cfg.get("dtype", "bfloat16"), **kwargs)
    raise ValueError(f"unknown backbone type {kind!r} (expected vllm, hf or mock)")


__all__ = ["build_backbone"]
