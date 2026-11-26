"""Qwen/Qwen2.5-Math-1.5B-Instruct backbone wrapper and metadata."""

from Backbones.Generative_Pretrained.qwen2_5_math_1_5b_instruct.wrapper import (
    BackboneConfig,
    Qwen25Math15BInstructBackbone,
    get_model_metadata,
)

__all__ = [
    "BackboneConfig",
    "Qwen25Math15BInstructBackbone",
    "get_model_metadata",
]
