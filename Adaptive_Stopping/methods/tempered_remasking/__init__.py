"""Tempered remasking for diffusion LMs (TLC / TCT).

Implementation of "A Tale of Two Temperatures: Simple, Efficient, and Diverse
Sampling from Diffusion Language Models" (Olausson, Jazbec, Wang,
Solar-Lezama, Naesseth, Mandt, Nalisnick; ICML 2026). See README.md.
"""

from .remasking import (
    Autoregressive,
    ConfidenceThreshold,
    LowConfidence,
    RandomRemasking,
    RemaskingPolicy,
    TemperedConfidenceThreshold,
    TemperedLowConfidence,
    build_policy,
    get_num_transfer_tokens,
    sample_tokens,
)
from .sampler import DecodingStopPolicy, GenerationResult, SamplerConfig, generate, generate_samples

__all__ = [
    "Autoregressive",
    "ConfidenceThreshold",
    "LowConfidence",
    "RandomRemasking",
    "RemaskingPolicy",
    "TemperedConfidenceThreshold",
    "TemperedLowConfidence",
    "build_policy",
    "get_num_transfer_tokens",
    "sample_tokens",
    "DecodingStopPolicy",
    "GenerationResult",
    "SamplerConfig",
    "generate",
    "generate_samples",
]

__version__ = "0.1.0"
