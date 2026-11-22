"""Baseline compute-allocation policies for Adaptive_Stopping."""

from Adaptive_Stopping.baselines.allocators import (
    Easy2HardAllocator,
    Hard2EasyAllocator,
    UniformAllocator,
)

__all__ = [
    "Easy2HardAllocator",
    "Hard2EasyAllocator",
    "UniformAllocator",
]
