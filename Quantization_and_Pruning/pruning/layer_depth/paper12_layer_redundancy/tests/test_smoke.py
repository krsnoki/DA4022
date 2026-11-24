"""
tests/test_smoke.py — Paper #12: Rethinking Layer Redundancy

CPU smoke tests that verify shapes and basic correctness.
Runs without GPU and without downloading any real model weights.
All tests should complete in under 60 seconds.

Run with:
    python -m pytest tests/test_smoke.py -v

Author: (your name)
Status: implementing
"""

import sys
import os
import copy
import torch
import torch.nn as nn
import pytest

# Allow importing from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_pruner import prune_layers, get_num_layers


# ---------------------------------------------------------------------------
# Minimal stub model for CPU testing (no real weights needed)
# ---------------------------------------------------------------------------

class FakeLayer(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return (self.linear(x),)


class FakeModel(nn.Module):
    """Minimal LLaMA-shaped model with configurable number of layers."""
    def __init__(self, n_layers=8, hidden_size=64):
        super().__init__()
        self.model = nn.ModuleDict({
            "layers": nn.ModuleList([FakeLayer(hidden_size) for _ in range(n_layers)])
        })
        self.n_layers = n_layers

    def forward(self, x):
        for layer in self.model["layers"]:
            x = layer(x)[0]
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_num_layers():
    model = FakeModel(n_layers=8)
    assert get_num_layers(model) == 8


def test_prune_layers_count():
    """After pruning k layers, model should have N-k layers."""
    model = FakeModel(n_layers=8)
    pruned = prune_layers(model, [2, 4, 6])
    assert get_num_layers(pruned) == 5


def test_prune_does_not_modify_original():
    """prune_layers with inplace=False must not change the original."""
    model = FakeModel(n_layers=8)
    _ = prune_layers(model, [0, 1, 2], inplace=False)
    assert get_num_layers(model) == 8


def test_prune_inplace_modifies_model():
    """prune_layers with inplace=True should modify the original."""
    model = FakeModel(n_layers=8)
    prune_layers(model, [0], inplace=True)
    assert get_num_layers(model) == 7


def test_prune_output_shape():
    """Pruned model output shape should be unchanged (same hidden size)."""
    model = FakeModel(n_layers=8, hidden_size=64)
    x = torch.randn(2, 10, 64)
    pruned = prune_layers(model, [3, 4], inplace=False)
    out = pruned(x)
    assert out.shape == (2, 10, 64)


def test_prune_zero_layers():
    """Pruning 0 layers should be a no-op."""
    model = FakeModel(n_layers=8)
    pruned = prune_layers(model, [], inplace=False)
    assert get_num_layers(pruned) == 8


def test_prune_all_but_one():
    """Prune all but 1 layer without crashing."""
    model = FakeModel(n_layers=8)
    pruned = prune_layers(model, list(range(7)), inplace=False)
    assert get_num_layers(pruned) == 1
