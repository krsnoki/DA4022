"""
tests/test_smoke.py — Paper #12: Rethinking Layer Redundancy

CPU smoke tests — no GPU, no real model weights needed.
Verifies shapes, dtypes, and interface contracts of all core modules.

Run with:
    python -m pytest tests/test_smoke.py -v

All tests complete in under 60 seconds on CPU.
"""

import sys
import os
import copy
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_pruner import prune_layers, get_num_layers
from search_algorithms import one_shot_search, greedy_search


# ---------------------------------------------------------------------------
# Minimal stub model (no real weights, CPU-only)
# ---------------------------------------------------------------------------

class FakeLayer(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.linear = nn.Linear(h, h)

    def forward(self, x):
        return (self.linear(x),)


class FakeModel(nn.Module):
    """Minimal LLaMA-shaped model: has model.layers (ModuleList)."""
    def __init__(self, n_layers=8, hidden=32):
        super().__init__()
        self.model = type("M", (), {
            "layers": nn.ModuleList([FakeLayer(hidden) for _ in range(n_layers)])
        })()

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


def dummy_objective(model):
    """Fake objective: returns number of remaining layers (lower = fewer layers)."""
    return float(get_num_layers(model))


# ---------------------------------------------------------------------------
# depth_pruner tests
# ---------------------------------------------------------------------------

class TestDepthPruner:

    def test_get_num_layers(self):
        m = FakeModel(n_layers=8)
        assert get_num_layers(m) == 8

    def test_prune_correct_count(self):
        m = FakeModel(n_layers=8)
        p = prune_layers(m, [2, 4, 6])
        assert get_num_layers(p) == 5

    def test_prune_no_inplace_original_unchanged(self):
        m = FakeModel(n_layers=8)
        _ = prune_layers(m, [0, 1, 2], inplace=False)
        assert get_num_layers(m) == 8

    def test_prune_inplace_modifies_model(self):
        m = FakeModel(n_layers=8)
        prune_layers(m, [0], inplace=True)
        assert get_num_layers(m) == 7

    def test_prune_output_shape_unchanged(self):
        m = FakeModel(n_layers=8, hidden=32)
        x = torch.randn(2, 10, 32)
        p = prune_layers(m, [3, 4], inplace=False)
        assert p(x).shape == (2, 10, 32)

    def test_prune_zero_layers_noop(self):
        m = FakeModel(n_layers=8)
        p = prune_layers(m, [], inplace=False)
        assert get_num_layers(p) == 8

    def test_prune_all_but_one(self):
        m = FakeModel(n_layers=8)
        p = prune_layers(m, list(range(7)), inplace=False)
        assert get_num_layers(p) == 1

    def test_prune_first_layer(self):
        m = FakeModel(n_layers=4)
        p = prune_layers(m, [0], inplace=False)
        assert get_num_layers(p) == 3

    def test_prune_last_layer(self):
        m = FakeModel(n_layers=4)
        p = prune_layers(m, [3], inplace=False)
        assert get_num_layers(p) == 3


# ---------------------------------------------------------------------------
# search_algorithm tests
# ---------------------------------------------------------------------------

class TestSearchAlgorithms:

    def test_one_shot_returns_k_indices(self):
        m = FakeModel(n_layers=8)
        removed = one_shot_search(m, dummy_objective, k=3, device="cpu", verbose=False)
        assert len(removed) == 3

    def test_one_shot_indices_in_range(self):
        m = FakeModel(n_layers=8)
        removed = one_shot_search(m, dummy_objective, k=3, device="cpu", verbose=False)
        assert all(0 <= idx < 8 for idx in removed)

    def test_one_shot_no_duplicate_indices(self):
        m = FakeModel(n_layers=8)
        removed = one_shot_search(m, dummy_objective, k=4, device="cpu", verbose=False)
        assert len(set(removed)) == len(removed)

    def test_one_shot_pruned_model_has_correct_layers(self):
        m = FakeModel(n_layers=8)
        removed = one_shot_search(m, dummy_objective, k=3, device="cpu", verbose=False)
        p = prune_layers(m, removed, inplace=False)
        assert get_num_layers(p) == 5

    def test_greedy_returns_k_indices(self):
        m = FakeModel(n_layers=6)
        removed = greedy_search(m, dummy_objective, k=2, device="cpu", verbose=False)
        assert len(removed) == 2

    def test_greedy_indices_in_original_range(self):
        m = FakeModel(n_layers=6)
        removed = greedy_search(m, dummy_objective, k=2, device="cpu", verbose=False)
        assert all(0 <= idx < 6 for idx in removed)

    def test_greedy_no_duplicate_indices(self):
        m = FakeModel(n_layers=6)
        removed = greedy_search(m, dummy_objective, k=3, device="cpu", verbose=False)
        assert len(set(removed)) == len(removed)

    def test_greedy_pruned_model_has_correct_layers(self):
        m = FakeModel(n_layers=6)
        removed = greedy_search(m, dummy_objective, k=2, device="cpu", verbose=False)
        p = prune_layers(m, removed, inplace=False)
        assert get_num_layers(p) == 4


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_pipeline_one_shot(self):
        """End-to-end: load model → search → prune → forward pass."""
        m       = FakeModel(n_layers=8, hidden=32)
        removed = one_shot_search(m, dummy_objective, k=3, device="cpu", verbose=False)
        pruned  = prune_layers(m, removed, inplace=False)
        x       = torch.randn(1, 5, 32)
        out     = pruned(x)
        assert out.shape == (1, 5, 32)

    def test_full_pipeline_greedy(self):
        """End-to-end: load model → greedy search → prune → forward pass."""
        m       = FakeModel(n_layers=6, hidden=32)
        removed = greedy_search(m, dummy_objective, k=2, device="cpu", verbose=False)
        pruned  = prune_layers(m, removed, inplace=False)
        x       = torch.randn(1, 5, 32)
        out     = pruned(x)
        assert out.shape == (1, 5, 32)

    def test_original_model_unchanged_after_pipeline(self):
        """Original model must be unmodified after full pipeline."""
        m       = FakeModel(n_layers=8)
        removed = one_shot_search(m, dummy_objective, k=3, device="cpu", verbose=False)
        _       = prune_layers(m, removed, inplace=False)
        assert get_num_layers(m) == 8
