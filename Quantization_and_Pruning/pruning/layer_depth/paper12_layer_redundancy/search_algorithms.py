"""
search_algorithms.py — Paper #12: Rethinking Layer Redundancy

Two search algorithms for finding which layers to prune:
  1. One-shot   — score each layer independently, remove k lowest-impact ones
  2. Greedy     — iteratively remove the least harmful layer, k times

Both accept any calibration objective function as a callable.

Author: (your name)
Status: implementing
"""

import copy
from typing import Callable
from depth_pruner import prune_layers, get_num_layers


# ---------------------------------------------------------------------------
# Algorithm 1: One-Shot Top-k
# ---------------------------------------------------------------------------

def one_shot_search(model, objective_fn: Callable, k: int, device: str = "cuda") -> list[int]:
    """
    Score each layer by its individual removal impact on objective_fn.
    Remove the k layers with lowest impact (most redundant).

    This is an O(N) algorithm — N forward passes through the calibration set.

    Args:
        model: base (unpruned) HuggingFace model.
        objective_fn: callable(model) → float. Lower = better model quality.
        k: number of layers to remove.
        device: 'cuda' or 'cpu'.

    Returns:
        List of k layer indices to remove (sorted ascending).

    Example:
        >>> indices = one_shot_search(model, perplexity_fn, k=7)
        >>> pruned  = prune_layers(model, indices)
    """
    # TODO: implement
    # Hint:
    #   baseline = objective_fn(model)
    #   for i in range(get_num_layers(model)):
    #       candidate = prune_layers(copy.deepcopy(model), [i])
    #       score = objective_fn(candidate)
    #       impact = score - baseline   # positive = removing layer i hurt quality
    #   sort by impact ascending, take first k indices
    raise NotImplementedError("one_shot_search not yet implemented")


# ---------------------------------------------------------------------------
# Algorithm 2: Greedy Iterative
# ---------------------------------------------------------------------------

def greedy_search(model, objective_fn: Callable, k: int, device: str = "cuda") -> list[int]:
    """
    Iteratively remove one layer at a time, always choosing the layer
    whose removal causes the smallest increase in objective_fn.

    This is an O(N*k) algorithm — N*k forward passes.

    Args:
        model: base (unpruned) HuggingFace model.
        objective_fn: callable(model) → float. Lower = better.
        k: number of layers to remove.
        device: 'cuda' or 'cpu'.

    Returns:
        List of k original layer indices that were removed (in removal order).

    Note:
        Layer indices are tracked in terms of the ORIGINAL model's indexing.
    """
    # TODO: implement
    # Hint:
    #   current_model = copy.deepcopy(model)
    #   removed_original_indices = []
    #   index_map = list(range(get_num_layers(model)))  # tracks original indices
    #   for step in range(k):
    #       find i in current model that minimises objective_fn after removal
    #       prune that layer from current_model
    #       record index_map[i] as removed
    #       update index_map
    raise NotImplementedError("greedy_search not yet implemented")
