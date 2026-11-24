"""
search_algorithms.py — Paper #12: Rethinking Layer Redundancy

Two search algorithms for finding which layers to prune:
  1. One-shot   — score each layer independently, remove k lowest-impact ones  (O(N))
  2. Greedy     — iteratively remove the least harmful layer, k times          (O(N·k))

Both accept any calibration objective function as a callable(model) → float,
where LOWER return value means BETTER model quality.

Reference: Song et al. 2024 (greedy), Men et al. 2025 (one-shot/ShortGPT).
"""

import copy
from typing import Callable

from depth_pruner import prune_layers, get_num_layers


# ---------------------------------------------------------------------------
# Algorithm 1: One-Shot Top-k
# ---------------------------------------------------------------------------

def one_shot_search(model, objective_fn: Callable, k: int,
                    device: str = "cuda", verbose: bool = True) -> list:
    """
    Score each layer by its individual removal impact on objective_fn.
    Remove the k layers whose removal causes the SMALLEST increase in loss
    (i.e. the most redundant layers).

    Complexity: O(N) objective evaluations.

    Args:
        model:       Base (unpruned) HuggingFace CausalLM.
        objective_fn: callable(model) → float.  Lower = better quality.
        k:           Number of layers to remove.
        device:      'cuda' or 'cpu'.
        verbose:     Print per-layer scores.

    Returns:
        Sorted list of k layer indices (in original model indexing) to remove.
    """
    n_layers  = get_num_layers(model)
    baseline  = objective_fn(model)

    if verbose:
        print(f"[one_shot] baseline score: {baseline:.4f}  |  N={n_layers}  k={k}")

    impacts = []          # (impact, original_index)
    for i in range(n_layers):
        candidate = prune_layers(copy.deepcopy(model), [i], inplace=True)
        score     = objective_fn(candidate)
        impact    = score - baseline    # positive  → removing layer i hurt quality
        impacts.append((impact, i))
        if verbose:
            print(f"  layer {i:3d}: impact = {impact:+.4f}")
        del candidate

    # Sort ascending: smallest impact first = most redundant
    impacts.sort(key=lambda x: x[0])
    removed = sorted([idx for _, idx in impacts[:k]])

    if verbose:
        print(f"[one_shot] removing layers: {removed}")

    return removed


# ---------------------------------------------------------------------------
# Algorithm 2: Greedy Iterative
# ---------------------------------------------------------------------------

def greedy_search(model, objective_fn: Callable, k: int,
                  device: str = "cuda", verbose: bool = True) -> list:
    """
    Iteratively remove one layer at a time, always choosing the layer
    whose removal causes the smallest increase in objective_fn.

    At each step the pruned model becomes the new starting point, so
    interactions between layers are partially captured.

    Complexity: O(N·k) objective evaluations.

    Args:
        model:        Base (unpruned) HuggingFace CausalLM.
        objective_fn: callable(model) → float.  Lower = better.
        k:            Number of layers to remove.
        device:       'cuda' or 'cpu'.
        verbose:      Print per-step selection.

    Returns:
        List of k ORIGINAL layer indices in removal order.
    """
    current_model = copy.deepcopy(model)
    # Maps current layer positions → original layer indices
    index_map     = list(range(get_num_layers(model)))
    removed_original = []

    for step in range(k):
        n_current = get_num_layers(current_model)
        best_impact   = float("inf")
        best_local_i  = -1

        for i in range(n_current):
            candidate = prune_layers(copy.deepcopy(current_model), [i], inplace=True)
            score     = objective_fn(candidate)
            if score < best_impact:
                best_impact  = score
                best_local_i = i
            del candidate

        # Prune the best local index from current model
        orig_idx = index_map[best_local_i]
        removed_original.append(orig_idx)
        current_model = prune_layers(current_model, [best_local_i], inplace=True)
        index_map.pop(best_local_i)

        if verbose:
            print(f"[greedy] step {step+1}/{k}: removed original layer {orig_idx}"
                  f"  (score after removal: {best_impact:.4f})")

    return removed_original
