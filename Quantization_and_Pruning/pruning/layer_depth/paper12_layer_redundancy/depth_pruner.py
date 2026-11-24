"""
depth_pruner.py — Paper #12: Rethinking Layer Redundancy

Removes Transformer blocks from a HuggingFace LLaMA-style model
given a list of layer indices to prune.

Author: (your name)
Status: implementing
"""

import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_num_layers(model) -> int:
    """Return the total number of Transformer blocks in the model."""
    return len(model.model.layers)


def prune_layers(model, layer_indices_to_remove: list[int], inplace: bool = False):
    """
    Remove Transformer blocks at the specified indices.

    Args:
        model: HuggingFace CausalLM model (LLaMA-style).
        layer_indices_to_remove: list of 0-based layer indices to remove.
        inplace: if False (default), operates on a deep copy.

    Returns:
        Pruned model with len(layers) - k blocks.

    Example:
        >>> pruned = prune_layers(model, [10, 11, 12, 13, 14, 15, 16])
        >>> print(get_num_layers(pruned))  # original - 7
    """
    if not inplace:
        model = copy.deepcopy(model)

    remove_set = set(layer_indices_to_remove)
    original_layers = model.model.layers

    kept = nn.ModuleList(
        [layer for i, layer in enumerate(original_layers) if i not in remove_set]
    )
    model.model.layers = kept
    return model


def load_model_and_tokenizer(model_id: str, revision: str = "main", dtype=torch.float16, device: str = "cuda"):
    """Load a HuggingFace model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    return model, tokenizer
