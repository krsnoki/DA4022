"""CafeQ: calibration-free weight quantization via learned transformations and adaptive rounding.

Paper: https://openreview.net/forum?id=CBtx52vftl (AdaptFM). Independent re-implementation; no
upstream code was available. See ``PAPER_CARD.md`` in this folder.

Public names are loaded lazily so that ``cafeq.config`` can be imported (e.g. for config
validation) without PyTorch installed.
"""
from importlib import import_module

_EXPORTS = {
    "CafeQConfig": "config",
    "ConfigError": "config",
    "CafeQLinear": "modules",
    "save_quantized": "modules",
    "load_quantized": "modules",
    "quantized_layout": "modules",
    "quantize_model": "apply",
    "apply_vo_transform": "apply",
    "attention_geometry": "apply",
    "AttentionGeometry": "apply",
    "decoder_layers": "apply",
    "learn_block_transform": "block_diagonal",
    "BlockTransformResult": "block_diagonal",
    "surrogate_loss": "block_diagonal",
    "transform_weight": "block_diagonal",
    "effective_weight": "block_diagonal",
    "learn_paired_transform": "paired",
    "PairedTransformResult": "paired",
    "pseudo_loss": "paired",
    "orth_penalty": "paired",
    "paired_error": "paired",
    "adaptive_rounding": "joint_rounding",
    "adaptive_rounding_vo": "joint_rounding",
    "vo_paired_error": "joint_rounding",
}

__all__ = sorted(_EXPORTS)

# An exported name must never equal a submodule name: importing the submodule would set the
# package attribute to the module object and shadow the lazily exported function.
assert not set(_EXPORTS) & set(_EXPORTS.values()), "exported name collides with a submodule"


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
