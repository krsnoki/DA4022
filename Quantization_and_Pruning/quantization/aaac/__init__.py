"""AAAC: Activation-Aware Adaptive Codebooks for 4-bit LLM weight quantization."""

from .assignment import assign_selection_groups, encode_indices, nearest_codeword
from .codebook_init import initial_codebooks, quantile_codebook, weighted_quantiles
from .codec import (
    PackedAAACLayer,
    decode_scales_and_selectors,
    encode_scales_with_selector,
    pack_aaac_fit,
    pack_aaac_layer,
    pack_int4_indices,
    pack_selector_bits,
    reconstruct_aaac_weight,
    unpack_int4_indices,
    unpack_selector_bits,
)
from .config import NUM_CODEWORDS, NUM_TABLES, AAACConfig
from .fit import AAACLayerFit, fit_aaac_layer, fit_aaac_model
from .linear import AAACLinear, measure_model_storage_bytes, replace_linear_layers_with_aaac
from .rtn import RTNLinear, quantize_model_rtn, quantize_weight_rtn_int4, reconstruct_rtn_weight
from .scaling import compute_group_scales, expand_group_scales, normalize_weight

__all__ = [
    "NUM_CODEWORDS",
    "NUM_TABLES",
    "AAACConfig",
    "AAACLayerFit",
    "AAACLinear",
    "PackedAAACLayer",
    "RTNLinear",
    "assign_selection_groups",
    "compute_group_scales",
    "decode_scales_and_selectors",
    "encode_indices",
    "encode_scales_with_selector",
    "expand_group_scales",
    "fit_aaac_layer",
    "fit_aaac_model",
    "initial_codebooks",
    "measure_model_storage_bytes",
    "nearest_codeword",
    "normalize_weight",
    "pack_aaac_fit",
    "pack_aaac_layer",
    "pack_int4_indices",
    "pack_selector_bits",
    "quantile_codebook",
    "quantize_model_rtn",
    "quantize_weight_rtn_int4",
    "reconstruct_aaac_weight",
    "reconstruct_rtn_weight",
    "replace_linear_layers_with_aaac",
    "unpack_int4_indices",
    "unpack_selector_bits",
    "weighted_quantiles",
]