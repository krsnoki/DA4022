"""Shared quantization primitives (uniform quantizer, structured transforms)."""
from .transforms import (
    block_diag_dense,
    block_diag_matmul,
    hadamard_matrix,
    pick_block_size,
    random_block_diagonal,
    random_hadamard,
    random_orthogonal,
)
from .uniform import (
    QuantSpec,
    dequantize_codes,
    group_ranges,
    quantize_dequantize,
    quantize_paper_layout,
    quantize_to_codes,
    relative_l2,
)

__all__ = [
    "QuantSpec",
    "quantize_dequantize",
    "quantize_to_codes",
    "dequantize_codes",
    "group_ranges",
    "relative_l2",
    "quantize_paper_layout",
    "hadamard_matrix",
    "random_hadamard",
    "random_orthogonal",
    "block_diag_matmul",
    "block_diag_dense",
    "pick_block_size",
    "random_block_diagonal",
]
