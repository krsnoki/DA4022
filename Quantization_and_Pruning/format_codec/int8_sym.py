"""Per-tensor symmetric INT8 quantization."""

import numpy as np
from typing import Tuple


def encode_int8_sym(weights_fp32: np.ndarray) -> Tuple[np.ndarray, float]:
    """Per-tensor symmetric INT8 quantization.

    Returns:
        packed: int8 quantized weights
        scale: scale factor for dequantization (max_abs / 127.0)
    """
    max_abs = float(np.max(np.abs(weights_fp32)))
    scale = max_abs / 127.0 if max_abs > 0 else 1.0

    # Quantize to INT8: clip then round
    q = np.clip(np.round(weights_fp32 / scale), -127, 127).astype(np.int8)

    return q, float(scale)


def decode_int8_sym(packed: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize INT8 weights to FP32.

    Args:
        packed: int8 quantized weights
        scale: scale factor for dequantization

    Returns:
        FP32 dequantized weights
    """
    return packed.astype(np.float32) * scale