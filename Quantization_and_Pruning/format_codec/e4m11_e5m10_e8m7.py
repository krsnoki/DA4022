"""E4M11, E5M10, and E8M7 floating-point format encoders/decoders."""

import math
from typing import Tuple

import numpy as np


def _encode_expm(x: np.float32, total_bits: int, exp_bits: int) -> int:
    """Encode a float32 value using a sign, exponent, and mantissa layout."""
    mantissa_bits = total_bits - 1 - exp_bits
    exponent_mask = (1 << exp_bits) - 1
    bias = (1 << (exp_bits - 1)) - 1
    sign = int(np.signbit(x))
    sign_field = sign << (total_bits - 1)

    value = float(abs(x))
    if math.isnan(value):
        return sign_field | (exponent_mask << mantissa_bits) | 1
    if math.isinf(value):
        return sign_field | (exponent_mask << mantissa_bits)
    if value == 0:
        return sign_field

    max_finite_exponent = exponent_mask - 1 - bias
    min_normal_exponent = 1 - bias
    _, exponent = math.frexp(value)
    unbiased_exponent = exponent - 1

    if unbiased_exponent < min_normal_exponent:
        subnormal_unit = math.ldexp(1.0, min_normal_exponent - mantissa_bits)
        mantissa = int(round(value / subnormal_unit))
        if mantissa == 0:
            return sign_field
        if mantissa >= (1 << mantissa_bits):
            exponent_field = 1
            mantissa = 0
        else:
            exponent_field = 0
    else:
        significand = math.ldexp(value, -unbiased_exponent)
        mantissa = int(round((significand - 1.0) * (1 << mantissa_bits)))
        if mantissa == (1 << mantissa_bits):
            mantissa = 0
            unbiased_exponent += 1
        if unbiased_exponent > max_finite_exponent:
            exponent_field = exponent_mask - 1
            mantissa = (1 << mantissa_bits) - 1
        else:
            exponent_field = unbiased_exponent + bias

    return sign_field | (exponent_field << mantissa_bits) | mantissa


def _decode_expm(
    encoded: int,
    total_bits: int,
    exp_bits: int,
    clamp_exponent: bool = False,
) -> float:
    """Decode a sign, exponent, and mantissa value to Python float."""
    sign = (encoded >> (total_bits - 1)) & 1
    mantissa_bits = total_bits - 1 - exp_bits
    exponent_mask = (1 << exp_bits) - 1
    exponent_field = (encoded >> mantissa_bits) & exponent_mask
    mantissa = encoded & ((1 << mantissa_bits) - 1)
    bias = (1 << (exp_bits - 1)) - 1

    if exponent_field == exponent_mask:
        if clamp_exponent:
            exponent_field -= 1
        elif mantissa == 0:
            return -math.inf if sign else math.inf
        else:
            return math.nan

    if exponent_field == 0:
        value = math.ldexp(mantissa / (1 << mantissa_bits), 1 - bias)
    else:
        significand = 1.0 + mantissa / (1 << mantissa_bits)
        value = math.ldexp(significand, exponent_field - bias)

    return -value if sign else value


def _encode_array(weights_fp32: np.ndarray, exp_bits: int) -> np.ndarray:
    packed = np.zeros(weights_fp32.shape, dtype=np.uint16)
    for index, value in enumerate(weights_fp32.ravel()):
        packed.ravel()[index] = _encode_expm(np.float32(value), 16, exp_bits)
    return packed


def _decode_array(
    packed: np.ndarray, exp_bits: int, clamp_exponent: bool = False
) -> np.ndarray:
    decoded = np.zeros(packed.shape, dtype=np.float32)
    for index, value in enumerate(packed.ravel()):
        decoded.ravel()[index] = _decode_expm(
            int(value), 16, exp_bits, clamp_exponent
        )
    return decoded


def encode_e4m11(weights_fp32: np.ndarray) -> Tuple[np.ndarray, float]:
    """Encode FP32 values as E4M11; the format itself represents the value range."""
    return _encode_array(weights_fp32, 4), 1.0


def decode_e4m11(packed: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Decode E4M11 values to FP32, applying an optional external scale."""
    return _decode_array(packed, 4, clamp_exponent=True) * scale


def encode_e5m10(weights_fp32: np.ndarray) -> Tuple[np.ndarray, float]:
    """Encode FP32 values as E5M10; the format itself represents the value range."""
    return _encode_array(weights_fp32, 5), 1.0


def decode_e5m10(
    packed: np.ndarray, scale: float = 1.0, clamp_exponent: bool = False
) -> np.ndarray:
    """Decode E5M10 values to FP32, applying an optional external scale."""
    return _decode_array(packed, 5, clamp_exponent=clamp_exponent) * scale


def encode_e8m7(weights_fp32: np.ndarray) -> Tuple[np.ndarray, float]:
    """Encode FP32 values as E8M7; the format itself represents the value range."""
    return _encode_array(weights_fp32, 8), 1.0


def decode_e8m7(
    packed: np.ndarray, scale: float = 1.0, clamp_exponent: bool = False
) -> np.ndarray:
    """Decode E8M7 values to FP32, applying an optional external scale."""
    return _decode_array(packed, 8, clamp_exponent=clamp_exponent) * scale


def roundtrip_encode_decode(
    weights_fp32: np.ndarray, format_name: str = "e4m11"
) -> np.ndarray:
    """Encode then decode weights, returning the round-tripped result."""
    codecs = {
        "e4m11": (encode_e4m11, decode_e4m11),
        "e5m10": (encode_e5m10, decode_e5m10),
        "e8m7": (encode_e8m7, decode_e8m7),
    }
    try:
        encode, decode = codecs[format_name]
    except KeyError as error:
        raise ValueError(f"Unknown format: {format_name}") from error
    packed, scale = encode(weights_fp32)
    return decode(packed, scale)
