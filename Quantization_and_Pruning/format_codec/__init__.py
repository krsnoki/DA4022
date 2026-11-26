"""Format codec for custom floating-point and integer formats.

Implements ExMY encoding/decoding for E4M11, E5M10, E8M7 and
per-tensor symmetric INT8 quantization.
"""

from .e4m11_e5m10_e8m7 import (
    encode_e4m11, decode_e4m11,
    encode_e5m10, decode_e5m10,
    encode_e8m7, decode_e8m7,
    roundtrip_encode_decode,
    _encode_expm, _decode_expm
)
from .int8_sym import (
    encode_int8_sym,
    decode_int8_sym
)

__all__ = [
    "encode_e4m11", "decode_e4m11",
    "encode_e5m10", "decode_e5m10",
    "encode_e8m7", "decode_e8m7",
    "encode_int8_sym", "decode_int8_sym",
    "roundtrip_encode_decode",
    "_encode_expm", "_decode_expm"
]