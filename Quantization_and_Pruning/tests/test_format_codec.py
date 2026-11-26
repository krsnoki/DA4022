"""Tests for floating-point formats, INT8 quantization, and fault injection."""

import numpy as np
import pytest

from Quantization_and_Pruning.bit_flip_injector import BitFlipInjector
from Quantization_and_Pruning.format_codec import (
    decode_e4m11,
    decode_e5m10,
    decode_e8m7,
    decode_int8_sym,
    encode_e4m11,
    encode_e5m10,
    encode_e8m7,
    encode_int8_sym,
)


def bit_count(values: np.ndarray) -> int:
    return sum(bin(int(value)).count("1") for value in values.ravel())


@pytest.mark.parametrize(
    "encode,decode",
    [
        (encode_e4m11, decode_e4m11),
        (encode_e5m10, decode_e5m10),
        (encode_e8m7, decode_e8m7),
    ],
)
def test_float_formats_roundtrip_values_without_double_scaling(encode, decode):
    values = np.array([0.0, -0.0, 0.5, 1.0, 3.14, -3.14], dtype=np.float32)
    packed, scale = encode(values)

    decoded = decode(packed, scale)

    assert decoded.shape == values.shape
    assert decoded.dtype == np.float32
    assert np.allclose(decoded, values, rtol=0.01, atol=1e-6)


def test_e4m11_paper_test_vector():
    packed, scale = encode_e4m11(np.array([3.14], dtype=np.float32))

    assert packed[0] == 0x448F
    assert np.isclose(decode_e4m11(packed, scale)[0], 3.1396, rtol=1e-4)


def test_e4m11_max_finite_value():
    packed, scale = encode_e4m11(np.array([255.9375], dtype=np.float32))

    assert np.isclose(decode_e4m11(packed, scale)[0], 255.9375)


def test_int8_roundtrip_and_zero():
    values = np.array([0.0, 1.0, -1.0, -2.5, 3.0], dtype=np.float32)
    packed, scale = encode_int8_sym(values)
    decoded = decode_int8_sym(packed, scale)

    assert decoded[0] == 0
    assert np.allclose(decoded, values, rtol=0.1)


def test_int8_scale():
    values = np.array([10.0, -20.0, 5.0], dtype=np.float32)

    _, scale = encode_int8_sym(values)

    assert np.isclose(scale, np.max(np.abs(values)) / 127.0)


@pytest.mark.parametrize(
    "bit_group,expected_positions",
    [
        ("exponent", [11, 12, 13, 14]),
        ("sign", [15]),
        ("mantissa_MSB", [6, 7, 8, 9, 10]),
        ("mantissa_LSB", [0, 1, 2, 3, 4, 5]),
    ],
)
def test_injector_uses_e4m11_bit_positions(bit_group, expected_positions):
    injector = BitFlipInjector(seed=42)
    packed = np.array([0x448F], dtype=np.uint16)

    flipped, metadata = injector.inject(packed, k=1, bit_group=bit_group)

    assert metadata["bit_positions"] == expected_positions
    assert bit_count(packed ^ flipped) == 1


def test_exponent_positions_follow_format_width():
    assert BitFlipInjector._get_bit_positions("exponent", 16, 5) == [
        10, 11, 12, 13, 14
    ]
    assert BitFlipInjector._get_bit_positions("exponent", 16, 8) == list(
        range(7, 15)
    )


def test_same_seed_produces_same_faults():
    packed = np.array([0x448F, 0x5A3E], dtype=np.uint16)
    first, first_metadata = BitFlipInjector(seed=123).inject(
        packed, k=2, bit_group="exponent"
    )
    second, second_metadata = BitFlipInjector(seed=123).inject(
        packed, k=2, bit_group="exponent"
    )

    assert np.array_equal(first, second)
    assert first_metadata == second_metadata


def test_zero_faults_preserve_the_buffer():
    packed = np.array([0x448F, 0x5A3E], dtype=np.uint16)

    flipped, _ = BitFlipInjector(seed=42).inject(
        packed, k=0, bit_group="exponent"
    )

    assert np.array_equal(flipped, packed)


def test_injector_rejects_negative_fault_count():
    with pytest.raises(ValueError, match="non-negative"):
        BitFlipInjector(seed=42).inject(
            np.array([0x448F], dtype=np.uint16), k=-1
        )


def test_clamp_exponent_e5m10_and_e8m7():
    # E5M10 with all-ones exponent (0x7C00 is +inf in IEEE FP16)
    all_ones_exp_e5m10 = np.array([0x7C00], dtype=np.uint16)
    assert np.isinf(decode_e5m10(all_ones_exp_e5m10, clamp_exponent=False)[0])
    decoded_clamped_e5m10 = decode_e5m10(all_ones_exp_e5m10, clamp_exponent=True)[0]
    assert np.isfinite(decoded_clamped_e5m10)
    assert decoded_clamped_e5m10 > 0

    # E8M7 with all-ones exponent (0x7F80 is +inf in BF16)
    all_ones_exp_e8m7 = np.array([0x7F80], dtype=np.uint16)
    assert np.isinf(decode_e8m7(all_ones_exp_e8m7, clamp_exponent=False)[0])
    decoded_clamped_e8m7 = decode_e8m7(all_ones_exp_e8m7, clamp_exponent=True)[0]
    assert np.isfinite(decoded_clamped_e8m7)
    assert decoded_clamped_e8m7 > 0

