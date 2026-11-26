"""Seeded bit-flip injection for packed weight buffers."""

from typing import List, Optional, Tuple

import numpy as np


class BitFlipInjector:
    """Inject random XOR bit flips into a packed 16-bit weight buffer."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def inject(
        self,
        packed: np.ndarray,
        k: int,
        bit_group: str = "exponent",
        total_bits: int = 16,
        exponent_bits: int = 4,
    ) -> Tuple[np.ndarray, dict]:
        """Return a copy with k injected flips and metadata describing the draws."""
        if total_bits != 16 or packed.dtype != np.uint16:
            raise ValueError("packed must be a uint16 buffer with total_bits=16")
        if not 1 <= exponent_bits < total_bits:
            raise ValueError("exponent_bits must be between 1 and total_bits - 1")
        if k < 0:
            raise ValueError("k must be non-negative")

        bit_positions = self._get_bit_positions(
            bit_group, total_bits, exponent_bits
        )
        if k and packed.size == 0:
            raise ValueError("cannot inject faults into an empty buffer")

        flipped = packed.copy()
        indices = self.rng.choice(packed.size, size=k, replace=True)
        for index in indices:
            bit_offset = int(self.rng.choice(bit_positions))
            flipped.flat[int(index)] ^= 1 << bit_offset

        metadata = {
            "indices": indices.tolist(),
            "bit_group": bit_group,
            "bit_positions": bit_positions,
            "k": k,
            "total_bits": total_bits,
            "exponent_bits": exponent_bits,
            "n_weights": packed.size,
        }
        return flipped, metadata

    @staticmethod
    def _get_bit_positions(
        bit_group: str, total_bits: int, exponent_bits: int = 4
    ) -> List[int]:
        """Return positions for a sign, exponent, or mantissa bit group."""
        mantissa_bits = total_bits - exponent_bits - 1
        if bit_group == "sign":
            return [total_bits - 1]
        if bit_group == "exponent":
            return list(range(mantissa_bits, mantissa_bits + exponent_bits))
        if bit_group == "mantissa_MSB":
            upper_width = mantissa_bits // 2
            if upper_width == 0:
                raise ValueError("format has no upper mantissa bit group")
            return list(range(mantissa_bits - upper_width, mantissa_bits))
        if bit_group == "mantissa_LSB":
            upper_width = mantissa_bits // 2
            return list(range(mantissa_bits - upper_width))
        raise ValueError(
            "Unknown bit_group. Choose from: exponent, sign, "
            "mantissa_MSB, mantissa_LSB"
        )
