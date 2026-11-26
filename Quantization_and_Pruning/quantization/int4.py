import torch
from torch import Tensor
from torch.nn.functional import pad


def affine_int4(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x = x.float()
    lo = x.amin(dim=-1, keepdim=True).clamp_max(0)
    hi = x.amax(dim=-1, keepdim=True).clamp_min(0)
    span = hi - lo
    scale = torch.where(span.isinf(), hi / 15 - lo / 15, span / 15)
    scale = torch.where(scale > 0, scale, 1.0)
    zero = (-lo / scale).round().clamp(0, 15).to(torch.uint8)
    codes = ((x / scale).round() + zero).clamp(0, 15).to(torch.uint8)
    return codes, scale, zero


def dequantize_affine(codes: Tensor, scale: Tensor, zero: Tensor) -> Tensor:
    return (codes.float() - zero) * scale


def fake_quantize_int4(x: Tensor) -> Tensor:
    return dequantize_affine(*affine_int4(x)).to(x.dtype)


def pack_nibbles(codes: Tensor) -> Tensor:
    """Pack uint8 codes in [0, 15], with the first code in the low nibble."""
    if codes.dtype != torch.uint8 or codes.ndim == 0:
        raise ValueError("expected uint8 codes with a feature axis")
    if codes.shape[-1] % 2:
        codes = pad(codes, (0, 1))
    return (codes[..., ::2] & 15) | (codes[..., 1::2] << 4)


def unpack_nibbles(packed: Tensor, width: int) -> Tensor:
    if packed.dtype != torch.uint8 or packed.ndim == 0 or width < 0 or packed.shape[-1] != (width + 1) // 2:
        raise ValueError("packed uint8 tensor does not match the requested width")
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)[..., :width]
