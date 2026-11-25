from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .config import NUM_CODEWORDS, NUM_TABLES
from .fit import AAACLayerFit
from .scaling import compute_group_scales, expand_group_scales


def scale_floor_for_dtype(scale_dtype: torch.dtype) -> float:
    """Documented positive floor shared with scaling.py (torch.finfo(scale_dtype).tiny)."""
    return float(torch.finfo(scale_dtype).tiny)


def pack_int4_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit indices (0..15) per uint8 byte: low nibble = even col, high nibble = odd col."""
    if indices.ndim != 2:
        raise ValueError(f"Expected 2D indices tensor, got shape {tuple(indices.shape)}")
    _, in_features = indices.shape
    if in_features % 2 != 0:
        raise ValueError(f"in_features ({in_features}) must be even for 4-bit packing")
    idx_i64 = indices.to(torch.int64)
    if idx_i64.numel() > 0 and (idx_i64.min().item() < 0 or idx_i64.max().item() > 15):
        raise ValueError("4-bit indices must lie strictly in [0, 15]")

    idx_u8 = indices.to(torch.uint8)
    low = idx_u8[:, 0::2] & 0x0F
    high = (idx_u8[:, 1::2] & 0x0F) << 4
    return (low | high).contiguous()


def unpack_int4_indices(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Unpack uint8 [out, in // 2] back into uint8 [out, in] with values in 0..15."""
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError(f"Expected 2D uint8 tensor, got {packed.dtype} {tuple(packed.shape)}")
    if packed.shape[1] * 2 != in_features:
        raise ValueError(f"Packed width {packed.shape[1]} does not match in_features={in_features}")

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack((low, high), dim=-1).reshape(packed.shape[0], in_features)


def pack_selector_bits(selectors: torch.Tensor) -> torch.Tensor:
    """Pack bool or 0/1 selector tensor into a flat uint8 bitset (8 selectors/byte, LSB-first)."""
    if selectors.dtype == torch.bool:
        flat = selectors.reshape(-1).to(torch.uint8)
    else:
        flat_i64 = selectors.reshape(-1).to(torch.int64)
        if flat_i64.numel() > 0 and (flat_i64.min().item() < 0 or flat_i64.max().item() > 1):
            raise ValueError("Selectors must be bool or binary (0/1)")
        flat = flat_i64.to(torch.uint8)

    num_bits = flat.numel()
    pad = (8 - (num_bits % 8)) % 8
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8, device=flat.device)], dim=0)

    bits = flat.reshape(-1, 8)
    shifts = torch.arange(8, dtype=torch.uint8, device=flat.device)
    packed = (bits << shifts).sum(dim=-1, dtype=torch.int32).to(torch.uint8)
    return packed.contiguous()


def unpack_selector_bits(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Unpack flat uint8 bitset back into bool [shape[0], shape[1]]."""
    if packed.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 bitset, got {packed.dtype}")
    num_bits = shape[0] * shape[1]
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 1).reshape(-1)
    return unpacked[:num_bits].to(torch.bool).reshape(shape)


def encode_scales_with_selector(
    scales: torch.Tensor,
    selectors: torch.Tensor,
    group_size: int,
    selection_size: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Encode selectors according to the AAAC contract:
      - S == g: store selector in sign of scale (+abs(scale) for False/0, -abs(scale) for True/1).
      - S <  g: keep scales strictly positive and store selectors in a packed uint8 bitset.
    """
    if group_size % selection_size != 0:
        raise ValueError(
            f"selection_size ({selection_size}) must divide group_size ({group_size})"
        )

    finfo = torch.finfo(scales.dtype)
    pos_scales = scales.detach().abs().clamp(min=finfo.tiny, max=finfo.max).to(scales.dtype)
    sel_bool = selectors.detach().to(torch.bool)

    if selection_size == group_size:
        if sel_bool.shape != pos_scales.shape:
            raise ValueError(
                f"For S==g, selectors shape {tuple(sel_bool.shape)} must match scales {tuple(pos_scales.shape)}"
            )
        signed = torch.where(sel_bool, -pos_scales, pos_scales)
        return signed.contiguous(), None

    return pos_scales.contiguous(), pack_selector_bits(sel_bool)


def decode_scales_and_selectors(
    scales: torch.Tensor,
    selector_bitset: Optional[torch.Tensor],
    out_features: int,
    in_features: int,
    group_size: int,
    selection_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (abs_scales [out, in//g], selectors bool [out, in//S])."""
    if selection_size == group_size:
        if selector_bitset is not None:
            raise ValueError("selector_bitset must be None when S == g")
        selectors = scales < 0
        return scales.abs(), selectors

    if selector_bitset is None:
        raise ValueError("selector_bitset is required when S < g")
    selector_shape = (out_features, in_features // selection_size)
    selectors = unpack_selector_bits(selector_bitset, selector_shape)
    return scales.abs(), selectors


@dataclass
class PackedAAACLayer:
    packed_indices: torch.Tensor              # uint8 [out, in // 2]
    codebooks: torch.Tensor                   # bfloat16 [2, 16]
    scales: torch.Tensor                      # scale_dtype [out, in // g]
    selector_bitset: Optional[torch.Tensor]   # None for S==g, uint8 [ceil(out*(in//S)/8)] for S<g
    out_features: int
    in_features: int
    group_size: int
    selection_size: int

    def packed_bytes_breakdown(self) -> dict[str, int]:
        idx_bytes = self.packed_indices.numel() * self.packed_indices.element_size()
        cb_bytes = self.codebooks.numel() * self.codebooks.element_size()
        scale_bytes = self.scales.numel() * self.scales.element_size()
        sel_bytes = (
            self.selector_bitset.numel() * self.selector_bitset.element_size()
            if self.selector_bitset is not None
            else 0
        )
        return {
            "packed_indices_bytes": idx_bytes,
            "codebooks_bytes": cb_bytes,
            "scales_bytes": scale_bytes,
            "selector_bitset_bytes": sel_bytes,
            "total_packed_bytes": idx_bytes + cb_bytes + scale_bytes + sel_bytes,
        }


def pack_aaac_layer(
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    selectors: torch.Tensor,
    group_size: int = 128,
    selection_size: int = 128,
) -> PackedAAACLayer:
    """Pack AAAC layer tensors into the serialized 4-bit format."""
    if codebooks.shape != (NUM_TABLES, NUM_CODEWORDS):
        raise ValueError(
            f"Expected codebooks shape {(NUM_TABLES, NUM_CODEWORDS)}, got {tuple(codebooks.shape)}"
        )
    out_features, in_features = indices.shape
    packed_indices = pack_int4_indices(indices)
    cb_bf16 = codebooks.detach().to(torch.bfloat16).contiguous()
    encoded_scales, selector_bitset = encode_scales_with_selector(
        scales=scales,
        selectors=selectors,
        group_size=group_size,
        selection_size=selection_size,
    )
    return PackedAAACLayer(
        packed_indices=packed_indices,
        codebooks=cb_bf16,
        scales=encoded_scales,
        selector_bitset=selector_bitset,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        selection_size=selection_size,
    )


def pack_aaac_fit(layer_fit: AAACLayerFit) -> PackedAAACLayer:
    """Directly pack Member 2's AAACLayerFit output."""
    return pack_aaac_layer(
        indices=layer_fit.indices,
        codebooks=layer_fit.codebooks,
        scales=layer_fit.scales,
        selectors=layer_fit.selectors,
        group_size=layer_fit.group_size,
        selection_size=layer_fit.selection_size,
    )


def reconstruct_aaac_weight(
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    selector_bitset: Optional[torch.Tensor],
    out_features: int,
    in_features: int,
    group_size: int,
    selection_size: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Bit-identical dense reconstruction matching AAACLayerFit.dequantize()."""
    indices = unpack_int4_indices(packed_indices, in_features).to(torch.long)
    abs_scales, selectors = decode_scales_and_selectors(
        scales=scales,
        selector_bitset=selector_bitset,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        selection_size=selection_size,
    )
    table_ids = selectors.to(torch.long).repeat_interleave(selection_size, dim=1)
    flat_cb = codebooks.to(torch.float32).reshape(-1)
    values = flat_cb[table_ids * NUM_CODEWORDS + indices]
    expanded_scales = expand_group_scales(abs_scales, group_size).to(values.device)
    return (values * expanded_scales).to(out_dtype)


__all__ = [
    "PackedAAACLayer",
    "compute_group_scales",
    "decode_scales_and_selectors",
    "encode_scales_with_selector",
    "pack_aaac_fit",
    "pack_aaac_layer",
    "pack_int4_indices",
    "pack_selector_bits",
    "reconstruct_aaac_weight",
    "scale_floor_for_dtype",
    "unpack_int4_indices",
    "unpack_selector_bits",
]