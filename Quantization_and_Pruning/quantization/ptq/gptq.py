from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass

import torch
from torch import Tensor

from Dataloaders.causal_lm import TokenBatch

from ..int4 import pack_nibbles
from .coat.llama import RotatedLlama


@dataclass(frozen=True)
class GPTQConfig:
    block_size: int = 128
    damp_percent: float = 0.01


@dataclass(frozen=True)
class PackedWeight:
    codes: Tensor
    scale: Tensor
    in_features: int


@torch.no_grad()
def collect_input_grams(
    rotated: RotatedLlama, batches: Iterable[TokenBatch],
) -> Iterator[tuple[int, dict[str, Tensor]]]:
    adapter = rotated.adapter
    decoder = rotated.model.model
    device = adapter.embeddings.weight.device
    replay = []
    has_tokens = False
    for batch in batches:
        mask = batch.attention_mask.cpu()
        has_tokens |= bool(mask.any())
        hidden = adapter.embeddings(batch.input_ids.to(device)).cpu()
        replay.append((mask, hidden))
    if not has_tokens:
        raise ValueError("GPTQ requires at least one valid calibration token")

    # Q/K/V and gate/up share their inputs in the floating-point Llama block.
    groups = (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"), ("o_proj",), ("down_proj",))
    valid: Tensor

    def collect(gram: Tensor):
        def update(_, args):
            x = args[0].flatten(0, 1).index_select(0, valid).float()
            for rows in x.split(2048):
                gram.addmm_(rows.T, rows)

        return update

    for i, layer in enumerate(adapter.layers):
        projections = adapter.projections(i)
        grams = {}
        with rotated.online_rotations(), ExitStack() as hooks:
            for first, *others in groups:
                proj = projections[first]
                gram = torch.zeros((proj.in_features, proj.in_features), device=device, dtype=torch.float32)
                for name in (first, *others):
                    grams[name] = gram
                hooks.enter_context(proj.register_forward_pre_hook(collect(gram)))

            for j, (mask, hidden) in enumerate(replay):
                x = hidden.to(device)
                mask_device = mask.to(device)
                valid = mask_device.flatten().nonzero(as_tuple=True)[0]
                pos = torch.arange(x.shape[1], device=device)
                causal = decoder._update_causal_mask(mask_device, x, pos, None, False)
                position_ids = pos.unsqueeze(0)
                rope = decoder.rotary_emb(x, position_ids)
                out = layer(x, attention_mask=causal, position_ids=position_ids,
                            cache_position=pos, position_embeddings=rope)[0]
                replay[j] = mask, out.cpu()
        yield i, grams


@torch.no_grad()
def gptq_quantize(weight: Tensor, gram: Tensor, config: GPTQConfig = GPTQConfig()) -> PackedWeight:
    if weight.ndim != 2 or gram.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("GPTQ needs output-by-input weights and an input Gram matrix")
    if config.block_size < 1 or not 0 < config.damp_percent < 1:
        raise ValueError("GPTQ requires a positive block size and damping fraction in (0, 1)")

    w = weight.to(dtype=torch.float32, copy=True)
    h = gram.to(device=w.device, dtype=w.dtype, copy=True)
    if not torch.isfinite(w).all() or not torch.isfinite(h).all():
        raise ValueError("GPTQ weights and Gram must be finite")
    scale = w.abs().amax(dim=1) / 7
    scale = torch.where(scale > 0, scale, 1.0)
    diag = h.diagonal()
    dead = diag == 0
    damping = config.damp_percent * diag.mean()
    diag.masked_fill_(dead, 1.0).add_(damping)
    w.masked_fill_(dead, 0)
    factor = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    del h
    codes = torch.empty_like(w, dtype=torch.uint8)
    width = w.shape[1]

    for start in range(0, width, config.block_size):
        stop = min(start + config.block_size, width)
        errors = torch.empty_like(w[:, start:stop])
        for j in range(start, stop):
            col = w[:, j]
            q = (col / scale).round().clamp(-7, 7)
            codes[:, j] = (q + 7).to(torch.uint8)
            err = (col - q * scale) / factor[j, j]
            errors[:, j - start] = err
            w[:, j + 1:stop].addr_(err, factor[j, j + 1:stop], alpha=-1)
        w[:, stop:].addmm_(errors, factor[start:stop, stop:], alpha=-1)

    return PackedWeight(pack_nibbles(codes), scale, width)
