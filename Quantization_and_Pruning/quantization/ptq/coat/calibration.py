from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass

import torch
from torch import Tensor

from Backbones.Generative_Pretrained.llama import LlamaAdapter
from Dataloaders.causal_lm import TokenBatch


@dataclass(frozen=True)
class Covariances:
    r1: Tensor
    r2: tuple[Tensor, ...]


class WelfordCovariance:
    def __init__(
        self, width: int, *, device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64, chunk_size: int = 2048,
    ):
        if width < 1 or chunk_size < 1:
            raise ValueError("width and chunk_size must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("covariance statistics require float32 or float64")
        self.count = 0
        self.mean = torch.zeros(width, device=device, dtype=dtype)
        self.scatter = torch.zeros((width, width), device=device, dtype=dtype)
        self.chunk_size = chunk_size

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        for rows in x.split(self.chunk_size):
            n = rows.shape[0]
            if not n:
                continue
            rows = rows.to(self.mean)
            mean = rows.mean(dim=0)
            centered = rows - mean
            self.scatter.addmm_(centered.T, centered)
            total = self.count + n
            delta = mean - self.mean
            self.scatter.addr_(delta, delta, alpha=self.count * n / total)
            self.mean.add_(delta, alpha=n / total)
            self.count = total

    def covariance(self) -> Tensor:
        if self.count < 2:
            raise ValueError("covariance requires at least two observations")
        cov = self.scatter / (self.count - 1)
        cov = cov * 0.5 + cov.T * 0.5
        if not torch.isfinite(cov).all() or not torch.isfinite(self.mean).all():
            raise ValueError("nonfinite covariance statistics")
        return cov


def collect_covariances(
    adapter: LlamaAdapter,
    batches: Iterable[TokenBatch],
    *,
    stats_dtype: torch.dtype = torch.float64,
    chunk_size: int = 2048,
) -> Covariances:
    model = adapter.model
    device = adapter.embeddings.weight.device
    r1 = WelfordCovariance(model.config.hidden_size, device=device, dtype=stats_dtype, chunk_size=chunk_size)
    r2 = [
        WelfordCovariance(layer.self_attn.head_dim, device=device, dtype=stats_dtype, chunk_size=chunk_size)
        for layer in adapter.layers
    ]
    valid: Tensor

    def norm_input(norm, args):
        x = args[0].flatten(0, 1).index_select(0, valid)
        xf = x.float()
        pure = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + norm.variance_epsilon)
        r1.update(pure.to(x.dtype))

    def attention_input(stat: WelfordCovariance):
        def collect(_, args):
            x = args[0].flatten(0, 1).index_select(0, valid)
            stat.update(x.reshape(-1, stat.mean.numel()))

        return collect

    training = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with ExitStack() as hooks, torch.inference_mode():
            for i, stat in enumerate(r2):
                for norm in adapter.norms(i):
                    hooks.enter_context(norm.register_forward_pre_hook(norm_input))
                proj = adapter.projections(i)["o_proj"]
                hooks.enter_context(proj.register_forward_pre_hook(attention_input(stat)))
            for batch in batches:
                batch = batch.to(device)
                valid = batch.attention_mask.flatten().nonzero(as_tuple=True)[0]
                model.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    finally:
        for module, mode in training:
            module.training = mode

    return Covariances(r1.covariance(), tuple(stat.covariance() for stat in r2))
