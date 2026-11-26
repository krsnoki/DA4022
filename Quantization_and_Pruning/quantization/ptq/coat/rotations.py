from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import LlamaConfig

from .calibration import Covariances

HADAMARD_44_URL = "https://neilsloane.com/hadamard/had.44.pal.txt"
HADAMARD_44_SHA256 = "6077980593166a4bfd599208d9618ca4557c0addccf920cdde801753ea4a7f3d"
HADAMARD_172_URL = "https://neilsloane.com/hadamard/had.172.will.txt"
HADAMARD_172_SHA256 = "66cd19c8ae98afaf89df7699b232b24a5d6af15de96ed5a5ca8d84e68f2a38bb"


def _load_sign_matrix(path: str | Path, order: int, checksum: str) -> Tensor:
    rows = Path(path).read_text(encoding="ascii").strip().splitlines()
    if len(rows) != order or any(len(row) != order or set(row) - {"+", "-"} for row in rows):
        raise ValueError(f"expected {order} rows of {order} Hadamard signs")
    signs = "".join(rows)
    if sha256(signs.encode("ascii")).hexdigest() != checksum:
        raise ValueError(f"order-{order} Hadamard checksum mismatch")
    h_mat = torch.tensor([1 if s == "+" else -1 for s in signs], dtype=torch.float64, device="cpu")
    h_mat = h_mat.reshape(order, order)
    if not torch.equal(h_mat.T @ h_mat, order * torch.eye(order, dtype=h_mat.dtype, device=h_mat.device)):
        raise ValueError(f"order-{order} Hadamard factor is not orthogonal")
    return h_mat / sqrt(order)


def load_hadamard44(path: str | Path) -> Tensor:
    return _load_sign_matrix(path, 44, HADAMARD_44_SHA256)


def load_hadamard172(path: str | Path) -> Tensor:
    return _load_sign_matrix(path, 172, HADAMARD_172_SHA256)


def _fwht(x: Tensor) -> Tensor:
    shape = x.shape
    stride = 1
    while stride < shape[-1]:
        pairs = shape[-1] // (2 * stride)
        a, b = x.reshape(*shape[:-1], pairs, 2, stride).unbind(dim=-2)
        x = torch.stack((a + b, a - b), dim=-2).reshape(shape)
        stride *= 2
    return x / sqrt(shape[-1])


class Hadamard(nn.Module):
    def __init__(self, power_of_two: int, factor: Tensor | None = None):
        super().__init__()
        if power_of_two < 1 or power_of_two & (power_of_two - 1):
            raise ValueError("Sylvester Hadamard order must be a power of two")
        if factor is not None and (factor.ndim != 2 or factor.shape[0] != factor.shape[1]):
            raise ValueError("Hadamard factor must be square")
        self.power_of_two = power_of_two
        self.width = power_of_two * (1 if factor is None else factor.shape[0])
        self.register_buffer("factor", factor)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.width:
            raise ValueError(f"expected Hadamard input width {self.width}, got {x.shape[-1]}")
        dtype = x.dtype
        if dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        if self.factor is None:
            return _fwht(x).to(dtype)

        # A row multiplied by A ⊗ B, reshaped as X, becomes A.T @ X @ B.
        shape = x.shape
        x = x.reshape(*shape[:-1], self.power_of_two, self.factor.shape[0]) @ self.factor.to(x)
        return _fwht(x.transpose(-1, -2)).transpose(-1, -2).reshape(shape).to(dtype)


class SignedHadamard(nn.Module):
    def __init__(self, hadamard: Hadamard, signs: Tensor):
        super().__init__()
        self.hadamard = hadamard
        self.register_buffer("signs", signs)

    def forward(self, x: Tensor) -> Tensor:
        return self.hadamard(x * self.signs)


@dataclass(frozen=True)
class RotationDimensions:
    hidden_size: int
    head_dim: int
    intermediate_size: int
    num_layers: int

    @classmethod
    def from_config(cls, cfg: LlamaConfig) -> "RotationDimensions":
        return cls(cfg.hidden_size, cfg.head_dim, cfg.intermediate_size, cfg.num_hidden_layers)


class RotationPlan(nn.Module):
    def __init__(
        self, r1: Tensor, r2: Sequence[Tensor], r3: Sequence[SignedHadamard], r4: Sequence[SignedHadamard],
        *, seed: int,
    ):
        super().__init__()
        self.register_buffer("r1", r1)
        self.register_buffer("r2", torch.stack(tuple(r2)))
        self.r3 = nn.ModuleList(r3)
        self.r4 = nn.ModuleList(r4)
        self.seed = seed


def _hadamard(width: int, factors: Mapping[int, str | Path]) -> Hadamard:
    if width == 5632:
        if 44 not in factors:
            raise ValueError(f"supply factors[44] with a local copy of {HADAMARD_44_URL}")
        return Hadamard(128, load_hadamard44(factors[44]))
    if width == 11008:
        if 172 not in factors:
            raise ValueError(f"supply factors[172] with a local copy of {HADAMARD_172_URL}")
        return Hadamard(64, load_hadamard172(factors[172]))
    return Hadamard(width)


def _coat_rotation(cov: Tensor, hadamard: Hadamard) -> Tensor:
    if cov.shape != (hadamard.width, hadamard.width):
        raise ValueError(f"expected square covariance of width {hadamard.width}")
    cov = cov.to(torch.float64)
    if not torch.isfinite(cov).all():
        raise ValueError("nonfinite covariance")
    _, eigvecs = torch.linalg.eigh(cov * 0.5 + cov.T * 0.5)
    return hadamard(eigvecs)


@torch.no_grad()
def build_rotations(
    covariances: Covariances,
    dimensions: RotationDimensions,
    *,
    seed: int = 0,
    factors: Mapping[int, str | Path] | None = None,
) -> RotationPlan:
    dims = dimensions
    if dims.num_layers < 1 or len(covariances.r2) != dims.num_layers:
        raise ValueError("expected one R2 covariance per decoder layer")
    factors = {} if factors is None else factors
    h1 = _hadamard(dims.hidden_size, factors)
    h2 = _hadamard(dims.head_dim, factors)
    h4 = _hadamard(dims.intermediate_size, factors)
    r1 = _coat_rotation(covariances.r1, h1)
    r2 = [_coat_rotation(cov, h2) for cov in covariances.r2]
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def signed(h: Hadamard) -> SignedHadamard:
        signs = torch.randint(2, (h.width,), generator=gen, device="cpu", dtype=torch.int8)
        return SignedHadamard(h, signs * 2 - 1)

    r3, r4 = zip(*[(signed(h2), signed(h4)) for _ in range(dims.num_layers)])
    return RotationPlan(r1, r2, r3, r4, seed=seed).to(device=r1.device)
