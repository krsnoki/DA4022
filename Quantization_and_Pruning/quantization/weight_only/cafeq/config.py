"""Typed, validated configuration for CafeQ experiments.

Configs are YAML files under ``Quantization_and_Pruning/configs/cafeq/``. Loading is strict:
unknown keys raise, and ``calibration_dataset`` must be present and ``null`` because CafeQ is a
calibration-free method (see ``Dataloaders/README_dataloader.md``).
"""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

METHODS = ("none", "rtn", "random_hadamard", "cafeq")
BLOCK_LOSSES = ("paper_eq4", "grouped_bound")
BLOCK_INITS = ("rotation", "hadamard", "identity")
PAIRED_LOSSES = ("logsumexp", "sum_sq", "sum_sq_weighted")
PAIRED_INITS = ("identity", "rotation")
OPTIMIZERS = ("adam", "sgd")
DTYPES = ("float32", "float16", "bfloat16")


class ConfigError(ValueError):
    """Raised for malformed or inconsistent configs."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


@dataclass
class BackboneConfig:
    id: str = "text.qwen2_5_0_5b.base"
    hf_name: str = "Qwen/Qwen2.5-0.5B"
    revision: Optional[str] = None  # pin an exact commit before reporting results
    dtype: str = "float32"

    def validate(self) -> None:
        _require(self.dtype in DTYPES, f"backbone.dtype must be one of {DTYPES}")


@dataclass
class EvaluationConfig:
    dataset_id: str = "text.wikitext2.raw.test"
    split: str = "test"
    seq_len: int = 2048
    max_chunks: Optional[int] = None  # None = whole split
    batch_size: int = 1

    def validate(self) -> None:
        _require(self.seq_len > 0, "evaluation.seq_len must be positive")
        _require(self.max_chunks is None or self.max_chunks > 0, "evaluation.max_chunks must be positive or null")
        _require(self.batch_size > 0, "evaluation.batch_size must be positive")


@dataclass
class TargetsConfig:
    """Which weight sets to quantize (mirrors the FF / QK / VO ablation in Table 3)."""

    ff: bool = True  # gate_proj, up_proj, down_proj
    qk: bool = True  # q_proj, k_proj
    vo: bool = True  # v_proj, o_proj


@dataclass
class QuantizationConfig:
    method: str = "cafeq"
    bits: int = 4
    group_size: Optional[int] = None  # None = per-channel along the contraction dimension
    targets: TargetsConfig = field(default_factory=TargetsConfig)

    def validate(self) -> None:
        _require(self.method in METHODS, f"quantization.method must be one of {METHODS}")
        _require(isinstance(self.bits, int) and 2 <= self.bits <= 8, "quantization.bits must be an int in [2, 8]")
        _require(self.group_size is None or self.group_size > 0, "quantization.group_size must be positive or null")


@dataclass
class BlockTransformConfig:
    """Section 3.1: learned block-diagonal transform for single matrices."""

    block_size: int = 128
    iterations: int = 2000  # paper sweeps used 20,000 (Appendix E)
    lr: float = 1e-3
    optimizer: str = "adam"  # not specified in the paper
    loss: str = "paper_eq4"
    init: str = "rotation"  # paper: each block a random rotation
    keep_best: bool = True
    fallback_if_worse: bool = True
    log_every: int = 0

    def validate(self) -> None:
        _require(self.block_size > 0, "block_transform.block_size must be positive")
        _require(self.iterations >= 0, "block_transform.iterations must be >= 0")
        _require(self.lr > 0, "block_transform.lr must be positive")
        _require(self.optimizer in OPTIMIZERS, f"block_transform.optimizer must be one of {OPTIMIZERS}")
        _require(self.loss in BLOCK_LOSSES, f"block_transform.loss must be one of {BLOCK_LOSSES}")
        _require(self.init in BLOCK_INITS, f"block_transform.init must be one of {BLOCK_INITS}")


@dataclass
class PairedTransformConfig:
    """Appendix A / F: learned paired transform for (W_v, W_o)."""

    enabled: bool = True
    iterations: int = 5000  # paper sweeps used 100,000 (Appendix F.2)
    lr: float = 1e-3
    optimizer: str = "adam"
    pseudo_loss: str = "logsumexp"
    temperature: float = 5.0
    orth_weight: float = 0.0
    normalize: bool = True  # our choice: make the temperature scale-free (see paper card)
    init: str = "identity"
    keep_best: bool = True
    log_every: int = 0

    def validate(self) -> None:
        _require(self.iterations >= 0, "paired_transform.iterations must be >= 0")
        _require(self.lr > 0, "paired_transform.lr must be positive")
        _require(self.optimizer in OPTIMIZERS, f"paired_transform.optimizer must be one of {OPTIMIZERS}")
        _require(self.pseudo_loss in PAIRED_LOSSES, f"paired_transform.pseudo_loss must be one of {PAIRED_LOSSES}")
        _require(self.temperature > 0, "paired_transform.temperature must be positive")
        _require(self.orth_weight >= 0, "paired_transform.orth_weight must be >= 0")
        _require(self.init in PAIRED_INITS, f"paired_transform.init must be one of {PAIRED_INITS}")


@dataclass
class AdaptiveRoundingConfig:
    """Appendix B, Algorithm 1."""

    enabled: bool = True
    iterations: int = 5
    keep_best: bool = True  # our addition: Algorithm 1 is not monotone in practice

    def validate(self) -> None:
        _require(self.iterations >= 0, "adaptive_rounding.iterations must be >= 0")


@dataclass
class RandomBaselineConfig:
    """Random block-Hadamard baseline (``random`` / ``Ran*`` in Tables 1-4)."""

    block_size: int = 1024  # capped to the largest power of two dividing each dimension

    def validate(self) -> None:
        _require(self.block_size > 0, "random_baseline.block_size must be positive")


@dataclass
class CafeQConfig:
    experiment_id: str = "cafeq_debug"
    paper_id: str = "adaptfm.cafeq"
    seed: int = 0
    device: str = "cpu"
    output_dir: str = "outputs/cafeq"
    calibration_dataset: Optional[str] = None
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    block_transform: BlockTransformConfig = field(default_factory=BlockTransformConfig)
    paired_transform: PairedTransformConfig = field(default_factory=PairedTransformConfig)
    adaptive_rounding: AdaptiveRoundingConfig = field(default_factory=AdaptiveRoundingConfig)
    random_baseline: RandomBaselineConfig = field(default_factory=RandomBaselineConfig)

    def validate(self) -> "CafeQConfig":
        _require(self.calibration_dataset is None, "CafeQ is calibration-free: calibration_dataset must be null")
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if hasattr(v, "validate"):
                v.validate()
        return self

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CafeQConfig":
        _require(isinstance(data, dict), "config root must be a mapping")
        _require(
            "calibration_dataset" in data,
            "config must set `calibration_dataset: null` explicitly (calibration-free method)",
        )
        return _build(cls, data, "").validate()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "CafeQConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, data: Dict[str, Any], prefix: str):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    _require(not unknown, f"unknown config keys at '{prefix or '<root>'}': {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        tp = hints[name]
        if dataclasses.is_dataclass(tp):
            _require(isinstance(value, dict), f"'{prefix}{name}' must be a mapping")
            kwargs[name] = _build(tp, value, f"{prefix}{name}.")
        else:
            kwargs[name] = value
    return cls(**kwargs)
