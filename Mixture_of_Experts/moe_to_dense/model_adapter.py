"""Adapter boundary between model-specific layouts and core conversion code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from torch import nn

from .merging import ExpertWeights


@dataclass(frozen=True)
class MoELayer:
    name: str
    module: nn.Module
    experts: list[ExpertWeights]


class MoEModelAdapter(ABC):
    """Backbones implement only extraction and replacement; algorithms stay generic."""

    @abstractmethod
    def iter_moe_layers(self, model: nn.Module) -> Iterable[MoELayer]:
        raise NotImplementedError

    @abstractmethod
    def replace_with_dense(self, model: nn.Module, layer: MoELayer, dense: nn.Module) -> None:
        raise NotImplementedError

    def assert_dense(self, model: nn.Module) -> None:
        remaining = list(self.iter_moe_layers(model))
        if remaining:
            raise AssertionError(f"{len(remaining)} routed MoE layers remain")


class TinyMoEAdapter(MoEModelAdapter):
    """Executable reference adapter used by CPU tests and the local reproduction."""

    def iter_moe_layers(self, model: nn.Module) -> Iterable[MoELayer]:
        from .synthetic import TinyMoELayer

        for name, module in model.named_modules():
            if isinstance(module, TinyMoELayer):
                yield MoELayer(name, module, [expert.export_weights() for expert in module.experts])

    def replace_with_dense(self, model: nn.Module, layer: MoELayer, dense: nn.Module) -> None:
        parent = model
        parts = layer.name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], dense)
