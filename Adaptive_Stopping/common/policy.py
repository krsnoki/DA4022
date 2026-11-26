"""Shared stopping-policy contract (see ``Adaptive_Stopping/README_adaptiveStopping.md``)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class StoppingPolicy(ABC):
    """Decides whether a variable-compute loop should continue.

    Implementations must document which entries of ``state`` and
    ``metrics`` they read and whether those are available without an
    extra forward pass.
    """

    @abstractmethod
    def reset(self, request_metadata: Optional[Dict[str, Any]] = None) -> None: ...

    @abstractmethod
    def should_stop(self, state: Dict[str, Any], metrics: Dict[str, Any], budget: Any) -> bool: ...

    @abstractmethod
    def report(self) -> Dict[str, Any]: ...


class NeverStop(StoppingPolicy):
    """No-op policy: run until the hard budget is exhausted (full-compute baseline)."""

    def reset(self, request_metadata=None):
        self._calls = 0

    def should_stop(self, state, metrics, budget):
        self._calls += 1
        return False

    def report(self):
        return {"policy": "never_stop", "calls": getattr(self, "_calls", 0)}


__all__ = ["StoppingPolicy", "NeverStop"]
