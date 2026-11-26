"""Shared pytest setup: repo root on sys.path (for ``Dataloaders``) and markers."""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    """Walk upwards until the directory containing ``Dataloaders/`` is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Dataloaders").is_dir():
            return parent
    return here.parents[3]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config) -> None:  # noqa: ANN001 - pytest hook signature
    config.addinivalue_line(
        "markers", "slow: needs a running GPU server (set STAIRCASE_LIVE_CONFIG to enable)"
    )
