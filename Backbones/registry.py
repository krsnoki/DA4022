"""Read entries from ``Backbones/registry.yaml``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).with_name("registry.yaml")


@lru_cache(maxsize=None)
def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of backbone IDs to entries")
    for key, entry in data.items():
        if entry.get("id") != key:
            raise ValueError(f"{path}: entry {key!r} has id {entry.get('id')!r}")
    return data


def get_backbone_entry(backbone_id: str, path: str | Path = REGISTRY_PATH) -> dict:
    """Return a copy of the registry entry for ``backbone_id``."""
    data = _load(str(path))
    if backbone_id not in data:
        raise KeyError(f"unknown backbone id {backbone_id!r}; known: {sorted(data)}")
    return dict(data[backbone_id])
