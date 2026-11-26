"""Load a YAML experiment config and build backends / MoA / staircase objects from it."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .backends import Backend, MockBackend, MockModelSpec, OpenAICompatBackend
from .moa_baseline import MoAConfig
from .proposers import AgentSpec
from .staircase import StaircaseConfig


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read YAML and deep-merge optional ``overrides`` (used by tests and ``--quick``)."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def endpoints(cfg: dict[str, Any]) -> dict[str, str]:
    """Logical model id -> OpenAI base URL, from the ``servers`` section."""
    host = cfg["backend"].get("host", "127.0.0.1")
    explicit = cfg["backend"].get("endpoints") or {}
    eps = {name: f"http://{host}:{srv['port']}/v1" for name, srv in (cfg.get("servers") or {}).items()}
    eps.update(explicit)
    return eps


def build_backend(cfg: dict[str, Any]) -> Backend:
    kind = cfg["backend"]["type"]
    if kind == "mock":
        specs = {name: MockModelSpec(**spec) for name, spec in cfg["mock_models"].items()}
        return MockBackend(specs)
    if kind == "openai_compat":
        return OpenAICompatBackend(endpoints(cfg), timeout_s=cfg["backend"].get("request_timeout_s", 600))
    raise ValueError(f"unknown backend type {kind!r}")


def build_moa(cfg: dict[str, Any], seed: int | None = None) -> MoAConfig:
    m = cfg["moa"]
    return MoAConfig(
        proposers=[AgentSpec(p, m["proposer_max_tokens"], dict(m.get("proposer_gen") or {}))
                   for p in m["proposers"]],
        aggregator=AgentSpec(m["aggregator"], m["aggregator_max_tokens"],
                             dict(m.get("aggregator_gen") or {})),
        proposer_system=m.get("proposer_system"),
        timeout_s=cfg["backend"].get("request_timeout_s", 600),
        seed=seed,
    )


def build_staircase(cfg: dict[str, Any], **overrides: Any) -> StaircaseConfig:
    return StaircaseConfig(**{**cfg["staircase"], **overrides})


def model_metadata(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per logical model: HF id, revision, backbone registry id (empty for mock)."""
    return {name: {"hf_model": s.get("hf_model"), "revision": s.get("revision"),
                   "backbone_id": s.get("backbone_id")}
            for name, s in (cfg.get("servers") or {}).items()}
