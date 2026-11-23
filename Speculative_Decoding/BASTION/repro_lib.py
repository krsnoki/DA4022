"""Helpers shared by the BASTION reproduction harness, analysis scripts and tests.

Nothing in here re-implements the BASTION algorithm: the tree builder, cost
model and generation loops are imported from the pinned upstream checkout in
``./upstream`` (MIT, commit 1989b80). This module only adds

* runtime registration of the local GPU in upstream's cost-model tables
  (delegated to ``tools/bastion_local.py`` so cache file names match the
  calibration run),
* a recording wrapper around ``bastion.tree_draft.build_adaptive_best_tree_from_draft_logits``
  that logs the realized tree size (and context length) of every BASTION step,
* dataset sub-sampling identical to upstream ``_limit_dataset``,
* JSONL record I/O with resume support.

It must stay importable on a CPU-only machine (tests import it).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

# Seed upstream's _run_transformers uses (random.seed(0)) right before
# _limit_dataset shuffles the dataset.
UPSTREAM_SELECTION_SEED = 0

# Environment variables read inside upstream code that alter the method.
METHOD_ENV_VARS = (
    "ADAPTIVE_MIN_TREE_SIZE",  # overrides min_tree_size inside the tree builder
    "ADAPTIVE_BUCKET_STEP",  # changes return-size bucketing of the tree builder
    "BASTION_SIGMA_SCALE",  # calibration fit weighting (cache validity check)
    "CALIBRATION_JSON_PATH",  # set by the harness itself
)

MODEL_KEYS = {
    "qwen3-4b": ("Qwen/Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16"),
    "llama3.1-8b": ("meta-llama/Llama-3.1-8B-Instruct", "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"),
}

PAPER_SHORT_DATASETS = ["gsm8k", "math500", "aime25", "humaneval", "mbpp", "lcb", "mt-bench", "alpaca"]
MODES = ("ar", "dflash", "bastion")


# ---------------------------------------------------------------------------
# GPU registration
# ---------------------------------------------------------------------------

# Fallback copy of the values in tools/bastion_local.py (used only if that file
# is missing). Keep in sync; metadata records which source was used.
_FALLBACK_GPU_KEY = "b4500"
_FALLBACK_GPU_SPEC = {
    "display_name": "NVIDIA RTX PRO 4500 Blackwell",
    "peak_throughput_bf16": 202 * 1e12,
    "memory_bandwidth": 896 * 1e9,
}
_FALLBACK_GPU_ALIASES = ("b4500", "rtxpro4500", "nvidiartxpro4500", "nvidiartxpro4500blackwell")


def register_local_gpu_specs() -> dict:
    """Register GPUs missing from upstream's tables (runtime only, no file edits).

    Prefers ``tools/bastion_local.py`` (the launcher used to create the
    calibration caches) so that the GPU key, and therefore the cache file
    names, are identical. Returns a description for the run metadata.
    """
    from bastion import cost_model

    launcher = ROOT / "tools" / "bastion_local.py"
    source = None
    if launcher.exists():
        spec = importlib.util.spec_from_file_location("_bastion_local_launcher", launcher)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)  # registers into cost_model tables; does not run main()
        source = str(launcher.relative_to(ROOT))
    else:
        cost_model.GPU_SPECS.setdefault(_FALLBACK_GPU_KEY, dict(_FALLBACK_GPU_SPEC))
        for alias in _FALLBACK_GPU_ALIASES:
            cost_model.GPU_ALIASES.setdefault(alias, _FALLBACK_GPU_KEY)
        source = "repro_lib fallback"
    added = {k: v for k, v in cost_model.GPU_SPECS.items() if k not in _UPSTREAM_GPU_KEYS}
    return {"source": source, "added_gpu_specs": added}


_UPSTREAM_GPU_KEYS = {"a5000", "a6000", "a100", "h100", "h200", "b6000"}


# ---------------------------------------------------------------------------
# Dataset selection (identical to upstream _limit_dataset)
# ---------------------------------------------------------------------------

def upstream_selection_indices(n: int, max_samples: int | None, seed: int = UPSTREAM_SELECTION_SEED) -> list[int]:
    """Original-dataset indices that upstream ``_limit_dataset`` keeps, in order.

    Upstream does ``random.seed(0)`` in ``_run_transformers`` and later
    ``random.shuffle(dataset); dataset[:max_samples]`` (no shuffle at all when
    ``len(dataset) <= max_samples``). ``random.Random(seed).shuffle`` yields the
    same permutation as seeding the global generator, without touching global
    state. Equivalence assumes nothing between upstream's ``random.seed(0)``
    and the shuffle consumes Python ``random`` draws (model/tokenizer loading;
    see tests/test_selection.py).
    """
    indices = list(range(n))
    if max_samples is None or n <= max_samples:
        return indices
    random.Random(seed).shuffle(indices)
    return indices[:max_samples]


def parse_caps(cap_args: Iterable[str] | None) -> dict[str, int]:
    caps: dict[str, int] = {}
    for item in cap_args or []:
        if "=" not in item:
            raise ValueError(f"--cap expects DATASET=N, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        caps[name] = int(value)
        if caps[name] <= 0:
            raise ValueError(f"cap for {name} must be positive")
    return caps


def parse_budgets(values: Iterable[str]) -> list[int | None]:
    """'adaptive' -> None (upstream adaptive controller); integers -> fixed N."""
    out: list[int | None] = []
    for raw in values:
        for item in str(raw).split(","):
            item = item.strip().lower()
            if not item:
                continue
            if item in {"adaptive", "none", "auto"}:
                budget = None
            else:
                budget = int(item)
                if budget < 2:
                    raise ValueError("fixed budget must be >= 2 (N=1 is a root-only tree)")
            if budget not in out:
                out.append(budget)
    return out


# ---------------------------------------------------------------------------
# Tree-size recording hook
# ---------------------------------------------------------------------------

class TreeSizeRecorder:
    """Wraps the upstream tree builder to log realized tree sizes.

    ``bastion.tree_draft.bastion_generate`` resolves
    ``build_adaptive_best_tree_from_draft_logits`` as a module global at call
    time, so replacing that attribute is enough. The wrapper calls the
    original function with *unchanged* arguments and only reads
    ``tree.token_ids.shape[0]`` (a host-side shape, no device sync).

    Fixed budgets are NOT injected here; they are passed through
    ``bastion_generate(min_tree_size=N, max_tree_size=N)``, which is part of
    upstream's public signature. ``expected_size`` is only used to count steps
    whose realized size differs from the requested N.
    """

    def __init__(self) -> None:
        self.active = False
        self.sizes: list[int] = []
        self.contexts: list[int] = []
        self.requested_min: list[int] = []
        self.requested_max: list[int] = []
        self._orig = None

    def install(self) -> "TreeSizeRecorder":
        from bastion import tree_draft

        if self._orig is not None:
            return self
        orig = tree_draft.build_adaptive_best_tree_from_draft_logits
        if getattr(orig, "_repro_wrapped", False):
            raise RuntimeError("tree builder already wrapped")
        self._orig = orig
        recorder = self

        def wrapped(*args, **kwargs):
            tree = orig(*args, **kwargs)
            if recorder.active:
                recorder.sizes.append(int(tree.token_ids.shape[0]))
                recorder.contexts.append(int(kwargs.get("context_prefix_length", -1)))
                recorder.requested_min.append(int(kwargs.get("min_tree_size", -1)))
                recorder.requested_max.append(int(kwargs.get("max_tree_size", -1)))
            return tree

        wrapped._repro_wrapped = True  # type: ignore[attr-defined]
        wrapped.__wrapped__ = orig  # type: ignore[attr-defined]
        tree_draft.build_adaptive_best_tree_from_draft_logits = wrapped
        return self

    def uninstall(self) -> None:
        from bastion import tree_draft

        if self._orig is not None:
            tree_draft.build_adaptive_best_tree_from_draft_logits = self._orig
            self._orig = None

    @contextmanager
    def recording(self):
        self.sizes, self.contexts, self.requested_min, self.requested_max = [], [], [], []
        self.active = True
        try:
            yield self
        finally:
            self.active = False


def check_method_env(allow_overrides: bool = False) -> dict[str, str | None]:
    """Return method-relevant env vars; refuse ones that silently alter BASTION."""
    values = {name: os.environ.get(name) for name in METHOD_ENV_VARS}
    bad = [n for n in ("ADAPTIVE_MIN_TREE_SIZE", "ADAPTIVE_BUCKET_STEP") if values.get(n)]
    if bad and not allow_overrides:
        raise RuntimeError(
            f"Environment variables {bad} are set. They change upstream's tree builder "
            "(ADAPTIVE_MIN_TREE_SIZE would also break fixed budgets). Unset them or pass "
            "--allow-env-overrides."
        )
    return values


# ---------------------------------------------------------------------------
# JSONL records
# ---------------------------------------------------------------------------

PROFILE_MODES = ("upstream", "warm")
# Records written before --profile-mode existed used upstream's profile cache.
LEGACY_PROFILE_MODE = "upstream"


def profile_variant(mode: str, budget, profile_mode: str | None) -> str | None:
    """Profile mode that distinguishes configurations, or None where it does not matter.

    Only adaptive BASTION uses the profiled draft / non-verify latencies (in the
    controller's stopping rule), so only (bastion, fixed_budget=None) records are
    split by profile mode. Fixed-budget trees have exactly N nodes regardless of
    the constants; AR / DFlash never read them.
    """
    if mode == "bastion" and budget is None:
        return profile_mode or LEGACY_PROFILE_MODE
    return None


def rec_profile_variant(rec: dict) -> str | None:
    return profile_variant(rec["mode"], rec.get("fixed_budget"), rec.get("profile_mode"))


def make_key(dataset: str, sample_idx: int, turn: int, mode: str, budget, profile_mode: str | None = None) -> tuple:
    return (
        dataset,
        int(sample_idx),
        int(turn),
        mode,
        None if budget is None else int(budget),
        profile_variant(mode, budget, profile_mode),
    )


def record_key(rec: dict) -> tuple:
    """Unique key of one generation inside a run file.

    (dataset, sample_idx, turn, mode, fixed_budget, profile variant); the last
    element is "warm"/"upstream" for adaptive BASTION and None otherwise.
    """
    return make_key(rec["dataset"], rec["sample_idx"], rec["turn"], rec["mode"], rec.get("fixed_budget"),
                    rec.get("profile_mode"))


def load_jsonl(path: str | Path) -> list[dict]:
    """Load records; a truncated final line (crash mid-write) is skipped."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"[warn] skipping truncated last line of {path}", file=sys.stderr)
                continue
            raise
    return out


class JsonlWriter:
    """Append-only writer, one flushed + fsynced line per record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_jsonl(self.path)
        self.records: dict[tuple, dict] = {}
        for rec in existing:
            self.records[record_key(rec)] = rec
        # Repair a truncated tail so subsequent appends start on a fresh line.
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as fh:
                fh.seek(-1, os.SEEK_END)
                last = fh.read(1)
            if last != b"\n":
                good = [json.dumps(r, ensure_ascii=False) for r in existing]
                tmp = self.path.with_suffix(self.path.suffix + ".repair")
                tmp.write_text("".join(g + "\n" for g in good), encoding="utf-8")
                os.replace(tmp, self.path)
        self._fh = self.path.open("a", encoding="utf-8")

    def has(self, key: tuple) -> bool:
        return key in self.records

    def get(self, key: tuple) -> dict | None:
        return self.records.get(key)

    def write(self, rec: dict) -> None:
        key = record_key(rec)
        if key in self.records:
            raise KeyError(f"duplicate record key {key}")
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.records[key] = rec

    def close(self) -> None:
        self._fh.close()


def input_hash(token_ids: Iterable[int]) -> str:
    h = hashlib.sha1()
    h.update(json.dumps([int(t) for t in token_ids]).encode())
    return h.hexdigest()[:16]


def sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def round_list(values: Iterable[float] | None, ndigits: int = 7) -> list[float] | None:
    if values is None:
        return None
    return [round(float(v), ndigits) for v in values]


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    return repr(obj)
