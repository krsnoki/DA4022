"""Shared helpers for the Jacobian-noise experiment scripts.

Nothing method-specific lives here: config loading, seeding, environment
metadata, the results CSV (schema of Performance_Metrics/README_metrics.md),
latency/memory measurement and a small generic training loop.
"""

from __future__ import annotations

import copy
import csv
import datetime as _dt
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402

PAPER_ID = "openreview:wxKn35RAhW"  # arXiv:2608.20988
RESULT_COLUMNS = [
    "experiment_id", "paper_id", "method", "commit", "backbone_id", "backbone_revision", "dataset_id", "split",
    "seed", "device", "framework_version", "dtype", "batch_size", "input_length", "output_length",
    "quality_metric", "quality_value", "latency_median_ms", "latency_p95_ms", "throughput", "peak_memory_mb",
    "energy_j", "status", "notes",
]


# ------------------------------------------------------------------ config
def _set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str, overrides: Optional[List[str]] = None) -> dict:
    """Load YAML (supports ``extends: other.yaml``) and apply ``a.b=value`` overrides."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "extends" in cfg:
        parent = os.path.join(os.path.dirname(path), cfg.pop("extends"))
        cfg = _merge(load_config(parent), cfg)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


# ------------------------------------------------------------- environment
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return f"{p.name} ({p.total_memory / 2**30:.0f} GB)"
    return f"cpu ({platform.processor() or platform.machine()})"


def env_metadata(device: torch.device) -> dict:
    import transformers

    meta = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": device_name(device),
        "commit": git_commit(),
        "platform": platform.platform(),
        "time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        import datasets

        meta["datasets"] = datasets.__version__
    except ImportError:
        pass
    return meta


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def resolve_precision(name: str, device: torch.device):
    """Return (autocast dtype or None, needs GradScaler). bf16 falls back to fp16 on GPUs without bf16 (T4/P100)."""
    if device.type != "cuda" or name == "fp32":
        return None, False
    if name == "bf16" and not torch.cuda.is_bf16_supported():
        print("[precision] bf16 not supported on this GPU, using fp16 + GradScaler")
        name = "fp16"
    return (torch.bfloat16, False) if name == "bf16" else (torch.float16, True)


# ----------------------------------------------------------------- results
def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def append_jsonl(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def append_result_rows(path: str, rows: List[dict]) -> None:
    """Append rows to a CSV with the repository result schema (unknown -> empty)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in RESULT_COLUMNS})


# -------------------------------------------------------------- measurement
def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_latency(fn: Callable[[], Any], device: torch.device, warmup: int = 3, iters: int = 10) -> dict:
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p95 = times[min(len(times) - 1, int(math.ceil(0.95 * len(times))) - 1)]
    return {"latency_median_ms": times[len(times) // 2], "latency_p95_ms": p95, "latency_iters": iters, "latency_warmup": warmup}


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return None  # not measured on CPU


# ------------------------------------------------------------------ training
def cosine_with_warmup(optimizer, warmup: int, total: int):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_loop(
    model: torch.nn.Module,
    params: Iterable[torch.nn.Parameter],
    micro_batches: Iterator[Any],
    loss_fn: Callable[[Any], torch.Tensor],
    tcfg: dict,
    device: torch.device,
    ctrl=None,
    on_micro_batch: Optional[Callable[[Any], None]] = None,
    evaluate: Optional[Callable[[], dict]] = None,
    log_path: Optional[str] = None,
) -> dict:
    """Generic loop: AdamW + cosine warmup + grad accumulation + clipping + AMP.

    ``micro_batches`` must yield at least ``max_steps * grad_accum`` batches.
    ``ctrl.step()`` is called after each optimiser step (noise schedule).
    """
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.0)))
    max_steps, accum = int(tcfg["max_steps"]), int(tcfg.get("grad_accum", 1))
    sched = cosine_with_warmup(opt, int(tcfg.get("warmup_steps", 0)), max_steps)
    amp_dtype, use_scaler = resolve_precision(tcfg.get("precision", "bf16"), device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    clip = tcfg.get("grad_clip", 1.0)
    log_every, eval_every = int(tcfg.get("log_every", 10)), int(tcfg.get("eval_every", 0) or 0)

    model.train()
    reset_peak_memory(device)
    t_start = time.perf_counter()
    running, n_running, history = 0.0, 0, []
    for step in range(max_steps):
        for _ in range(accum):
            batch = next(micro_batches)
            if on_micro_batch:
                on_micro_batch(batch)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = loss_fn(batch)
            scaler.scale(loss / accum).backward()
            running += float(loss.detach())
            n_running += 1
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, clip) if clip else torch.tensor(0.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if ctrl is not None:
            ctrl.step()
        if (step + 1) % log_every == 0 or step == max_steps - 1:
            rec = {
                "step": step + 1, "loss": running / max(1, n_running), "grad_norm": float(gnorm),
                "lr": sched.get_last_lr()[0], "elapsed_s": round(time.perf_counter() - t_start, 1),
            }
            if ctrl is not None:
                s = ctrl.summary()
                rec.update({k: s[k] for k in ("sigma_mean", "sigma_global", "in_warmup") if k in s})
            print(json.dumps(rec))
            if log_path:
                append_jsonl(log_path, rec)
            history.append(rec)
            running, n_running = 0.0, 0
        if evaluate and eval_every and (step + 1) % eval_every == 0:
            model.eval()
            ev = {"step": step + 1, **evaluate()}
            model.train()
            print(json.dumps(ev))
            if log_path:
                append_jsonl(log_path, ev)
    return {
        "train_time_s": time.perf_counter() - t_start,
        "train_peak_memory_mb": peak_memory_mb(device),
        "history": history,
    }


def cycle(make_iter: Callable[[int], Iterable[Any]]) -> Iterator[Any]:
    """Repeat ``make_iter(epoch)`` forever (each epoch gets a fresh iterator)."""
    epoch = 0
    while True:
        n = 0
        for x in make_iter(epoch):
            n += 1
            yield x
        if n == 0:
            raise RuntimeError("empty data iterator")
        epoch += 1
