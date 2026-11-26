"""Start / stop one ``vllm serve`` process per model listed under ``servers:``.

Usage (Linux GPU machine, repo root)::

    python -m Serving.staircase_streaming.launch_vllm --config <cfg.yaml> start
    python -m Serving.staircase_streaming.launch_vllm --config <cfg.yaml> status
    python -m Serving.staircase_streaming.launch_vllm --config <cfg.yaml> stop

Servers start one after another (vLLM sizes its KV-cache pool from the GPU memory
that is free at start-up, so concurrent start-ups on one GPU can collide).
Arguments listed in ``vllm.optional_args`` are dropped automatically if the
installed vLLM rejects them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import load_config


def _log_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["vllm"].get("log_dir") or Path(cfg["experiment"]["output_dir"]) / "vllm_logs")


def build_command(name: str, srv: dict[str, Any], cfg: dict[str, Any], optional: bool = True) -> list[str]:
    """The ``vllm serve`` command line for one server."""
    v = cfg["vllm"]
    host = cfg["backend"].get("host", "127.0.0.1")
    cmd = [
        "vllm", "serve", srv["hf_model"],
        "--revision", str(srv["revision"]),
        "--served-model-name", name,
        "--host", host,
        "--port", str(srv["port"]),
        "--gpu-memory-utilization", str(srv["gpu_memory_utilization"]),
        "--max-model-len", str(srv["max_model_len"]),
        "--dtype", str(srv.get("dtype", "auto")),
        "--seed", str(cfg["experiment"].get("seed", 0)),
        "--enable-prefix-caching" if v.get("enable_prefix_caching", True) else "--no-enable-prefix-caching",
    ]
    if optional:
        cmd += [str(a) for a in v.get("optional_args") or []]
    cmd += [str(a) for a in v.get("extra_args") or []]
    cmd += [str(a) for a in srv.get("extra_args") or []]
    return cmd


def _healthy(host: str, port: int, name: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=5) as r:
            ids = [m["id"] for m in json.loads(r.read().decode()).get("data", [])]
        return name in ids
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _start_one(name: str, srv: dict[str, Any], cfg: dict[str, Any], log_dir: Path) -> int:
    host = cfg["backend"].get("host", "127.0.0.1")
    timeout = float(cfg["vllm"].get("startup_timeout_s", 1200))
    env = dict(os.environ)
    if srv.get("gpu") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(srv["gpu"])
    for optional in (True, False):
        cmd = build_command(name, srv, cfg, optional=optional)
        log = log_dir / f"{name}.log"
        print(f"[launch] {name}: {' '.join(cmd)}", flush=True)
        with open(log, "w") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if _healthy(host, int(srv["port"]), name):
                print(f"[launch] {name} ready on port {srv['port']} (pid {proc.pid})", flush=True)
                return proc.pid
            if proc.poll() is not None:
                break
            time.sleep(3)
        else:
            proc.terminate()
            raise TimeoutError(f"{name} not healthy after {timeout}s; log tail:\n{_tail(log)}")
        text = _tail(log, 200)
        rejected = "unrecognized arguments" in text or "error: argument" in text
        if optional and rejected and cfg["vllm"].get("optional_args"):
            print(f"[launch] {name}: vLLM rejected an optional arg; retrying without "
                  f"{cfg['vllm']['optional_args']}", flush=True)
            continue
        raise RuntimeError(f"{name} exited with code {proc.returncode}; log tail:\n{_tail(log)}")
    raise RuntimeError("unreachable")


def start(cfg: dict[str, Any]) -> dict[str, int]:
    if shutil.which("vllm") is None:
        sys.exit("vllm executable not found: pip install -r Serving/staircase_streaming/requirements-gpu.txt")
    log_dir = _log_dir(cfg)
    log_dir.mkdir(parents=True, exist_ok=True)
    pids: dict[str, int] = {}
    pid_file = log_dir / "pids.json"
    try:
        for name, srv in cfg["servers"].items():
            pids[name] = _start_one(name, srv, cfg, log_dir)
            pid_file.write_text(json.dumps(pids, indent=2))
    except BaseException:
        stop(cfg)
        raise
    return pids


def stop(cfg: dict[str, Any]) -> None:
    pid_file = _log_dir(cfg) / "pids.json"
    if not pid_file.exists():
        print("[launch] no pids.json; nothing to stop")
        return
    pids = json.loads(pid_file.read_text())
    for name, pid in pids.items():
        try:
            os.killpg(pid, signal.SIGTERM)
            print(f"[launch] sent SIGTERM to {name} (pid group {pid})")
        except (ProcessLookupError, PermissionError, AttributeError):
            pass
    time.sleep(5)
    for pid in pids.values():
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError):
            pass
    pid_file.unlink(missing_ok=True)


def status(cfg: dict[str, Any]) -> bool:
    host = cfg["backend"].get("host", "127.0.0.1")
    ok = True
    for name, srv in cfg["servers"].items():
        up = _healthy(host, int(srv["port"]), name)
        ok &= up
        print(f"{name:<28} port {srv['port']}: {'UP' if up else 'DOWN'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("action", choices=["start", "stop", "status"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.action == "start":
        start(cfg)
    elif args.action == "stop":
        stop(cfg)
    else:
        sys.exit(0 if status(cfg) else 1)


if __name__ == "__main__":
    main()
