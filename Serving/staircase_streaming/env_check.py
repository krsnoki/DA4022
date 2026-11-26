"""Environment check for the staircase-streaming reproduction.

Stdlib only; optional packages are probed inside try/except so the script runs
on a bare Python install. Writes ``results/env_report.json`` (relative to the
repository root) and prints a readable summary.

Usage::

    python Serving/staircase_streaming/env_check.py
"""

from __future__ import annotations

import ctypes
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "results" / "env_report.json"


def _run(cmd: list[str], timeout: float = 15.0) -> str | None:
    """Run a command and return stripped stdout, or None if it fails."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        return out.stdout.strip() or out.stderr.strip()
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------- OS
def os_info() -> dict[str, Any]:
    """OS name, version, architecture and WSL detection."""
    release = platform.release()
    is_wsl = False
    if sys.platform.startswith("linux"):
        try:
            is_wsl = "microsoft" in Path("/proc/version").read_text().lower()
        except OSError:
            pass
    return {
        "system": platform.system(),
        "release": release,
        "version": platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "is_wsl": is_wsl,
    }


# ----------------------------------------------------------------------- Python
def python_info() -> dict[str, Any]:
    """Current interpreter, other interpreters found, pip/venv/git availability."""
    others: dict[str, str | None] = {}
    if sys.platform == "win32":
        others["py_launcher"] = _run(["py", "-0p"])
    for name in ("python3", "python"):
        exe = shutil.which(name)
        if exe:
            others[name] = f"{exe}: {_run([exe, '--version'])}"
    if shutil.which("uv"):
        others["uv_pythons"] = _run(["uv", "python", "list", "--only-installed"])
    try:
        import venv  # noqa: F401
        venv_ok = True
    except ImportError:
        venv_ok = False
    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "others": others,
        "pip": _run([sys.executable, "-m", "pip", "--version"]),
        "venv_module": venv_ok,
        "git": _run(["git", "--version"]),
    }


# -------------------------------------------------------------------------- CPU
def _cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _physical_cores() -> int | None:
    if sys.platform == "win32":
        out = _run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"]
        )
        return int(out) if out and out.isdigit() else None
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.physicalcpu"])
        return int(out) if out and out.isdigit() else None
    try:
        pairs = set()
        phys = core = None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[1].strip()
                pairs.add((phys, core))
        return len(pairs) or None
    except OSError:
        return None


def _memory_bytes() -> tuple[int | None, int | None]:
    """Return (total, available) system RAM in bytes."""
    if sys.platform == "win32":

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
        return None, None
    if sys.platform == "darwin":
        total = _run(["sysctl", "-n", "hw.memsize"])
        return (int(total) if total else None), None
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) * 1024
        return info.get("MemTotal"), info.get("MemAvailable")
    except OSError:
        return None, None


def hardware_info() -> dict[str, Any]:
    """CPU, RAM and free disk on the repository drive."""
    total, avail = _memory_bytes()
    disk = shutil.disk_usage(REPO_ROOT)
    gib = 1024**3
    return {
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "physical_cores": _physical_cores(),
        "ram_total_gib": round(total / gib, 2) if total else None,
        "ram_available_gib": round(avail / gib, 2) if avail else None,
        "disk_path": str(REPO_ROOT),
        "disk_total_gib": round(disk.total / gib, 2),
        "disk_free_gib": round(disk.free / gib, 2),
    }


# -------------------------------------------------------------------------- GPU
def gpu_info() -> dict[str, Any]:
    """NVIDIA GPUs via nvidia-smi, plus Apple MPS on macOS."""
    result: dict[str, Any] = {"nvidia": [], "nvidia_note": None, "cuda_driver_version": None}
    smi = shutil.which("nvidia-smi")
    if smi is None:
        result["nvidia_note"] = "no NVIDIA GPU (nvidia-smi not found)"
    else:
        q = _run([smi, "--query-gpu=name,memory.total,memory.free,driver_version",
                  "--format=csv,noheader,nounits"])
        if q:
            for line in q.splitlines():
                name, total, free, driver = [s.strip() for s in line.split(",")]
                result["nvidia"].append({
                    "name": name, "vram_total_mib": int(total),
                    "vram_free_mib": int(free), "driver_version": driver,
                })
        header = _run([smi]) or ""
        for token in header.split("|"):
            if "CUDA Version" in token:
                result["cuda_driver_version"] = token.split("CUDA Version:")[1].strip()
        if not result["nvidia"]:
            result["nvidia_note"] = "nvidia-smi present but returned no GPUs"
    if sys.platform == "darwin":
        result["mps"] = "check torch.backends.mps.is_available() (see packages.torch)"
    return result


# --------------------------------------------------------------------- Packages
def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def packages_info() -> dict[str, Any]:
    """Installed versions of the packages this module may use."""
    pkgs: dict[str, Any] = {
        name: _version(name)
        for name in ("transformers", "vllm", "accelerate", "openai", "pytest",
                     "huggingface_hub", "datasets", "pyyaml", "matplotlib", "numpy")
    }
    torch_info: dict[str, Any] = {"version": _version("torch")}
    if torch_info["version"]:
        try:
            import torch  # type: ignore[import-not-found]

            torch_info["cuda_available"] = bool(torch.cuda.is_available())
            torch_info["cuda_build"] = torch.version.cuda
            if torch.cuda.is_available():
                torch_info["devices"] = [torch.cuda.get_device_name(i)
                                         for i in range(torch.cuda.device_count())]
            mps = getattr(torch.backends, "mps", None)
            torch_info["mps_available"] = bool(mps and mps.is_available())
        except Exception as exc:  # noqa: BLE001 - report any import failure
            torch_info["import_error"] = repr(exc)
    pkgs["torch"] = torch_info
    pkgs["ollama_cli"] = _run(["ollama", "--version"]) if shutil.which("ollama") else None
    return pkgs


# --------------------------------------------------------------------- Network
def network_info(timeout: float = 5.0) -> dict[str, Any]:
    """Quick HEAD request to huggingface.co and pypi.org."""
    out: dict[str, Any] = {}
    for url in ("https://huggingface.co", "https://pypi.org"):
        start = time.perf_counter()
        try:
            req = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "env-check/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status: Any = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code  # reachable, just a non-2xx answer
        except (urllib.error.URLError, OSError) as exc:
            status = f"unreachable: {exc}"
        out[url] = {"status": status, "seconds": round(time.perf_counter() - start, 3)}
    return out


# ------------------------------------------------------------------------ Main
def collect() -> dict[str, Any]:
    """Gather the full report."""
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "os": os_info(),
        "python": python_info(),
        "hardware": hardware_info(),
        "gpu": gpu_info(),
        "packages": packages_info(),
        "network": network_info(),
    }


def summarize(r: dict[str, Any]) -> str:
    """Human-readable summary of the report."""
    o, p, h, g, k = r["os"], r["python"], r["hardware"], r["gpu"], r["packages"]
    lines = [
        f"OS        : {o['platform']} ({o['machine']}), WSL={o['is_wsl']}",
        f"Python    : {p['version']} at {p['executable']}",
        f"pip       : {p['pip']}",
        f"venv/git  : venv={p['venv_module']}, {p['git']}",
        f"CPU       : {h['cpu_model']} ({h['physical_cores']} physical / {h['logical_cores']} logical)",
        f"RAM       : {h['ram_total_gib']} GiB total, {h['ram_available_gib']} GiB available",
        f"Disk      : {h['disk_free_gib']} / {h['disk_total_gib']} GiB free at {h['disk_path']}",
    ]
    if g["nvidia"]:
        for d in g["nvidia"]:
            lines.append(f"GPU       : {d['name']}, VRAM {d['vram_total_mib']} MiB total / "
                         f"{d['vram_free_mib']} MiB free, driver {d['driver_version']}, "
                         f"CUDA (driver) {g['cuda_driver_version']}")
    else:
        lines.append(f"GPU       : {g['nvidia_note']}")
    t = k["torch"]
    lines.append(f"torch     : {t.get('version')} cuda_available={t.get('cuda_available')}")
    for name in ("transformers", "vllm", "accelerate", "openai", "pytest", "ollama_cli"):
        lines.append(f"{name:<10}: {k.get(name)}")
    for url, v in r["network"].items():
        lines.append(f"net       : {url} -> {v['status']} ({v['seconds']} s)")
    if p["others"]:
        lines.append("other pythons:")
        for key, val in p["others"].items():
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)


def main() -> None:
    report = collect()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(summarize(report))
    print(f"\nSaved JSON report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
