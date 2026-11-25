"""Optional NVML sampler: power/clock/temperature + trapezoidal energy.

Energy from NVML is an *estimate* (imprecise, see paper). Label results
accordingly and cross-check with wall-socket meters / Nsight when possible.
"""
from __future__ import annotations

import threading
import time


class NvmlSampler:
    def __init__(self, device_index: int = 0, interval_s: float = 0.05, enabled: bool = True):
        self.interval_s = interval_s
        self.samples: list[tuple[float, float, int, int]] = []  # t, W, SM MHz, tempC
        self._stop = threading.Event()
        self._thread = None
        self.available = False
        if not enabled:
            self._nv = None
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.available = True
        except Exception:
            self._nv = None

    def _run(self):
        nv, h = self._nv, self._h
        while not self._stop.is_set():
            try:
                p = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
                c = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
                t = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
                self.samples.append((time.perf_counter(), p, c, t))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __enter__(self):
        if self.available:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread:
            self._stop.set()
            self._thread.join()

    def summary(self) -> dict:
        s = self.samples
        if len(s) < 2:
            return {"nvml_available": self.available, "energy_j": None}
        energy = sum((s[i][0] - s[i - 1][0]) * (s[i][1] + s[i - 1][1]) / 2 for i in range(1, len(s)))
        n = len(s)
        mean = lambda i: sum(x[i] for x in s) / n  # noqa: E731
        var = lambda i: sum((x[i] - mean(i)) ** 2 for x in s) / n  # noqa: E731
        return {
            "nvml_available": True,
            "energy_j": energy,
            "avg_power_w": mean(1),
            "avg_sm_clock_mhz": mean(2),
            "sm_clock_std_mhz": var(2) ** 0.5,
            "avg_temp_c": mean(3),
            "temp_std_c": var(3) ** 0.5,
            "n_samples": n,
        }