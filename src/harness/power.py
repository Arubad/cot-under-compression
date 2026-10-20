"""GPU power polling (NVML, 10 Hz) and the tokens/s cost-model fallback.

    with PowerMeter(poll_hz=10) as pm:
        ...decode...
    pm.energy_j, pm.mean_w, pm.samples

NVML samples power coarsely and with lag (Part-time Power Measurements,
arXiv 2312.02741 reports up to 70% naive energy error on A100/H100), so the
protocol is: medians over ``repeats`` runs, spread reported, and a fallback
energy = W_active * wall_seconds when NVML is unavailable (Colab sometimes hides it).
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import numpy as np


class PowerMeter:
    def __init__(self, poll_hz: float = 10.0, device_index: int = 0, fallback_w_active: float = 70.0):
        self.poll_hz = poll_hz
        self.device_index = device_index
        self.fallback_w = fallback_w_active
        self.samples: List[float] = []
        self.times: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.available = False
        self.t_start = 0.0
        self.t_end = 0.0
        self._handle = None

    def _init_nvml(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            pynvml.nvmlDeviceGetPowerUsage(self._handle)
            self._pynvml = pynvml
            return True
        except Exception:
            return False

    def _loop(self):
        period = 1.0 / self.poll_hz
        while not self._stop.is_set():
            try:
                w = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                self.samples.append(w)
                self.times.append(time.time())
            except Exception:
                pass
            self._stop.wait(period)

    def __enter__(self):
        self.available = self._init_nvml()
        self.t_start = time.time()
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self.t_end = time.time()
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=2.0)
        return False

    @property
    def wall_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def mean_w(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")

    @property
    def energy_j(self) -> float:
        """Trapezoidal integral of the power trace; fallback W_active * wall time."""
        if len(self.samples) >= 2:
            return float(np.trapz(self.samples, self.times))
        return self.fallback_w * self.wall_s

    def summary(self) -> Dict[str, float]:
        return {"energy_j": self.energy_j, "wall_s": self.wall_s, "mean_w": self.mean_w, "n_samples": len(self.samples), "nvml_available": self.available, "energy_source": "nvml" if len(self.samples) >= 2 else "fallback_model"}


def median_with_spread(values: List[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
