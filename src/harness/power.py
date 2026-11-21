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
        self._e0_mj: Optional[int] = None
        self._e1_mj: Optional[int] = None

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

    def _counter_mj(self) -> Optional[int]:
        """Hardware cumulative energy counter (mJ since driver load); None if not exposed."""
        try:
            return int(self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
        except Exception:
            return None

    def __enter__(self):
        self.available = self._init_nvml()
        if self.available:
            self._e0_mj = self._counter_mj()
        self.t_start = time.time()
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self.t_end = time.time()
        if self.available and self._e0_mj is not None:
            self._e1_mj = self._counter_mj()
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
    def counter_energy_j(self) -> float:
        if self._e0_mj is None or self._e1_mj is None or self._e1_mj < self._e0_mj:
            return float("nan")
        return (self._e1_mj - self._e0_mj) / 1000.0

    @property
    def polled_energy_j(self) -> float:
        if len(self.samples) >= 2:
            trapezoid = getattr(np, "trapezoid", None) or np.trapz  # np.trapz was removed in NumPy 2.x
            return float(trapezoid(self.samples, self.times))
        return float("nan")

    @property
    def energy_j(self) -> float:
        """Primary: NVML hardware energy counter (amendment 2026-11-21, Q5).  Then the 10 Hz polled
        trapezoid; then W_active * wall time."""
        if np.isfinite(self.counter_energy_j):
            return self.counter_energy_j
        if len(self.samples) >= 2:
            trapezoid = getattr(np, "trapezoid", None) or np.trapz  # np.trapz was removed in NumPy 2.x
            return float(trapezoid(self.samples, self.times))
        return self.fallback_w * self.wall_s

    def summary(self) -> Dict[str, float]:
        return {"energy_j": self.energy_j, "wall_s": self.wall_s, "mean_w": self.mean_w, "n_samples": len(self.samples), "nvml_available": self.available, "energy_source": "nvml_counter" if np.isfinite(self.counter_energy_j) else ("nvml_polled" if len(self.samples) >= 2 else "fallback_model"), "counter_energy_j": self.counter_energy_j, "polled_energy_j": self.polled_energy_j}


def median_with_spread(values: List[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
