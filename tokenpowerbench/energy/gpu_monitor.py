"""
GPU-only energy monitor via NVIDIA NVML.

No elevated privileges required. Works for any data-center or workstation
user who has CUDA access.
"""

import threading
import time
from typing import List

import numpy as np

from .base import EnergyMonitor, EnergyMetrics

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

_GPU_SAMPLE_INTERVAL = 0.1  # 100 ms — NVML updates at ~100 ms granularity
_EDGE_TRIM_FRAC = 0.10      # Drop first/last 10% of samples (ramp-up / idle tail)


def trim_edges(readings: list, frac: float = _EDGE_TRIM_FRAC) -> list:
    """Remove the first and last `frac` fraction of a reading list."""
    n = len(readings)
    if n == 0:
        return readings
    buf = max(1, int(n * frac))
    if n <= buf * 2:
        return readings
    return readings[buf:-buf]


class GPUEnergyMonitor(EnergyMonitor):
    """
    Monitors GPU power draw via NVML at 100 ms intervals.

    Suitable for any user with CUDA access — no root required.
    """

    def __init__(self) -> None:
        if not _NVML_AVAILABLE:
            raise RuntimeError(
                "pynvml is not installed. Run: pip install nvidia-ml-py"
            )
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        print(f"[GPUEnergyMonitor] Found {n} GPU(s):")
        for i, h in enumerate(self._handles):
            name = pynvml.nvmlDeviceGetName(h)
            print(f"  GPU {i}: {name}")

        self._readings: List[List[float]] = []
        self._lock = threading.Lock()
        self._active = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._readings = []
        self._active = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        time.sleep(2.0)  # let GPU settle before inference starts

    def stop(self) -> None:
        self._active = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def compute_metrics(
        self,
        duration: float,
        total_output_tokens: int,
        num_responses: int,
    ) -> EnergyMetrics:
        with self._lock:
            readings = list(self._readings)

        readings = trim_edges(readings)
        if not readings:
            return EnergyMetrics(
                duration=duration,
                total_output_tokens=total_output_tokens,
                num_responses=num_responses,
            )

        arr = np.array(readings)           # shape: (n_samples, n_gpus)
        per_gpu_avg = np.mean(arr, axis=0) # shape: (n_gpus,)
        total_gpu_w = float(np.sum(per_gpu_avg))

        return EnergyMetrics(
            duration=duration,
            total_output_tokens=total_output_tokens,
            num_responses=num_responses,
            gpu_avg_power_w=total_gpu_w,
            gpu_energy_j=total_gpu_w * duration,
            per_gpu_power_w={i: float(v) for i, v in enumerate(per_gpu_avg)},
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_loop(self) -> None:
        while self._active:
            sample = []
            for handle in self._handles:
                try:
                    mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    sample.append(mw / 1000.0)  # mW → W
                except Exception:
                    sample.append(0.0)
            with self._lock:
                self._readings.append(sample)
            time.sleep(_GPU_SAMPLE_INTERVAL)
