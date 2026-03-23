"""
tokenpowerbench.energy
======================

Energy monitoring for LLM inference benchmarking.

Two monitor implementations:

    GPUEnergyMonitor     -- GPU power only via NVIDIA NVML.
                            No elevated privileges required.
                            Suitable for any data-center / cloud / workstation user.

    FullNodeEnergyMonitor -- GPU + CPU (Intel RAPL) + total node (IPMI).
                             Requires root or read access to RAPL sysfs.
                             Provides complete node-level energy breakdown.

Quick start
-----------
    from tokenpowerbench.energy import create_monitor

    monitor = create_monitor()       # auto-detect best available mode
    monitor.start()
    run_inference(...)
    monitor.stop()
    metrics = monitor.compute_metrics(duration, total_tokens, num_responses)
    print(metrics.summary())
    print(f"GPU energy/token: {metrics.gpu_mj_per_token:.3f} mJ/token")
    print(f"Total energy/token: {metrics.total_mj_per_token:.3f} mJ/token")
"""

import os

from .base import EnergyMetrics, EnergyMonitor
from .gpu_monitor import GPUEnergyMonitor
from .full_node_monitor import FullNodeEnergyMonitor

__all__ = [
    "EnergyMetrics",
    "EnergyMonitor",
    "GPUEnergyMonitor",
    "FullNodeEnergyMonitor",
    "create_monitor",
]


def create_monitor(mode: str = "auto") -> EnergyMonitor:
    """
    Factory for energy monitors.

    Parameters
    ----------
    mode : str
        "auto"       Detect automatically. Uses FullNodeEnergyMonitor if RAPL
                     sysfs is readable (root or permitted), otherwise falls back
                     to GPUEnergyMonitor.
        "gpu_only"   GPU power via NVML only. No root required.
        "full_node"  GPU + CPU (Intel RAPL) + node total (IPMI).
                     Requires root or appropriate sysfs permissions.

    Returns
    -------
    EnergyMonitor
        Call .start() before inference, .stop() after, then .compute_metrics().
    """
    if mode == "gpu_only":
        return GPUEnergyMonitor()
    if mode == "full_node":
        return FullNodeEnergyMonitor()
    if mode == "auto":
        if _rapl_accessible():
            return FullNodeEnergyMonitor()
        return GPUEnergyMonitor()
    raise ValueError(f"Unknown monitor mode: {mode!r}. Choose 'auto', 'gpu_only', or 'full_node'.")


def _rapl_accessible() -> bool:
    """Return True if at least one RAPL energy_uj file is readable."""
    rapl_root = "/sys/class/powercap/intel-rapl"
    if not os.path.exists(rapl_root):
        return False
    try:
        for entry in os.listdir(rapl_root):
            energy_path = os.path.join(rapl_root, entry, "energy_uj")
            if os.access(energy_path, os.R_OK):
                return True
    except OSError:
        pass
    return False
