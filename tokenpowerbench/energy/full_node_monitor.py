"""
Full-node energy monitor for root / sudo users.

Extends GPUEnergyMonitor with:
  - CPU & DRAM energy via Intel RAPL sysfs interface
  - Total node power via IPMI dcmi

Privilege requirements
----------------------
Intel RAPL  : Requires either root or that the sysfs files under
              /sys/class/powercap/intel-rapl are readable by the user.
              Some HPC systems grant this to all users; on others you need
              ``sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj``
              or a kernel module like ``msr-safe``.

IPMI dcmi   : Requires root or sudo access to ipmitool, or that the BMC
              channel is configured for non-root access.

If neither RAPL nor IPMI is accessible, this monitor falls back to
GPU-only metrics (identical to GPUEnergyMonitor).
"""

import os
import subprocess
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from .base import EnergyMetrics
from .gpu_monitor import GPUEnergyMonitor, trim_edges

_RAPL_ROOT = "/sys/class/powercap/intel-rapl"
_CPU_SAMPLE_INTERVAL = 0.5   # 500 ms — RAPL counter precision ~1 s
_IPMI_SAMPLE_INTERVAL = 1.0  # 1 s — ipmitool round-trip is slow


class FullNodeEnergyMonitor(GPUEnergyMonitor):
    """
    Collects GPU + CPU/DRAM (RAPL) + total node (IPMI) power.

    Use ``create_monitor("full_node")`` rather than instantiating directly.
    """

    def __init__(self) -> None:
        super().__init__()
        self._rapl = _RaplReader()
        self._cpu_readings: List[Dict[str, float]] = []
        self._system_readings: List[float] = []
        self._cpu_thread: Optional[threading.Thread] = None
        self._ipmi_thread: Optional[threading.Thread] = None

        if self._rapl.domains:
            print(f"[FullNodeEnergyMonitor] RAPL domains found: {list(self._rapl.domains)}")
        else:
            print("[FullNodeEnergyMonitor] No RAPL domains accessible (need root or sysfs permissions).")

        if _ipmi_available():
            print("[FullNodeEnergyMonitor] IPMI power readings available.")
        else:
            print("[FullNodeEnergyMonitor] IPMI not available (ipmitool missing or no permission).")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._cpu_readings = []
        self._system_readings = []
        self._active = True

        # GPU sampling thread (from parent)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

        # CPU / DRAM via RAPL
        if self._rapl.domains:
            self._cpu_thread = threading.Thread(
                target=self._rapl_sample_loop, daemon=True
            )
            self._cpu_thread.start()

        # Total node power via IPMI
        if _ipmi_available():
            self._ipmi_thread = threading.Thread(
                target=self._ipmi_sample_loop, daemon=True
            )
            self._ipmi_thread.start()

        time.sleep(2.0)  # baseline settle

    def stop(self) -> None:
        self._active = False
        for t in (self._thread, self._cpu_thread, self._ipmi_thread):
            if t is not None:
                t.join(timeout=2.0)
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def compute_metrics(
        self,
        duration: float,
        total_output_tokens: int,
        num_responses: int,
    ) -> EnergyMetrics:
        # GPU metrics from parent
        metrics = super().compute_metrics(duration, total_output_tokens, num_responses)

        with self._lock:
            cpu_samples = list(self._cpu_readings)
            sys_samples = list(self._system_readings)

        # ---- CPU / DRAM via RAPL ----
        cpu_samples = trim_edges(cpu_samples)
        cpu_avg_w = 0.0
        dram_avg_w = 0.0
        if cpu_samples:
            by_domain = _group_by_domain(cpu_samples)
            for domain, values in by_domain.items():
                avg = _robust_mean(values)
                if "dram" in domain.lower():
                    dram_avg_w += avg
                else:
                    cpu_avg_w += avg

        metrics.cpu_avg_power_w = cpu_avg_w
        metrics.cpu_energy_j = cpu_avg_w * duration
        metrics.dram_avg_power_w = dram_avg_w
        metrics.dram_energy_j = dram_avg_w * duration

        # ---- Node total via IPMI ----
        sys_samples = trim_edges(sys_samples)
        if sys_samples:
            metrics.system_avg_power_w = float(_robust_mean(sys_samples))
            metrics.system_energy_j = metrics.system_avg_power_w * duration

        return metrics

    # ------------------------------------------------------------------
    # Internal sampling loops
    # ------------------------------------------------------------------

    def _rapl_sample_loop(self) -> None:
        prev_energy = self._rapl.read_energy()
        prev_time = time.time()
        while self._active:
            time.sleep(_CPU_SAMPLE_INTERVAL)
            curr_time = time.time()
            curr_energy = self._rapl.read_energy()
            dt = curr_time - prev_time
            if dt > 0:
                power = self._rapl.compute_power(prev_energy, curr_energy, dt)
                with self._lock:
                    self._cpu_readings.append(power)
            prev_energy = curr_energy
            prev_time = curr_time

    def _ipmi_sample_loop(self) -> None:
        while self._active:
            w = _read_ipmi_power()
            if w is not None:
                with self._lock:
                    self._system_readings.append(w)
            time.sleep(_IPMI_SAMPLE_INTERVAL)


# ---------------------------------------------------------------------------
# RAPL helper
# ---------------------------------------------------------------------------

class _RaplReader:
    """Low-level reader for Intel RAPL sysfs energy counters."""

    def __init__(self) -> None:
        self.domains: Dict[str, str] = {}      # name -> sysfs directory path
        self._max_energy: Dict[str, int] = {}   # for wraparound handling
        if not os.path.exists(_RAPL_ROOT):
            return
        self.domains = self._discover()
        for name, path in self.domains.items():
            try:
                with open(os.path.join(path, "max_energy_range_uj")) as f:
                    self._max_energy[name] = int(f.read())
            except OSError:
                self._max_energy[name] = 2**32  # safe fallback

    def _discover(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        try:
            for entry in os.listdir(_RAPL_ROOT):
                if not entry.startswith("intel-rapl:"):
                    continue
                domain_path = os.path.join(_RAPL_ROOT, entry)
                try:
                    name = open(os.path.join(domain_path, "name")).read().strip()
                except OSError:
                    continue
                out[name] = domain_path
                # Sub-domains (e.g., package-0-dram)
                for sub in os.listdir(domain_path):
                    if not sub.startswith("intel-rapl:"):
                        continue
                    sub_path = os.path.join(domain_path, sub)
                    try:
                        sub_name = open(os.path.join(sub_path, "name")).read().strip()
                    except OSError:
                        continue
                    out[f"{name}-{sub_name}"] = sub_path
        except OSError:
            pass
        return out

    def read_energy(self) -> Dict[str, Optional[int]]:
        """Read current energy_uj counter for every domain."""
        out: Dict[str, Optional[int]] = {}
        for name, path in self.domains.items():
            try:
                with open(os.path.join(path, "energy_uj")) as f:
                    out[name] = int(f.read())
            except OSError:
                out[name] = None
        return out

    def compute_power(
        self,
        prev: Dict[str, Optional[int]],
        curr: Dict[str, Optional[int]],
        dt: float,
    ) -> Dict[str, float]:
        """Convert two energy snapshots to instantaneous power (W)."""
        out: Dict[str, float] = {}
        for name in self.domains:
            p, c = prev.get(name), curr.get(name)
            if p is None or c is None:
                out[name] = 0.0
                continue
            diff = c - p
            if diff < 0:  # counter wraparound
                diff += self._max_energy.get(name, 2**32)
            out[name] = (diff / 1_000_000) / dt  # µJ → J → W
        return out


# ---------------------------------------------------------------------------
# IPMI helper
# ---------------------------------------------------------------------------

def _ipmi_available() -> bool:
    """Return True if ipmitool dcmi power reading succeeds."""
    try:
        result = subprocess.run(
            ["ipmitool", "dcmi", "power", "reading"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _read_ipmi_power() -> Optional[float]:
    """Return instantaneous node power in watts, or None on failure."""
    try:
        out = subprocess.check_output(
            ["ipmitool", "dcmi", "power", "reading"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        )
        for line in out.splitlines():
            if "Instantaneous power reading" in line:
                return float(line.split(":")[1].strip().split()[0])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _group_by_domain(
    samples: List[Dict[str, float]]
) -> Dict[str, List[float]]:
    """Transpose a list-of-dicts into a dict-of-lists."""
    out: Dict[str, List[float]] = {}
    for sample in samples:
        for domain, val in sample.items():
            out.setdefault(domain, []).append(val)
    return out


def _robust_mean(values: List[float], max_plausible: float = 1000.0) -> float:
    """Mean with outlier removal (3-sigma) and a sanity-cap."""
    arr = np.array([v for v in values if 0 < v <= max_plausible])
    if len(arr) == 0:
        return 0.0
    if len(arr) > 5:
        mean, std = np.mean(arr), np.std(arr)
        mask = (arr >= mean - 3 * std) & (arr <= mean + 3 * std)
        if mask.sum() > 0:
            arr = arr[mask]
    return float(np.mean(arr))
