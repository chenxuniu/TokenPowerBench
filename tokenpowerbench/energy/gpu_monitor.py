"""Selected-device NVML sampling and timestamp-based energy integration.

No torch or NVML import occurs until a monitor is constructed. GPU selection
uses explicit physical NVML indices/UUIDs, or UUIDs of torch-visible devices.
"""

import copy
import importlib
import math
import threading
import time

from .base import EnergyMetrics, EnergyMonitor

_GPU_SAMPLE_INTERVAL = 0.1
_GPU_MINIMUM_WINDOW = 1.0


def integrate_power(readings, start_time, end_time, min_samples=2):
    """Integrate ``(perf_counter timestamp, watts or None)`` pairs.

    Boundary power is linearly interpolated. Extrapolation and gaps containing
    missing readings are forbidden. Two actual samples in the requested window
    are required by default; use min_samples=0 for mathematical interpolation
    tests. Hardware resolution is enforced separately by each monitor.
    """
    if not (math.isfinite(start_time) and math.isfinite(end_time)) or end_time <= start_time:
        raise ValueError("Energy window must have finite timestamps and positive duration")
    result = {"energy_j": None, "average_power_w": None, "valid": False,
              "sample_count": 0, "reason": None}
    points = list(readings)
    if any(not math.isfinite(t) for t, _ in points):
        result["reason"] = "non-finite sample timestamp"
        return result
    if any(right[0] <= left[0] for left, right in zip(points, points[1:])):
        result["reason"] = "sample timestamps are not strictly increasing"
        return result
    result["sample_count"] = sum(start_time <= t <= end_time for t, _ in points)
    if len(points) < 2 or points[0][0] > start_time or points[-1][0] < end_time:
        result["reason"] = "samples do not cover the complete measurement window"
        return result
    if result["sample_count"] < min_samples:
        result["reason"] = "fewer than two samples within the window; sensor cannot resolve this phase"
        return result
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        lo, hi = max(t0, start_time), min(t1, end_time)
        if hi <= lo:
            continue
        if any(p is None or not math.isfinite(p) or p < 0 for p in (p0, p1)):
            result["reason"] = "missing or invalid power reading in measurement window"
            return result
        slope = (p1 - p0) / (t1 - t0)
        power_lo, power_hi = p0 + slope * (lo - t0), p0 + slope * (hi - t0)
        energy += (power_lo + power_hi) * 0.5 * (hi - lo)
    result.update(energy_j=energy, average_power_w=energy / (end_time - start_time), valid=True)
    return result


def trim_edges(readings, frac=0.0):
    """Compatibility helper: measurements now retain all samples."""
    return list(readings)


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _torch_device_uuids():
    try:
        torch = importlib.import_module("torch")
        count = torch.cuda.device_count()
        uuids = [getattr(torch.cuda.get_device_properties(index), "uuid", None)
                 for index in range(count)]
    except Exception as exc:
        raise RuntimeError("Cannot resolve torch CUDA device UUIDs; pass device_indices "
                           "(physical NVML indices) or device_uuids explicitly") from exc
    if not uuids or any(value is None for value in uuids):
        raise RuntimeError("Torch did not expose CUDA device UUIDs; pass device_indices "
                           "(physical NVML indices) or device_uuids explicitly")
    return [_text(value) for value in uuids]


class GPUEnergyMonitor(EnergyMonitor):
    def __init__(self, device_indices=None, device_uuids=None, interval_s=0.1):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("GPU sample interval must be positive and finite")
        if device_indices is not None and device_uuids is not None:
            raise ValueError("Choose device_indices or device_uuids, not both")
        if device_indices is not None:
            if (not device_indices or any(type(i) is not int or i < 0 for i in device_indices)
                    or len(set(device_indices)) != len(device_indices)):
                raise ValueError("device_indices must contain unique nonnegative physical NVML indices")
        if device_uuids is not None and (not device_uuids or len(set(device_uuids)) != len(device_uuids)):
            raise ValueError("device_uuids must be nonempty and unique")
        self._closed = False
        self._nvml_initialized = False
        self._active = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._threads = []
        self._samples = {"gpu": [], "rapl": [], "ipmi": []}
        self._sample_errors = {}
        self._capabilities = {
            key: {"available": False, "reason": "disabled in gpu_only mode"}
            for key in ("cpu", "dram", "system")
        }
        self._start_time = self._end_time = None
        self._interval_s = interval_s
        self._handles = []
        self._devices = []
        try:
            self._nvml = importlib.import_module("pynvml")
        except ImportError as exc:
            raise RuntimeError("NVML bindings missing; install nvidia-ml-py") from exc
        try:
            self._nvml.nvmlInit()
            self._nvml_initialized = True
            if device_indices is not None:
                count = self._nvml.nvmlDeviceGetCount()
                if any(index >= count for index in device_indices):
                    raise ValueError(f"Physical NVML GPU index out of range (device count: {count})")
                selected = [self._nvml.nvmlDeviceGetHandleByIndex(i) for i in device_indices]
            else:
                selected_uuids = device_uuids if device_uuids is not None else _torch_device_uuids()
                if any(str(uuid).startswith("MIG-") for uuid in selected_uuids):
                    raise ValueError("MIG instances are unsupported: NVML power is physical-GPU scoped")
                selected = [self._nvml.nvmlDeviceGetHandleByUUID(uuid) for uuid in selected_uuids]
            for handle in selected:
                if hasattr(self._nvml, "nvmlDeviceGetMigMode"):
                    try:
                        mig_enabled = bool(self._nvml.nvmlDeviceGetMigMode(handle)[0])
                    except Exception as exc:
                        unsupported = getattr(self._nvml, "NVMLError_NotSupported", ())
                        if not isinstance(exc, unsupported):
                            raise RuntimeError(f"Cannot determine selected GPU MIG status: {exc}") from exc
                        mig_enabled = False
                    if mig_enabled:
                        raise ValueError("MIG-enabled GPUs are unsupported: physical power cannot be attributed to an instance")
                index = self._nvml.nvmlDeviceGetIndex(handle)
                uuid = _text(self._nvml.nvmlDeviceGetUUID(handle))
                if any(device["uuid"] == uuid for device in self._devices):
                    raise ValueError("Selected GPUs resolve to the same physical UUID")
                power = self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                if not math.isfinite(power) or power < 0:
                    raise RuntimeError(f"Invalid NVML power reading on GPU {index}")
                self._handles.append(handle)
                self._devices.append({"index": index, "uuid": uuid,
                                      "name": _text(self._nvml.nvmlDeviceGetName(handle))})
            self._capabilities["gpu"] = {
                "available": True, "reason": "NVML power probe succeeded",
                "scope": "selected_physical_gpus", "devices": self._devices,
                "sampling_interval_s": interval_s, "power_source": "nvmlDeviceGetPowerUsage",
                "averaging_window_s": None, "minimum_window_s": _GPU_MINIMUM_WINDOW,
                "resolution_policy": "conservative_unknown_architecture",
                "warning": "NVML can report 1-second averaged power; 100 ms polling does not imply 100 ms sensor resolution. Sub-second windows are unavailable.",
            }
        except Exception:
            self.close()
            raise

    @property
    def samples(self):
        with self._lock:
            return copy.deepcopy(self._samples)

    @property
    def capabilities(self):
        capabilities = copy.deepcopy(self._capabilities)
        for sensor, capability in capabilities.items():
            capability["sampling_errors"] = {
                key: value for key, value in self._sample_errors.items()
                if key == sensor or key.startswith(sensor + ":")
            }
        return capabilities

    def _sample_gpu(self):
        powers = []
        for device, handle in zip(self._devices, self._handles):
            try:
                power = self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                if not math.isfinite(power) or power < 0:
                    raise ValueError("invalid power value")
                powers.append(power)
            except Exception as exc:
                powers.append(None)
                self._sample_errors[f"gpu:{device['index']}"] = str(exc)
        sample = {"timestamp_s": time.perf_counter(), "power_w": powers}
        with self._lock:
            self._samples["gpu"].append(sample)

    def _samplers(self):
        return [(self._sample_gpu, self._interval_s)]

    def _sample_loop(self, sample_fn, interval):
        deadline = time.perf_counter() + interval
        while not self._stop_event.wait(max(0.0, deadline - time.perf_counter())):
            sample_fn()
            deadline += interval
            now = time.perf_counter()
            if deadline <= now:
                deadline += (math.floor((now - deadline) / interval) + 1) * interval

    def start(self):
        if self._closed:
            raise RuntimeError("Monitor is closed")
        if self._active:
            raise RuntimeError("Monitor is already running")
        self._samples = {"gpu": [], "rapl": [], "ipmi": []}
        self._sample_errors = {}
        self._stop_event.clear()
        self._end_time = None
        self._threads = []
        samplers = self._samplers()
        # Slow BMC acquisition first, so GPU/RAPL boundary reads stay close to
        # the caller's inference start even when ipmitool takes several seconds.
        for sampler, _ in reversed(samplers):
            sampler()
        self._start_time = time.perf_counter()
        self._active = True
        for sampler, interval in samplers:
            thread = threading.Thread(target=self._sample_loop, args=(sampler, interval), daemon=True)
            self._threads.append(thread)
            thread.start()

    def stop(self):
        if not self._active:
            return
        self._end_time = time.perf_counter()
        self._stop_event.set()
        for thread, (sampler, _) in zip(self._threads, self._samplers()):
            thread.join()  # IPMI subprocesses have a bounded timeout.
            sampler()
        self._active = False

    def close(self):
        if self._closed:
            return
        try:
            self.stop()
        finally:
            if self._nvml_initialized:
                try:
                    self._nvml.nvmlShutdown()
                finally:
                    self._nvml_initialized = False
            self._closed = True

    def _window(self, duration, start_time, end_time):
        if (start_time is None) != (end_time is None):
            raise ValueError("Provide both start_time and end_time")
        if start_time is None:
            if self._start_time is None:
                raise RuntimeError("Monitor has not been started")
            start_time = self._start_time
            end_time = start_time + duration
        if not (math.isfinite(start_time) and math.isfinite(end_time)) or end_time <= start_time:
            raise ValueError("Measurement duration must be positive and finite")
        return start_time, end_time

    def compute_metrics(self, duration, total_output_tokens, num_responses,
                        start_time=None, end_time=None):
        start_time, end_time = self._window(duration, start_time, end_time)
        samples = self.samples["gpu"]
        metrics = EnergyMetrics(duration=end_time - start_time,
                                total_output_tokens=total_output_tokens, num_responses=num_responses,
                                start_time=start_time, end_time=end_time, capabilities=self.capabilities)
        metrics.warnings.append(self._capabilities["gpu"]["warning"])
        energies = []
        for column, device in enumerate(self._devices):
            integrated = integrate_power([(s["timestamp_s"], s["power_w"][column]) for s in samples],
                                         start_time, end_time)
            if metrics.duration < _GPU_MINIMUM_WINDOW:
                integrated.update(energy_j=None, average_power_w=None, valid=False,
                                  reason="window shorter than conservative 1-second NVML averaging resolution")
            index = device["index"]
            metrics.per_gpu_power_w[index] = integrated["average_power_w"]
            metrics.per_gpu_energy_j[index] = integrated["energy_j"]
            metrics.sample_counts[f"gpu:{index}"] = integrated["sample_count"]
            if not integrated["valid"]:
                metrics.warnings.append(f"GPU {index}: {integrated['reason']}")
            energies.append(integrated["energy_j"])
        if energies and all(value is not None for value in energies):
            metrics.gpu_energy_j = sum(energies)
            metrics.gpu_avg_power_w = metrics.gpu_energy_j / metrics.duration
            metrics.energy_scope = "selected_gpus"
        return metrics
