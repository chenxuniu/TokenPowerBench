"""Local driver diagnostics for source-checkout distributed callers.

This adapter reads only the host where it is constructed. It cannot measure
cluster energy or normalize energy by the cluster's output-token count. Whole
local-node energy requires an actual readable IPMI sensor; unavailable local
sensors remain None. Sensor access starts only when a monitor is constructed.
"""

from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenpowerbench.energy import create_monitor


class PowerMonitor:
    """Collect local diagnostics; always close() in a finally block.

    device_indices selects physical NVML indices on this host. With no explicit
    indices or UUIDs, the shared monitor resolves torch-visible CUDA devices.
    """

    def __init__(self, mode="auto", device_indices=None, device_uuids=None):
        self.monitor = create_monitor(mode, device_indices=device_indices, device_uuids=device_uuids)

    def start_monitoring(self):
        self.monitor.start()

    def stop_monitoring(self):
        self.monitor.stop()

    def close(self):
        self.monitor.close()

    def calculate_metrics(self, duration, total_output_tokens=0, num_responses=None,
                          start_time=None, end_time=None):
        """Integrate an explicit driver perf_counter window without normalization.

        Pass the shared engine's measurement_window start_s and end_s values.
        Workload token/response counts are retained as metadata only.
        """
        if start_time is None or end_time is None:
            raise ValueError("Provide explicit perf_counter start_time and end_time from measurement_window")
        metrics = self.monitor.compute_metrics(duration, 0, 0, start_time=start_time, end_time=end_time)
        local = asdict(metrics)
        local.pop("total_output_tokens", None)
        local.pop("num_responses", None)
        result = {
            "duration": metrics.duration,
            "total_output_tokens": total_output_tokens,
            "responses": num_responses,
            "monitoring_scope": "local_driver_host",
            "energy_scope": "unmeasured_cluster",
            "phase_status": "not_measured_distributed_requests_may_overlap",
            "cluster_energy_j": None,
            "cluster_energy_per_token_j": None,
            "total_energy_j": None,
            "total_mj_per_token": None,
            "energy_per_token": None,
            "energy_per_token_j": None,
            "gpu_mj_per_token": None,
            "driver_diagnostics": {"monitoring_scope": "local_driver_host", "energy": local},
            "capabilities": self.monitor.capabilities,
            "power_samples": self.monitor.samples,
            "warnings": metrics.warnings,
            "formatted_output": "Local driver-host diagnostics only; cluster energy is unavailable.\n" + metrics.summary(),
        }
        # Unscoped compatibility fields cannot represent measurements of all workers.
        for label in ("gpu", "cpu", "dram", "total"):
            for suffix in ("avg_power", "energy", "energy_per_second", "energy_per_token", "energy_per_response"):
                result[f"{label}_{suffix}"] = None
        return result

    def print_metrics(self, metrics):
        print(metrics["formatted_output"])
