"""Compatibility adapter for the legacy single-node benchmark.

Measurement lives in tokenpowerbench.energy. Unavailable sensors return None;
only IPMI is called total node energy. No import-time downloads or installs.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenpowerbench.energy import create_monitor


class PowerMonitor:
    def __init__(self, mode="auto"):
        self.monitor = create_monitor(mode)

    def start_monitoring(self):
        self.monitor.start()

    def stop_monitoring(self):
        self.monitor.stop()

    def close(self):
        self.monitor.close()

    def calculate_metrics(self, duration, total_output_tokens, num_responses=None,
                          start_time=None, end_time=None):
        m = self.monitor.compute_metrics(duration, total_output_tokens, num_responses or 0,
                                         start_time=start_time, end_time=end_time)
        result = {"duration": duration, "total_output_tokens": total_output_tokens,
                  "responses": num_responses, "capabilities": self.monitor.capabilities,
                  "power_samples": self.monitor.samples, "warnings": m.warnings,
                  "energy_scope": m.energy_scope, "phase_status": "not_measured"}
        for label, source in (("gpu", "gpu"), ("cpu", "cpu"), ("dram", "dram"), ("total", "system")):
            power, energy = getattr(m, source + "_avg_power_w"), getattr(m, source + "_energy_j")
            result[label + "_avg_power"] = power
            result[label + "_energy"] = energy
            result[label + "_energy_per_second"] = power
            result[label + "_energy_per_token"] = energy / total_output_tokens if energy is not None and total_output_tokens else None
            result[label + "_energy_per_response"] = energy / num_responses if energy is not None and num_responses else None
        result["energy_per_token"] = result["total_energy_per_token"]
        result["formatted_output"] = m.summary()
        return result

    def print_metrics(self, metrics):
        print(metrics["formatted_output"])
