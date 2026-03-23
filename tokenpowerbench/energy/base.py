"""
Abstract base classes for energy monitoring.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EnergyMetrics:
    """
    Aggregated energy metrics for one benchmark run.

    Fields populated in all modes (GPU-only or full-node):
        gpu_avg_power_w, gpu_energy_j, per_gpu_power_w

    Fields populated only in full-node mode (root/sudo):
        cpu_avg_power_w, cpu_energy_j  – Intel RAPL package domains
        dram_avg_power_w, dram_energy_j – Intel RAPL DRAM domains
        system_avg_power_w, system_energy_j – IPMI node-level total
    """

    # Required: set by caller
    duration: float = 0.0
    total_output_tokens: int = 0
    num_responses: int = 0

    # GPU (NVML) — always collected
    gpu_avg_power_w: float = 0.0
    gpu_energy_j: float = 0.0
    per_gpu_power_w: Dict[int, float] = field(default_factory=dict)

    # CPU (Intel RAPL) — root / sudo only
    cpu_avg_power_w: float = 0.0
    cpu_energy_j: float = 0.0

    # DRAM (Intel RAPL) — root / sudo only
    dram_avg_power_w: float = 0.0
    dram_energy_j: float = 0.0

    # Full node (IPMI dcmi) — root / sudo only
    system_avg_power_w: float = 0.0
    system_energy_j: float = 0.0

    # ---------- derived properties ----------

    @property
    def total_energy_j(self) -> float:
        """Best-available total energy estimate.

        Priority:
            1. IPMI system energy (most accurate, covers everything)
            2. GPU + CPU + DRAM from RAPL
            3. GPU only
        """
        if self.system_energy_j > 0:
            return self.system_energy_j
        if self.cpu_energy_j > 0 or self.dram_energy_j > 0:
            return self.gpu_energy_j + self.cpu_energy_j + self.dram_energy_j
        return self.gpu_energy_j

    @property
    def energy_per_token_j(self) -> float:
        return self.total_energy_j / self.total_output_tokens if self.total_output_tokens > 0 else 0.0

    @property
    def gpu_mj_per_token(self) -> float:
        return self.gpu_energy_j / self.total_output_tokens * 1000 if self.total_output_tokens > 0 else 0.0

    @property
    def total_mj_per_token(self) -> float:
        return self.energy_per_token_j * 1000

    def summary(self) -> str:
        lines = [
            "Energy Metrics",
            "=" * 50,
            f"  Duration:          {self.duration:.2f} s",
            f"  Output tokens:     {self.total_output_tokens}",
            f"  Responses:         {self.num_responses}",
            "",
            "  -- GPU (NVML) --",
            f"  Avg power:         {self.gpu_avg_power_w:.1f} W",
            f"  Energy:            {self.gpu_energy_j:.1f} J",
            f"  Energy/token:      {self.gpu_mj_per_token:.3f} mJ/token",
        ]
        if self.per_gpu_power_w:
            for idx, w in sorted(self.per_gpu_power_w.items()):
                lines.append(f"    GPU {idx}:           {w:.1f} W")

        if self.cpu_avg_power_w > 0 or self.dram_avg_power_w > 0:
            lines += [
                "",
                "  -- CPU / DRAM (Intel RAPL) --",
                f"  CPU avg power:     {self.cpu_avg_power_w:.1f} W",
                f"  CPU energy:        {self.cpu_energy_j:.1f} J",
                f"  DRAM avg power:    {self.dram_avg_power_w:.1f} W",
                f"  DRAM energy:       {self.dram_energy_j:.1f} J",
            ]

        if self.system_avg_power_w > 0:
            lines += [
                "",
                "  -- Node total (IPMI) --",
                f"  System avg power:  {self.system_avg_power_w:.1f} W",
                f"  System energy:     {self.system_energy_j:.1f} J",
            ]

        lines += [
            "",
            "  -- Combined --",
            f"  Total energy:      {self.total_energy_j:.1f} J",
            f"  Energy/token:      {self.total_mj_per_token:.3f} mJ/token",
            "=" * 50,
        ]
        return "\n".join(lines)


class EnergyMonitor(ABC):
    """Interface for energy monitors."""

    @abstractmethod
    def start(self) -> None:
        """Start background sampling threads. Blocks ~2 s for baseline settling."""

    @abstractmethod
    def stop(self) -> None:
        """Stop background sampling threads."""

    @abstractmethod
    def compute_metrics(
        self,
        duration: float,
        total_output_tokens: int,
        num_responses: int,
    ) -> EnergyMetrics:
        """Compute metrics from the collected samples."""
