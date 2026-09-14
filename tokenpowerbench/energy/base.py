"""Energy metrics with explicit coverage and measurement scope."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EnergyMetrics:
    duration: float = 0.0
    total_output_tokens: int = 0
    num_responses: int = 0
    gpu_avg_power_w: Optional[float] = None
    gpu_energy_j: Optional[float] = None
    per_gpu_power_w: Dict[int, Optional[float]] = field(default_factory=dict)
    cpu_avg_power_w: Optional[float] = None
    cpu_energy_j: Optional[float] = None
    dram_avg_power_w: Optional[float] = None
    dram_energy_j: Optional[float] = None
    system_avg_power_w: Optional[float] = None
    system_energy_j: Optional[float] = None
    energy_scope: str = "unavailable"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    capabilities: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    sample_counts: dict = field(default_factory=dict)
    per_gpu_energy_j: dict = field(default_factory=dict)

    @property
    def total_energy_j(self) -> Optional[float]:
        """Whole-node energy is available only from the IPMI sensor."""
        return self.system_energy_j

    @property
    def components_energy_j(self) -> Optional[float]:
        """Sum of available components; this is not whole-node energy."""
        present = [value for value in (self.gpu_energy_j, self.cpu_energy_j, self.dram_energy_j)
                   if value is not None]
        return sum(present) if present else None

    @property
    def energy_per_token_j(self) -> Optional[float]:
        if self.total_energy_j is None or self.total_output_tokens <= 0:
            return None
        return self.total_energy_j / self.total_output_tokens

    @property
    def gpu_mj_per_token(self) -> Optional[float]:
        if self.gpu_energy_j is None or self.total_output_tokens <= 0:
            return None
        return self.gpu_energy_j / self.total_output_tokens * 1000

    @property
    def total_mj_per_token(self) -> Optional[float]:
        value = self.energy_per_token_j
        return None if value is None else value * 1000

    def summary(self) -> str:
        def formatted(value, unit):
            return "unavailable" if value is None else f"{value:.3f} {unit}"

        lines = [
            "Energy Metrics",
            f"  Duration: {self.duration:.3f} s; output tokens: {self.total_output_tokens}",
            f"  Scope: {self.energy_scope}",
            f"  Selected GPU energy: {formatted(self.gpu_energy_j, 'J')}",
            f"  GPU energy/token: {formatted(self.gpu_mj_per_token, 'mJ/token')}",
            f"  CPU package energy: {formatted(self.cpu_energy_j, 'J')}",
            f"  DRAM energy: {formatted(self.dram_energy_j, 'J')}",
            f"  Whole-node energy (IPMI): {formatted(self.system_energy_j, 'J')}",
            f"  Whole-node energy/token: {formatted(self.total_mj_per_token, 'mJ/token')}",
        ]
        lines.extend(f"  Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


class EnergyMonitor(ABC):
    @abstractmethod
    def start(self) -> None:
        """Take boundary samples and start background monitoring."""

    @abstractmethod
    def stop(self) -> None:
        """Stop sampling and take final boundary samples."""

    @abstractmethod
    def close(self) -> None:
        """Release monitoring resources; safe to call in finally."""

    @abstractmethod
    def compute_metrics(self, duration, total_output_tokens, num_responses,
                        start_time=None, end_time=None) -> EnergyMetrics:
        """Integrate samples over the requested perf_counter time window."""
