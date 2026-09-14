"""Measure LLM inference energy with explicit sensor scope and phase windows."""

__version__ = "1.0.0"

from .api import BenchmarkResult, benchmark, check_environment
from .energy import EnergyMetrics, create_monitor

__all__ = ["benchmark", "check_environment", "BenchmarkResult", "create_monitor", "EnergyMetrics", "__version__"]
