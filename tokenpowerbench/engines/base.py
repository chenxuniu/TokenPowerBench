"""Abstract base class for single-node inference engines."""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class InferenceEngine(ABC):
    """Common interface for vLLM, Transformers, DeepSpeed, TensorRT-LLM."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """Return True if the required library is installed."""

    @abstractmethod
    def setup_model(self, model_path: str, **kwargs) -> Any:
        """Load the model. Returns the model object, or None on failure."""

    @abstractmethod
    def run_inference(
        self, prompts: List[str], batch_size: int, max_tokens: int, **kwargs
    ) -> List[Any]:
        """Run a single inference pass. Returns raw engine outputs."""

    @abstractmethod
    def run_benchmark(
        self, prompts: List[str], num_samples: int, batch_size: int, max_tokens: int
    ) -> Tuple[List[Any], float, float]:
        """Run the full benchmark sweep.

        Returns (outputs, start_time, end_time).
        """

    @abstractmethod
    def estimate_tokens(self, outputs: List[Any]) -> int:
        """Estimate total generated tokens from raw outputs."""
