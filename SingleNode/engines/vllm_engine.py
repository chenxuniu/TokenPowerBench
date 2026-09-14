"""Compatibility import; maintained vLLM implementation lives in the package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokenpowerbench.engines.vllm_engine import VLLMEngine

__all__ = ["VLLMEngine"]
