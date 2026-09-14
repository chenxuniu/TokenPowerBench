"""Distributed inference components, imported without initializing Ray or vLLM."""

from importlib import import_module

from .ray_cluster import RayClusterConfig

__all__ = ["RayClusterConfig", "VLLMDistributedEngine", "VLLMPredictor"]


def __getattr__(name):
    modules = {"VLLMDistributedEngine": ".vllm_distributed", "VLLMPredictor": ".predictor"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
