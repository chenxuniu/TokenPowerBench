"""Distributed inference adapters using TokenPowerBench's shared engine.

The optional timer argument is accepted for source-checkout callers. Phase
boundaries are unavailable for overlapping distributed requests; no timer
callback is interpreted as a first-token event.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenpowerbench.distributed import RayClusterConfig
from tokenpowerbench.distributed.predictor import VLLMPredictor as _Predictor
from tokenpowerbench.distributed.vllm_distributed import VLLMDistributedEngine as _Engine


class VLLMDistributedEngine(_Engine):
    """Accept a benchmark config and use the resolved Ray cluster address.

    A standalone run_benchmark() call prepares and closes its own workers.
    For several measured calls, use prepare(), then close() in a finally block.
    """

    def __init__(self, config, timer=None):
        super().__init__(RayClusterConfig.resolve(), config)


class VLLMPredictor(_Predictor):
    """Accept the optional timer argument and use strict shared batch results.

    Batches must include parallel text and request_id columns. Results contain
    token-ID counts and one shared batch duration, without per-request latency.
    """

    def __init__(self, model_path, tensor_parallel_size, pipeline_parallel_size,
                 sampling_params, timer=None, verbose=False, model_kwargs=None):
        super().__init__(model_path, tensor_parallel_size, pipeline_parallel_size,
                         sampling_params, verbose=verbose, model_kwargs=model_kwargs)
