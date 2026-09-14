"""Ray actor that submits complete prompt batches to one distributed vLLM engine."""

from __future__ import annotations

from collections.abc import Mapping
import gc
import math
import os
import sys
import time
import uuid


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class VLLMPredictor:
    """Own a vLLM engine in a Ray actor and return exact token counts.

    Ray determines GPU visibility and captures the engine's worker tasks in the
    actor's placement group. This actor never widens CUDA_VISIBLE_DEVICES.
    Inference dependencies are imported only when constructing the actor.
    """

    def __init__(self, model_path: str, tensor_parallel_size: int,
                 pipeline_parallel_size: int, sampling_params,
                 verbose: bool = False, model_kwargs: dict | None = None):
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("model_path must be a nonempty string")
        _positive_integer("tensor_parallel_size", tensor_parallel_size)
        _positive_integer("pipeline_parallel_size", pipeline_parallel_size)
        if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
            raise TypeError("model_kwargs must be a mapping")
        extra = dict(model_kwargs or {})
        reserved = {"model", "tensor_parallel_size", "pipeline_parallel_size",
                    "distributed_executor_backend"}
        if reserved.intersection(extra):
            raise ValueError("model_kwargs cannot override model, parallelism, or Ray executor settings")

        from vllm import LLM, SamplingParams

        self.sampling_params = (SamplingParams(**dict(sampling_params))
                                if isinstance(sampling_params, Mapping) else sampling_params)
        if getattr(self.sampling_params, "n", None) != 1:
            raise ValueError("Distributed benchmarking requires exactly one completion per request (n=1)")
        sampling_seed = getattr(self.sampling_params, "seed", None)
        if sampling_seed is not None:
            if "seed" in extra and extra["seed"] != sampling_seed:
                raise ValueError("model_kwargs seed must match the sampling seed")
            extra.setdefault("seed", sampling_seed)
        self.verbose = verbose
        self.llm = None
        self.model_options = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "trust_remote_code": True,
            "enable_prefix_caching": False,
            **extra,
            "distributed_executor_backend": "ray",
        }
        self.llm = LLM(**self.model_options)

    def ready(self):
        """Confirm construction and report the worker's engine configuration."""
        self._require_model()
        from ..engines.vllm_engine import _engine_config
        return {"engine_config": _engine_config(self.llm, self.model_options),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def warmup(self):
        """Complete a short inference without changing benchmark sampling settings."""
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=1, temperature=0, n=1,
                                seed=getattr(self.sampling_params, "seed", None))
        return self._generate({"request_id": ["warmup"], "text": ["Hello"]}, params)

    def _require_model(self):
        if self.llm is None:
            raise RuntimeError("The predictor is closed")
        return self.llm

    def __call__(self, batch):
        """Return one row per request; duration describes the shared batch window."""
        return self._generate(batch, self.sampling_params)

    def _generate(self, batch, params):
        if not isinstance(batch, Mapping) or "text" not in batch or "request_id" not in batch:
            raise ValueError("A batch must contain text and stable request_id columns")
        if isinstance(batch["text"], (str, bytes)) or isinstance(batch["request_id"], (str, bytes)):
            raise TypeError("Batch columns must contain sequences, not single strings")
        prompts, request_ids = list(batch["text"]), list(batch["request_id"])
        if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError("A batch must contain nonempty prompt strings")
        if len(request_ids) != len(prompts) or any(
                isinstance(value, bool) or not isinstance(value, (str, int)) or value == ""
                for value in request_ids):
            raise ValueError("request_id must contain one integer or nonempty string per prompt")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_id values must be unique within a batch")
        llm = self._require_model()
        started = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("Batch duration must be positive and finite")
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM returned a different number of responses than submitted prompts")
        columns = {name: [] for name in ("request_id", "prompt", "generated_text", "input_tokens",
                                         "output_tokens", "batch_id", "batch_duration_s", "batch_size")}
        batch_id = uuid.uuid4().hex
        # LLM.generate returns completed outputs in input order.
        for request_id, prompt, output in zip(request_ids, prompts, outputs):
            if not getattr(output, "finished", False):
                raise RuntimeError("vLLM returned an unfinished response")
            returned_prompt = getattr(output, "prompt", None)
            if returned_prompt is not None and returned_prompt != prompt:
                raise RuntimeError("vLLM responses do not match submitted prompt order")
            completions = getattr(output, "outputs", None)
            if not completions or len(completions) != 1:
                raise RuntimeError("vLLM must return one completion for each request")
            finish_reason = getattr(completions[0], "finish_reason", None)
            if finish_reason is not None and finish_reason not in ("stop", "length"):
                raise RuntimeError(f"vLLM request did not complete normally: {finish_reason}")
            token_ids = getattr(completions[0], "token_ids", None)
            input_ids = getattr(output, "prompt_token_ids", None)
            text = getattr(completions[0], "text", None)
            if token_ids is None or len(token_ids) == 0 or input_ids is None:
                raise RuntimeError("vLLM response lacks prompt or generated token IDs")
            if not isinstance(text, str):
                raise RuntimeError("vLLM response lacks generated text")
            values = (request_id, prompt, text, len(input_ids), len(token_ids), batch_id, elapsed, len(prompts))
            for name, value in zip(columns, values):
                columns[name].append(value)
        if self.verbose:
            print(f"[VLLMPredictor] {len(prompts)} requests, {sum(columns['output_tokens'])} tokens, {elapsed:.3f} s")
        return columns

    def close(self):
        """Release vLLM workers before the owner removes this actor's placement group."""
        llm = self.llm
        if llm is None:
            return
        self.llm = None
        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        shutdown = getattr(core, "shutdown", None)
        executor = None
        if not callable(shutdown):
            executor = getattr(engine, "model_executor", None)
            shutdown = getattr(executor, "shutdown", None)
        try:
            if not callable(shutdown):
                raise RuntimeError("vLLM exposes no supported shutdown method")
            shutdown()
        finally:
            if executor is not None:
                engine.model_executor = None
            llm = engine = core = executor = shutdown = None
            gc.collect()
            cuda = getattr(sys.modules.get("torch"), "cuda", None)
            initialized = getattr(cuda, "is_initialized", None)
            if callable(initialized) and initialized():
                cuda.empty_cache()
