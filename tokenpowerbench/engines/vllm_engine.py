"""vLLM inference with optional serial, host-observed phase profiling."""

from __future__ import annotations

import importlib.util
import math
import time
import uuid
from typing import Any, List, Optional, Tuple

from .base import InferenceEngine
from ..phases import PhaseTimingError, build_phase_event


class VLLMEngine(InferenceEngine):
    """Single-node vLLM wrapper; CUDA dependencies load only when used."""

    def __init__(self) -> None:
        self._llm: Optional[Any] = None
        self._phase_profiling = False
        self._seed = 0
        self._temperature = 0.7
        self.resolved_config: dict = {}

    @property
    def available(self) -> bool:
        return importlib.util.find_spec("vllm") is not None

    def setup_model(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        *,
        phase_profiling: bool = False,
        seed: int = 0,
        temperature: float = 0.7,
        tensor_parallel_size: Optional[int] = None,
    ) -> Any:
        """Load vLLM, with isolated scheduling when phase profiling is enabled.

        Phase mode requests disabled prefix caching and chunked prefill and
        admits one sequence at a time. Effective settings are recorded because
        some vLLM versions override requested settings. Observed boundaries
        include host/scheduler overhead and cannot establish exact GPU kernel
        execution intervals.
        """
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if max_model_len is not None:
            _positive_int("max_model_len", max_model_len)
        if tensor_parallel_size is not None:
            _positive_int("tensor_parallel_size", tensor_parallel_size)
        if not self.available:
            raise RuntimeError("vLLM is not installed. Install tokenpowerbench[vllm].")

        import torch
        from vllm import LLM

        torch.cuda.empty_cache()
        tp = tensor_parallel_size or max(torch.cuda.device_count(), 1)
        kwargs = {
            "model": model_path,
            "tensor_parallel_size": tp,
            "gpu_memory_utilization": gpu_memory_utilization,
            "seed": seed,
            "trust_remote_code": True,
        }
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if phase_profiling:
            kwargs.update(
                enable_prefix_caching=False,
                enable_chunked_prefill=False,
                max_num_seqs=1,
            )

        self._llm = LLM(**kwargs)
        self.resolved_config = _engine_config(self._llm, kwargs)
        self._phase_profiling = phase_profiling
        self._seed = seed
        self._temperature = temperature
        return self._llm

    def _sampling_params(
        self, max_tokens: int, temperature: Optional[float] = None, *, profile: bool = False
    ) -> Any:
        from vllm import SamplingParams

        kwargs = {
            "max_tokens": max_tokens,
            "temperature": self._temperature if temperature is None else temperature,
            "seed": self._seed,
            "n": 1,
        }
        if profile:
            from vllm.sampling_params import RequestOutputKind

            kwargs["output_kind"] = RequestOutputKind.CUMULATIVE
        return SamplingParams(**kwargs)

    def _require_model(self) -> Any:
        if self._llm is None:
            raise RuntimeError("Call setup_model before running inference.")
        return self._llm

    def run_inference(
        self,
        prompts: List[str],
        batch_size: int,
        max_tokens: int = 200,
        temperature: Optional[float] = None,
    ) -> List[Any]:
        _validate_workload(prompts, len(prompts), batch_size, max_tokens)
        if temperature is not None and (not math.isfinite(temperature) or temperature < 0):
            raise ValueError("temperature must be finite and nonnegative")
        llm = self._require_model()
        params = self._sampling_params(max_tokens, temperature)
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            outputs = llm.generate(batch, params)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned a different number of responses than submitted prompts.")
            results.extend(outputs)
        return results

    def run_benchmark(
        self,
        prompts: List[str],
        num_samples: int,
        batch_size: int,
        max_tokens: int,
    ) -> Tuple[List[Any], float, float]:
        _validate_workload(prompts, num_samples, batch_size, max_tokens)
        self._require_model()
        full = [prompts[i % len(prompts)] for i in range(num_samples)]
        all_outputs: List[Any] = []
        t0 = time.perf_counter()
        for i in range(0, num_samples, batch_size):
            batch = full[i:i + batch_size]
            all_outputs.extend(self.run_inference(batch, batch_size, max_tokens))
        t1 = time.perf_counter()
        return all_outputs, t0, t1

    def run_profiled_benchmark(
        self,
        prompts: List[str],
        num_samples: int,
        batch_size: int,
        max_tokens: int,
    ) -> Tuple[List[Any], float, float, List[dict]]:
        """Observe each request's first token using incremental engine steps.

        Each step timestamp is taken after the synchronous engine call returns.
        These are host-observed proxies, not GPU kernel timestamps. A request
        is completed before submitting the next to avoid mixed request phases.
        Engines that coalesce the first output tokens are rejected.
        """
        _validate_workload(prompts, num_samples, batch_size, max_tokens)
        llm = self._require_model()
        if batch_size != 1:
            raise ValueError("Phase profiling requires batch_size=1 for isolated requests.")
        if not self._phase_profiling:
            raise PhaseTimingError("Load the model with phase_profiling=True first.")
        engine = getattr(llm, "llm_engine", None)
        methods = ("add_request", "step", "has_unfinished_requests", "abort_request")
        if engine is None or any(not callable(getattr(engine, name, None)) for name in methods):
            raise PhaseTimingError("This vLLM version lacks the required incremental engine API.")
        if engine.has_unfinished_requests():
            raise PhaseTimingError("Phase profiling requires an idle engine.")

        full = [prompts[i % len(prompts)] for i in range(num_samples)]
        params = self._sampling_params(max_tokens, profile=True)
        run_id = uuid.uuid4().hex
        all_outputs, events = [], []
        t0 = time.perf_counter()
        for index, prompt in enumerate(full):
            request_id = f"tpbench-{run_id}-{index}"
            submitted = time.perf_counter()
            first_token = None
            final_output = None
            previous_ids: tuple = ()
            try:
                engine.add_request(request_id, prompt, params)
                # V1 may already be executing in a background process by now.
                # The submission timestamp is the start of the prefill proxy.
                dispatch_completed = time.perf_counter()
                while engine.has_unfinished_requests():
                    step_outputs = engine.step()
                    observed = time.perf_counter()
                    for output in step_outputs:
                        if str(getattr(output, "request_id", "")) != request_id:
                            raise PhaseTimingError("Unexpected concurrent request in phase profiling.")
                        completions = getattr(output, "outputs", None)
                        if not completions:
                            if getattr(output, "finished", False):
                                raise PhaseTimingError("Request finished without generated token IDs.")
                            continue
                        if len(completions) != 1:
                            raise PhaseTimingError("Phase profiling requires one completion per request.")
                        token_ids = getattr(completions[0], "token_ids", None)
                        if token_ids is None:
                            raise PhaseTimingError("vLLM output does not expose generated token IDs.")
                        current_ids = tuple(token_ids)
                        if current_ids[:len(previous_ids)] != previous_ids:
                            raise PhaseTimingError("Engine outputs are not cumulative token IDs.")
                        if current_ids and first_token is None:
                            if len(current_ids) != 1:
                                raise PhaseTimingError(
                                    "First engine observation contains multiple tokens; "
                                    "the prefill/decode boundary cannot be measured."
                                )
                            first_token = observed
                        previous_ids = current_ids
                        if getattr(output, "finished", False):
                            if first_token is None:
                                raise PhaseTimingError("Request finished without a first token.")
                            if final_output is not None:
                                raise PhaseTimingError("Duplicate finished output for a request.")
                            input_ids = getattr(output, "prompt_token_ids", None)
                            if input_ids is None:
                                raise PhaseTimingError("vLLM output does not expose prompt token IDs.")
                            event = build_phase_event(
                                request_id=request_id,
                                submitted_s=submitted,
                                dispatch_completed_s=dispatch_completed,
                                first_token_s=first_token,
                                finished_s=observed,
                                input_tokens=len(input_ids),
                                output_tokens=len(current_ids),
                            )
                            final_output = output
                    if final_output is not None:
                        if engine.has_unfinished_requests():
                            raise PhaseTimingError("Engine is not idle after the profiled request finished.")
                        break
                if final_output is None:
                    raise PhaseTimingError("Engine stopped without a completed request output.")
            except BaseException:
                try:
                    engine.abort_request([request_id])
                except Exception:
                    pass
                raise
            all_outputs.append(final_output)
            events.append(event)
        t1 = time.perf_counter()
        return all_outputs, t0, t1, events

    def estimate_tokens(self, outputs: List[Any]) -> int:
        """Count generated token IDs exactly; retained name for compatibility."""
        total = 0
        for output in outputs:
            completions = getattr(output, "outputs", None)
            if not completions:
                raise ValueError("Cannot count tokens: response has no completions.")
            for completion in completions:
                token_ids = getattr(completion, "token_ids", None)
                if token_ids is None:
                    raise ValueError("Cannot count tokens: response has no generated token IDs.")
                total += len(token_ids)
        return total


def _engine_config(llm: Any, requested: dict) -> dict:
    """Keep requested and effective vLLM settings distinct in run artifacts."""
    engine = getattr(llm, "llm_engine", None)
    config = getattr(engine, "vllm_config", None)
    fields = {
        "scheduler_config": ("enable_chunked_prefill", "max_num_seqs", "max_num_batched_tokens"),
        "cache_config": ("enable_prefix_caching", "gpu_memory_utilization"),
        "model_config": ("max_model_len",),
        "parallel_config": ("tensor_parallel_size",),
    }
    effective = {}
    for section, names in fields.items():
        values = getattr(config, section, None)
        if values is None:
            # Older engines expose these configs directly on LLMEngine.
            values = getattr(engine, section, None)
        for name in names:
            effective[name] = getattr(values, name, None)
    return {
        "requested": dict(requested),
        "effective": effective,
        "engine_class": None if engine is None else f"{type(engine).__module__}.{type(engine).__qualname__}",
    }


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_workload(prompts: List[str], num_samples: int, batch_size: int, max_tokens: int) -> None:
    for name, value in (("num_samples", num_samples), ("batch_size", batch_size), ("max_tokens", max_tokens)):
        _positive_int(name, value)
    if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must contain nonempty strings")
