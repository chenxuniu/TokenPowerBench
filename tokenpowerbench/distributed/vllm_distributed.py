"""Distributed vLLM batches with explicit ownership of Ray actors and resources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import importlib
import ipaddress
import math
from numbers import Real
import socket
import sys
import time
from typing import Any
from urllib.parse import urlsplit

from .ray_cluster import RayClusterConfig
from .predictor import VLLMPredictor

_MIN_RAY_VERSION = "2.43.0"


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(name, value, minimum=0.0, *, inclusive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (value == minimum and not inclusive):
        relation = "at least" if inclusive else "greater than"
        raise ValueError(f"{name} must be {relation} {minimum}")
    return float(value)


def _prompts(prompts):
    if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a sequence of nonempty strings")
    prompts = list(prompts)
    if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must contain nonempty strings")
    return prompts


def _same_gcs_endpoint(requested, actual):
    """Compare native endpoints without confusing client ports with GCS ports."""
    if not isinstance(actual, str) or "://" in requested or "://" in actual:
        return False
    try:
        endpoints = [urlsplit("//" + address) for address in (requested, actual)]
        if endpoints[0].port is None or endpoints[0].port != endpoints[1].port:
            return False
        hosts = [endpoint.hostname.rstrip(".").lower() for endpoint in endpoints]
    except (AttributeError, ValueError):
        return False
    if hosts[0] == hosts[1]:
        return True

    def addresses(host):
        try:
            values = [ipaddress.ip_address(host)]
        except ValueError:
            try:
                values = [ipaddress.ip_address(item[4][0]) for item in
                          socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)]
            except (OSError, ValueError):
                return set()
        return {str(value.ipv4_mapped or value) if isinstance(value, ipaddress.IPv6Address)
                else str(value) for value in values}

    return bool(addresses(hosts[0]) & addresses(hosts[1]))


class VLLMDistributedEngine:
    """Coordinate model replicas, each using TP * PP GPU workers.

    Call prepare() before starting external measurements to exclude Ray setup,
    model loading, and warmup. A prepared engine supports repeated workloads
    until close(). run_benchmark() also supports standalone use: it prepares and
    closes resources automatically if the caller has not already prepared them.

    Config includes model_path, tensor_parallel_size, pipeline_parallel_size,
    concurrency, batch_size, max_tokens, temperature, top_p, seed, model_kwargs,
    placement_timeout_s, startup_timeout_s, inference_timeout_s, and
    shutdown_timeout_s. Placement uses PACK and reserves exactly one GPU/CPU
    bundle per model worker. Timing is observed on the driver; no cluster power
    or per-request prefill/decode attribution is provided.
    """

    def __init__(self, cluster: RayClusterConfig, config: Mapping[str, Any]) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        self.cluster = cluster
        self.config = copy.deepcopy(dict(config))
        self.model_path = self.config.get("model_path")
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("model_path must be a nonempty string")
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "concurrency", "batch_size"):
            value = _positive_integer(name, self.config.get(name, 1))
            self.config[name] = value
            setattr(self, name, value)
        _positive_integer("max_tokens", self.config.setdefault("max_tokens", 512))
        self.config["temperature"] = _finite_number("temperature", self.config.get("temperature", 0.0), inclusive=True)
        self.config["top_p"] = _finite_number("top_p", self.config.get("top_p", 1.0))
        if self.config["top_p"] > 1:
            raise ValueError("top_p must be at most 1")
        seed = self.config.setdefault("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        model_kwargs = self.config.setdefault("model_kwargs", {})
        if not isinstance(model_kwargs, dict):
            raise TypeError("model_kwargs must be a dict")
        if {"model", "tensor_parallel_size", "pipeline_parallel_size", "distributed_executor_backend"}.intersection(model_kwargs):
            raise ValueError("model_kwargs cannot override model, parallelism, or Ray executor settings")
        for name, default in (("placement_timeout_s", 120), ("startup_timeout_s", 900),
                              ("inference_timeout_s", 3600), ("shutdown_timeout_s", 30)):
            value = _finite_number(name, self.config.get(name, default))
            self.config[name] = value
            setattr(self, name, value)
        self.sampling_params = {"temperature": self.config["temperature"], "top_p": self.config["top_p"],
                                "max_tokens": self.config["max_tokens"], "seed": seed, "n": 1}
        self.verbose = bool(self.config.get("verbose", False))
        self._ray = None
        self._owns_ray = False
        self._actors = []
        self._groups = []
        self._prepared = False
        self._running = False
        self._preparation_info = None
        self.cleanup_errors = []

    def _init_ray(self):
        """Connect and verify that whole GPU/CPU bundles can fit on live nodes."""
        from packaging.version import Version

        ray = importlib.import_module("ray")
        self._ray = ray
        if Version(ray.__version__) < Version(_MIN_RAY_VERSION):
            raise RuntimeError(f"Ray >= {_MIN_RAY_VERSION} is required; found {ray.__version__}")
        requested_address = self.cluster.ray_init_address
        already_initialized = ray.is_initialized()
        if already_initialized and requested_address == "local":
            raise RuntimeError("Ray is already initialized; address='local' requires a new instance. "
                               "Use address='auto' to reuse the caller's connection.")
        if not already_initialized:
            self._owns_ray = True
            ray.init(address=requested_address, ignore_reinit_error=True)
        try:
            actual_address = ray.get_runtime_context().gcs_address
        except Exception:
            actual_address = None
        if not isinstance(actual_address, str) or not actual_address:
            actual_address = None
        if (already_initialized and requested_address != "auto"
                and not _same_gcs_endpoint(requested_address, actual_address)):
            raise RuntimeError(
                f"Cannot verify that the existing Ray connection ({actual_address or 'GCS address unavailable'}) "
                f"matches requested address {requested_address!r}. Use address='auto' to intentionally "
                "reuse it, or disconnect it before requesting another cluster."
            )
        alive = [node for node in ray.nodes() if node.get("Alive", False)]
        resources = [node.get("Resources", {}) for node in alive]
        slots = sum(min(int(resource.get("GPU", 0)), int(resource.get("CPU", 0))) for resource in resources)
        required = self.tensor_parallel_size * self.pipeline_parallel_size * self.concurrency
        if slots < required:
            raise RuntimeError(f"Insufficient live-node capacity: need {required} bundles with one GPU and one CPU; have {slots}")
        return {"alive_nodes": len(alive), "schedulable_gpu_cpu_bundles": slots, "required_bundles": required,
                "ray_version": ray.__version__, "requested_address": requested_address,
                "gcs_address": actual_address}

    def prepare(self) -> dict:
        """Reserve resources, load every replica, and warm all actors before timing."""
        if self._prepared:
            return copy.deepcopy(self._preparation_info)
        if self._actors or self._groups:
            raise RuntimeError("Previous actor/placement-group cleanup is incomplete; call close() before preparing")
        started = time.perf_counter()
        self.cleanup_errors = []
        try:
            cluster_info = self._init_ray()
            ray = self._ray
            strategy_type = importlib.import_module("ray.util.scheduling_strategies").PlacementGroupSchedulingStrategy
            width = self.tensor_parallel_size * self.pipeline_parallel_size
            for _ in range(self.concurrency):
                group = ray.util.placement_group([{"GPU": 1, "CPU": 1} for _ in range(width)], strategy="PACK")
                self._groups.append(group)
            ray.get([group.ready() for group in self._groups], timeout=self.placement_timeout_s)
            deadline = time.perf_counter() + self.startup_timeout_s
            actor_type = ray.remote(VLLMPredictor)
            for group in self._groups:
                # The CPU-only coordinator shares a GPU-bearing node with vLLM's
                # rank-0 worker. The reserved GPU remains available to that child.
                strategy = strategy_type(placement_group=group, placement_group_bundle_index=0,
                                         placement_group_capture_child_tasks=True)
                actor = actor_type.options(num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0,
                                           scheduling_strategy=strategy).remote(
                    model_path=self.model_path, tensor_parallel_size=self.tensor_parallel_size,
                    pipeline_parallel_size=self.pipeline_parallel_size, sampling_params=self.sampling_params,
                    verbose=self.verbose, model_kwargs=self.config["model_kwargs"])
                self._actors.append(actor)
            workers = ray.get([actor.ready.remote() for actor in self._actors], timeout=self._remaining(deadline, "startup"))
            warmups = ray.get([actor.warmup.remote() for actor in self._actors], timeout=self._remaining(deadline, "warmup"))
            self._preparation_info = {"duration_s": time.perf_counter() - started, "cluster": cluster_info,
                                      "workers": workers, "warmups": warmups, "configuration": copy.deepcopy(self.config)}
            self._prepared = True
            return copy.deepcopy(self._preparation_info)
        except BaseException as error:
            self._close_preserving(error)
            raise

    @staticmethod
    def _remaining(deadline, operation):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"Distributed {operation} timed out")
        return remaining

    def run_benchmark(self, prompts: Sequence[str]) -> dict:
        """Dispatch true batches and return exact token counts and a driver window.

        Errors propagate. A failed run closes its owned actors and placement
        groups even if prepare() was called explicitly. Successful explicit
        preparation remains available for repeated workloads until close().
        """
        prompts = _prompts(prompts)
        if self._running:
            raise RuntimeError("This engine is already running a benchmark")
        automatic = not self._prepared
        self._running = True
        try:
            if automatic:
                self.prepare()
            batches = [{"request_id": list(range(offset, min(offset + self.batch_size, len(prompts)))),
                        "text": prompts[offset:offset + self.batch_size]}
                       for offset in range(0, len(prompts), self.batch_size)]
            raw_batches, start, end = self._dispatch(batches)
            rows = self._flatten_batches(raw_batches)
            return self._process_results(rows, start, end, prompts)
        finally:
            primary = sys.exc_info()[1]
            self._running = False
            if automatic or primary is not None:
                self._close_preserving(primary)

    def _dispatch(self, batches):
        ray = self._ray
        batches = iter(batches)
        pending = {}
        collected = []
        start = time.perf_counter()
        deadline = start + self.inference_timeout_s
        for actor in self._actors:
            batch = next(batches, None)
            if batch is None:
                break
            pending[actor.__call__.remote(batch)] = actor
        while pending:
            ready, _ = ray.wait(list(pending), num_returns=1, timeout=self._remaining(deadline, "inference"))
            if not ready:
                raise TimeoutError("Distributed inference timed out")
            reference = ready[0]
            actor = pending.pop(reference)
            collected.append(ray.get(reference, timeout=self._remaining(deadline, "inference")))
            batch = next(batches, None)
            if batch is not None:
                pending[actor.__call__.remote(batch)] = actor
        end = time.perf_counter()
        return collected, start, end

    @staticmethod
    def _flatten_batches(batches):
        rows = []
        required = ("request_id", "prompt", "generated_text", "input_tokens", "output_tokens",
                    "batch_id", "batch_duration_s", "batch_size")
        for batch in batches:
            if not isinstance(batch, Mapping) or any(key not in batch for key in required):
                raise RuntimeError("Worker output is missing required batch columns")
            columns = {}
            for key in required:
                value = batch[key]
                if isinstance(value, (str, bytes)):
                    raise RuntimeError(f"Worker column {key} must contain one value per request")
                try:
                    columns[key] = list(value)
                except TypeError as error:
                    raise RuntimeError(f"Worker column {key} is not a sequence") from error
            count = len(columns["request_id"])
            if not count or any(len(value) != count for value in columns.values()):
                raise RuntimeError("Worker output columns have inconsistent lengths")
            rows.extend({key: values[index] for key, values in columns.items()} for index in range(count))
        return rows

    def _process_results(self, outputs, t0, t1, prompts):
        duration = _finite_number("measurement duration", t1 - t0)
        by_id = {}
        batches = {}
        for row in outputs:
            request_id = row.get("request_id")
            if isinstance(request_id, bool) or not isinstance(request_id, int) or not 0 <= request_id < len(prompts):
                raise RuntimeError("Worker returned an invalid request_id")
            if request_id in by_id:
                raise RuntimeError(f"Worker returned duplicate request_id {request_id}")
            if row.get("prompt") != prompts[request_id] or not isinstance(row.get("generated_text"), str):
                raise RuntimeError("Worker output does not match its submitted request")
            tokens = _positive_integer("output_tokens", row.get("output_tokens"))
            input_tokens = row.get("input_tokens")
            if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0:
                raise RuntimeError("Worker returned an invalid input_tokens count")
            batch_id = row.get("batch_id")
            if not isinstance(batch_id, str) or not batch_id:
                raise RuntimeError("Worker returned an invalid batch_id")
            batch_size = _positive_integer("worker batch_size", row.get("batch_size"))
            batch_duration = _finite_number("worker batch_duration_s", row.get("batch_duration_s"))
            metadata = batches.setdefault(batch_id, {"duration_s": batch_duration, "size": batch_size, "request_ids": []})
            if metadata["duration_s"] != batch_duration or metadata["size"] != batch_size:
                raise RuntimeError("Worker returned inconsistent metadata for one batch")
            metadata["request_ids"].append(request_id)
            by_id[request_id] = {"id": request_id + 1, "request_id": request_id, "prompt": prompts[request_id],
                                 "response": row["generated_text"], "input_tokens": input_tokens,
                                 "tokens_generated": tokens, "processing_time_s": None, "tokens_per_second": None,
                                 "batch_id": batch_id, "batch_duration_s": batch_duration, "batch_size": batch_size}
        if set(by_id) != set(range(len(prompts))):
            raise RuntimeError("Distributed inference returned incomplete request coverage")
        expected_groups = {frozenset(range(offset, min(offset + self.batch_size, len(prompts))))
                           for offset in range(0, len(prompts), self.batch_size)}
        for batch in batches.values():
            request_ids = frozenset(batch["request_ids"])
            if len(request_ids) != batch["size"] or request_ids not in expected_groups:
                raise RuntimeError("Worker batch metadata does not match the dispatched workload")
        total_tokens = sum(row["tokens_generated"] for row in by_id.values())
        batch_time = math.fsum(batch["duration_s"] for batch in batches.values())
        metrics = {"total_prompts": len(prompts), "total_time_s": duration, "total_tokens": total_tokens,
                   "throughput_tokens_per_s": total_tokens / duration, "success_rate": 1.0,
                   "avg_processing_time_s": None, "total_batches": len(batches),
                   "sum_worker_batch_time_s": batch_time, "avg_batch_time_s": batch_time / len(batches),
                   "tensor_parallel_size": self.tensor_parallel_size, "pipeline_parallel_size": self.pipeline_parallel_size,
                   "concurrency": self.concurrency, "batch_size": self.batch_size,
                   "token_count_source": "generated_token_ids", "timing_scope": "driver_dispatch_and_collection"}
        return {"results": [by_id[index] for index in range(len(prompts))], "performance_metrics": metrics,
                "configuration": copy.deepcopy(self.config), "preparation": copy.deepcopy(self._preparation_info),
                "measurement_window": {"start_s": t0, "end_s": t1, "clock": "time.perf_counter",
                                       "scope": "driver_dispatch_and_collection"},
                "phase_status": "unavailable_for_distributed_batches"}

    def _close_preserving(self, primary):
        try:
            self.close()
        except BaseException as cleanup_error:
            if primary is None:
                raise
            add_note = getattr(primary, "add_note", None)
            if callable(add_note):
                add_note(f"Distributed cleanup also failed: {cleanup_error}")

    def close(self) -> None:
        """Release only this engine's actors, placement groups, and Ray connection."""
        if self._ray is None:
            return
        ray = self._ray
        deadline = time.perf_counter() + self.shutdown_timeout_s
        failures = []
        references = []
        for actor in self._actors:
            try:
                references.append(actor.close.remote())
            except BaseException as error:
                failures.append(error)
        if references:
            try:
                ray.get(references, timeout=self._remaining(deadline, "shutdown"))
            except BaseException as error:
                failures.append(error)
        for actor in list(self._actors):
            try:
                ray.kill(actor, no_restart=True)
                self._actors.remove(actor)
            except BaseException as error:
                failures.append(error)
        for group in self._groups:
            try:
                ray.util.remove_placement_group(group)
            except BaseException as error:
                failures.append(error)
        while self._groups:
            try:
                self._groups = [group for group in self._groups
                                if ray.util.placement_group_table(group).get("state") != "REMOVED"]
                if not self._groups:
                    break
                time.sleep(min(0.1, self._remaining(deadline, "placement-group removal")))
            except BaseException as error:
                failures.append(error)
                break
        if self._owns_ray:
            try:
                ray.shutdown()
                self._owns_ray = False
                self._actors.clear()
                self._groups.clear()
            except BaseException as error:
                failures.append(error)
        self._prepared = False
        self._preparation_info = None
        if not self._actors and not self._groups and not self._owns_ray:
            self._ray = None
        self.cleanup_errors.extend(f"{type(error).__name__}: {error}" for error in failures)
        if failures:
            raise failures[0]
