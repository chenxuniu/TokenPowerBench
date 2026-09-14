"""Public single-node benchmarking API with explicit sensor scope."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import math
from numbers import Real
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid

from .energy import create_monitor
from .engines import VLLMEngine
from .runtime import runtime_identity


@dataclass(frozen=True)
class BenchmarkResult:
    """Completed benchmark results and the directory containing its artifacts."""

    output_dir: Path
    results: dict

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation, including the artifact path."""
        return {"output_dir": str(self.output_dir), "results": self.results}


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _path(name, value):
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{name} must be a path string or os.PathLike")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} must not be empty")
    return Path(value).expanduser()


def _prompt_list(prompts):
    if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a sequence of strings, not a single string")
    result = list(prompts)
    if not result or any(not isinstance(prompt, str) or not prompt.strip() for prompt in result):
        raise ValueError("prompts must contain nonempty strings")
    return result


def _validate_options(options):
    """Normalize API and CLI inputs before initializing hardware or writing files."""
    options = dict(options)
    model = options["model"]
    if not isinstance(model, str):
        raise TypeError("model must be a string")
    if not model.strip():
        raise ValueError("model must not be empty")
    sources = (options["prompts"], options["prompts_file"], options["dataset"])
    if sum(source is not None for source in sources) > 1:
        raise ValueError("Provide only one of prompts, prompts_file, or dataset")
    if options["prompts"] is not None:
        options["prompts"] = _prompt_list(options["prompts"])
    if options["prompts_file"] is not None:
        options["prompts_file"] = _path("prompts_file", options["prompts_file"])
    if options["dataset"] is not None:
        if not isinstance(options["dataset"], str):
            raise TypeError("dataset must be a string")
        options["dataset"] = options["dataset"].strip().lower()
        if options["dataset"] not in ("alpaca", "dolly", "longbench", "humaneval"):
            raise ValueError("dataset must be alpaca, dolly, longbench, or humaneval")
    if not any(source is not None for source in sources):
        options["dataset"] = "alpaca"
    sizes = options["batch_sizes"]
    if isinstance(sizes, (str, bytes)) or not isinstance(sizes, Sequence):
        raise TypeError("batch_sizes must be a sequence of positive integers")
    options["batch_sizes"] = [_integer("batch_sizes", size) for size in sizes]
    if not options["batch_sizes"]:
        raise ValueError("batch_sizes must not be empty")
    if len(set(options["batch_sizes"])) != len(options["batch_sizes"]):
        raise ValueError("batch_sizes must not contain duplicates")
    if not isinstance(options["phase_profiling"], bool):
        raise TypeError("phase_profiling must be a bool")
    if options["phase_profiling"] and options["batch_sizes"] != [1]:
        raise ValueError("phase_profiling requires batch_sizes=(1,): concurrent requests mix prefill and decode power")
    for name in ("num_samples", "max_model_len", "tensor_parallel_size"):
        if options[name] is not None:
            _integer(name, options[name])
    _integer("output_tokens", options["output_tokens"])
    _integer("seed", options["seed"], minimum=0)
    _integer("min_words", options["min_words"], minimum=0)
    _integer("max_words", options["max_words"])
    if options["min_words"] > options["max_words"]:
        raise ValueError("min_words must not exceed max_words")
    options["temperature"] = _number("temperature", options["temperature"])
    if options["temperature"] < 0:
        raise ValueError("temperature must be nonnegative")
    options["gpu_memory_utilization"] = _number("gpu_memory_utilization", options["gpu_memory_utilization"])
    if not 0 < options["gpu_memory_utilization"] <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if options["monitor"] not in ("auto", "gpu_only", "full_node"):
        raise ValueError("monitor must be auto, gpu_only, or full_node")
    options["output_dir"] = _path("output_dir", options["output_dir"])
    return options


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _load_prompts(options):
    prompts = options["prompts"]
    if options["prompts_file"] is not None:
        prompts = _prompt_list(json.loads(options["prompts_file"].read_text()))
    if prompts is None:
        from .data import DatasetLoader
        prompts = DatasetLoader(seed=options["seed"]).load(
            options["dataset"], num_samples=options["num_samples"] or 1000,
            min_words=options["min_words"], max_words=options["max_words"])
        prompts = _prompt_list(prompts)
        count = options["num_samples"] or 1000
    else:
        count = options["num_samples"] or len(prompts)
    # Repetition is ordered, and the expanded list is saved as the exact workload.
    return (prompts * math.ceil(count / len(prompts)))[:count]


def resolve_model_path(model):
    """Resolve local HF cache layouts without guessing a revision from hash order."""
    path = Path(model).expanduser()
    if not path.is_dir():
        return model  # Hugging Face model ID; resolution is handled by vLLM.
    if (path / "config.json").is_file():
        return str(path.resolve())
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        ref = path / "refs" / "main"
        if ref.is_file():
            revision = ref.read_text().strip()
            if revision and "/" not in revision and "\\" not in revision:
                candidate = snapshots / revision
                if (candidate / "config.json").is_file():
                    return str(candidate.resolve())
        candidates = [p for p in snapshots.iterdir() if (p / "config.json").is_file()]
        if len(candidates) == 1:
            return str(candidates[0].resolve())
        raise ValueError("Model cache has no unambiguous revision; pass an explicit snapshots/<revision> directory")
    return str(path.resolve())


def environment():
    """Record package/runtime versions and only this package's own source checkout."""
    from . import __version__
    versions = {"tokenpowerbench": __version__}
    for name in ("torch", "vllm", "nvidia-ml-py", "datasets", "transformers"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parent.parent
    # A wheel can live under another Git checkout. Do not search parent folders
    # or record that unrelated repository as the benchmark's source revision.
    project = root / "pyproject.toml"
    own_checkout = False
    if (root / ".git").exists() and project.is_file():
        try:
            project_section = project.read_text().split("[project]", 1)[1].split("\n[", 1)[0]
            own_checkout = re.search(r'^name\s*=\s*["\']tokenpowerbench["\']\s*$', project_section, re.M) is not None
        except (OSError, IndexError):
            pass
    def git(*args):
        if not own_checkout:
            return None
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                           stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"runtime": runtime_identity(), "python": sys.version, "platform": platform.platform(), "packages": versions,
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "clock": "time.perf_counter (host monotonic seconds)",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def metric_dict(metrics):
    result = asdict(metrics)
    for key in ("total_energy_j", "components_energy_j", "energy_per_token_j", "gpu_mj_per_token", "total_mj_per_token"):
        result[key] = getattr(metrics, key)
    return result


def phase_results(monitor, events):
    results = []
    for event in events:
        record = dict(event)
        for name, start, end, tokens in (
            ("prefill_proxy", event["prefill_start_s"], event["first_token_s"], 0),
            ("decode", event["first_token_s"], event["finished_s"], max(0, event["output_tokens"] - 1)),
        ):
            if end == start:
                record[name + "_energy"] = {"duration": 0.0, "status": "empty_window", "gpu_energy_j": None,
                                            "cpu_energy_j": None, "system_energy_j": None}
            else:
                record[name + "_energy"] = metric_dict(monitor.compute_metrics(
                    end - start, tokens, 1, start_time=start, end_time=end))
        results.append(record)
    return results


def _cleanup(actions, diagnostics):
    """Attempt all cleanup actions while preserving an active primary error."""
    primary_error = sys.exc_info()[1]
    failures = []
    for name, action in actions:
        try:
            action()
        except BaseException as cleanup_error:
            failures.append(cleanup_error)
            diagnostic = f"{name}: {type(cleanup_error).__name__}: {cleanup_error}"
            diagnostics.append(diagnostic)
            add_note = getattr(primary_error, "add_note", None)
            if callable(add_note):
                add_note(f"Cleanup also failed: {diagnostic}")
    if failures and primary_error is None:
        raise failures[0]


def check_environment(monitor="auto", device_indices=None, device_uuids=None) -> dict:
    """Probe runtime identity and actual sensor access without loading a model.

    Explicit device indices or UUIDs can be used without importing torch.
    Default device discovery follows CUDA visibility. The monitor is closed
    before returning; initialization and permission errors propagate normally.
    """
    identity = runtime_identity()
    sensor = create_monitor(monitor, device_indices=device_indices, device_uuids=device_uuids)
    try:
        return {"runtime": identity, "sensors": sensor.capabilities}
    finally:
        _cleanup([("monitor.close", sensor.close)], [])


def benchmark(
    *,
    model: str,
    prompts: Sequence[str] | None = None,
    prompts_file: str | os.PathLike | None = None,
    dataset: str | None = None,
    num_samples: int | None = None,
    batch_sizes: Sequence[int] = (1,),
    output_tokens: int = 128,
    phase_profiling: bool = False,
    monitor: str = "auto",
    output_dir: str | os.PathLike = "results",
    seed: int = 42,
    temperature: float = 0,
    max_model_len: int | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float = 0.9,
    min_words: int = 2,
    max_words: int = 300,
) -> BenchmarkResult:
    """Run a single-node vLLM benchmark and save reproducible measurement artifacts.

    Select one prompt source. With supplied prompts or a prompt file, omitted
    num_samples uses its length. With no source, use 1000 Alpaca requests.
    Phase profiling requires serial batch size 1. Numeric phase energy remains
    subject to each sensor's sampling-resolution policy.

    Errors propagate to the caller. Once an artifact directory exists, failures
    and KeyboardInterrupt leave a terminal status and available raw samples.
    The engine and monitor close before this function returns or raises.
    """
    options = _validate_options(locals())
    sensor = None
    engine = None
    run_dir = None
    cleanup_errors = []
    try:
        try:
            sensor = create_monitor(options["monitor"])
            if options["tensor_parallel_size"] is not None:
                devices = sensor.capabilities.get("gpu", {}).get("devices", [])
                if options["tensor_parallel_size"] != len(devices):
                    raise ValueError("tensor_parallel_size must equal the number of CUDA-visible GPUs; set CUDA_VISIBLE_DEVICES to select exactly those devices")
            requests = _load_prompts(options)
            run_dir = (options["output_dir"] / ("local_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])).resolve()
            run_dir.mkdir(parents=True, exist_ok=False)
            configuration = {key: str(value) if isinstance(value, Path) else value
                             for key, value in options.items() if key != "prompts"}
            configuration.update(engine="vllm", check_monitor=False, num_samples=len(requests))
            configuration["prompt_sha256"] = hashlib.sha256(json.dumps(requests, ensure_ascii=False).encode()).hexdigest()
            configuration["prompt_policy"] = "raw text, no implicit chat template; repeat ordered prompts to num_samples"
            write_json(run_dir / "status.json", {"status": "running"})
            write_json(run_dir / "config.json", configuration)
            write_json(run_dir / "prompts.json", requests)
            write_json(run_dir / "environment.json", environment())
            write_json(run_dir / "runtime.json", runtime_identity())
            write_json(run_dir / "capabilities.json", sensor.capabilities)
            engine = VLLMEngine()
            engine.setup_model(resolve_model_path(options["model"]), phase_profiling=options["phase_profiling"],
                               seed=options["seed"], temperature=options["temperature"],
                               max_model_len=options["max_model_len"], tensor_parallel_size=options["tensor_parallel_size"],
                               gpu_memory_utilization=options["gpu_memory_utilization"])
            write_json(run_dir / "engine_config.json", getattr(engine, "resolved_config", {}))
            if options["phase_profiling"]:
                engine.run_profiled_benchmark([requests[0]], 1, 1, min(20, options["output_tokens"]))
            else:
                engine.run_inference([requests[0]], batch_size=1, max_tokens=min(20, options["output_tokens"]))
            results = {}
            for batch in options["batch_sizes"]:
                events = []
                try:
                    sensor.start()
                    if options["phase_profiling"]:
                        outputs, start, end, events = engine.run_profiled_benchmark(requests, len(requests), batch, options["output_tokens"])
                    else:
                        outputs, start, end = engine.run_benchmark(requests, len(requests), batch, options["output_tokens"])
                finally:
                    _cleanup([
                        ("monitor.stop", sensor.stop),
                        ("save raw samples", lambda: write_json(run_dir / f"batch_{batch}_power_samples.json", sensor.samples)),
                    ], cleanup_errors)
                total_tokens = engine.estimate_tokens(outputs)
                if len(outputs) != len(requests) or total_tokens <= 0:
                    raise RuntimeError("Inference returned incomplete responses or no generated token IDs")
                metrics = sensor.compute_metrics(end - start, total_tokens, len(outputs), start_time=start, end_time=end)
                result = {"batch_size": batch, "start_s": start, "end_s": end,
                          "total_output_tokens": total_tokens, "num_responses": len(outputs),
                          "output_tokens_per_s": total_tokens / (end - start),
                          "energy": metric_dict(metrics), "phases": phase_results(sensor, events),
                          "phase_status": "serial_host_boundaries" if events else "not_requested"}
                write_json(run_dir / f"batch_{batch}_result.json", result)
                results[f"batch_{batch}"] = result
                write_json(run_dir / "capabilities.json", sensor.capabilities)
            write_json(run_dir / "results.json", results)
            completed = BenchmarkResult(output_dir=run_dir, results=results)
        finally:
            _cleanup([(name, resource.close) for name, resource in
                      (("engine.close", engine), ("monitor.close", sensor)) if resource is not None], cleanup_errors)
        write_json(run_dir / "status.json", {"status": "completed"})
        return completed
    except KeyboardInterrupt:
        if run_dir is not None:
            status = {"status": "interrupted"}
            if cleanup_errors:
                status["cleanup_errors"] = cleanup_errors
            write_json(run_dir / "status.json", status)
        raise
    except Exception as exc:
        if run_dir is not None:
            status = {"status": "failed", "error": str(exc)}
            if cleanup_errors:
                status["cleanup_errors"] = cleanup_errors
            write_json(run_dir / "status.json", status)
        raise
