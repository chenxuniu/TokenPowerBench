#!/usr/bin/env python3
"""Multi-node vLLM inference with explicit cluster and driver measurement scopes."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import sys
import time
import uuid

from tokenpowerbench.api import _cleanup, environment, metric_dict, write_json
from tokenpowerbench.distributed import RayClusterConfig, VLLMDistributedEngine
from tokenpowerbench.energy import create_monitor
from tokenpowerbench.runtime import runtime_identity


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _csv(value, numeric=False):
    items = [part.strip() for part in value.split(",")]
    if not items or any(not part for part in items):
        raise ValueError("comma-separated lists must not contain empty entries")
    if numeric:
        items = [_positive_int(part) for part in items]
    if len(set(items)) != len(items):
        raise ValueError("comma-separated lists must not contain duplicates")
    return items


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage="%(prog)s --configure-cluster [--cluster-config PATH]\n       %(prog)s --models MODELS [options]",
    )
    parser.add_argument("--configure-cluster", action="store_true",
                        help="Open the terminal cluster setup wizard; no --models needed in setup mode")
    parser.add_argument("--cluster-config", type=Path,
                        help="Saved manual-IP or dynamic SLURM profile (setup output default: cluster.json)")
    parser.add_argument("--models", required=True, help="Comma-separated model paths relative to --model-dir")
    parser.add_argument("--model-dir", type=Path, default=Path("~/models"),
                        help="Model directory visible at the same absolute path on GPU nodes")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--datasets", help="Comma-separated datasets (default: alpaca)")
    inputs.add_argument("--prompts-file", type=Path, help="JSON array of raw prompts; no dataset download")
    parser.add_argument("--num-samples", type=_positive_int, help="Default: prompt file length or 1000 dataset requests")
    parser.add_argument("--min-words", "--min-length", type=int, default=5)
    parser.add_argument("--max-words", "--max-length", type=_positive_int, default=100)
    parser.add_argument("--tensor-parallel", default="8")
    parser.add_argument("--pipeline-parallel", default="2")
    parser.add_argument("--concurrency", default="1", help="Comma-separated numbers of independent model replicas")
    parser.add_argument("--batch-sizes", default="256", help="Prompts passed together to one replica's generate call")
    parser.add_argument("--max-tokens", type=_positive_int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-kwargs-file", type=Path, help="JSON object of explicit vLLM model settings")
    parser.add_argument("--max-model-len", type=_positive_int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--ray-head-address", help="Explicit Ray address; otherwise profile, environment, SLURM, then auto")
    parser.add_argument("--ray-head-port", type=int)
    for name, default in (("placement", 120), ("startup", 900), ("inference", 3600), ("shutdown", 30)):
        parser.add_argument(f"--{name}-timeout-s", type=float, default=default)
    parser.add_argument("--monitor", choices=("none", "auto", "gpu_only", "full_node"), default="none",
                        help="Optional driver-host diagnostics only; cluster energy is not collected")
    parser.add_argument("--driver-device-indices", help="Physical NVML GPU indices on the driver, e.g. 0,1")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.models = _csv(args.models)
        args.datasets = _csv(args.datasets) if args.datasets is not None else ([] if args.prompts_file else ["alpaca"])
        unknown = set(args.datasets) - {"alpaca", "dolly", "longbench", "humaneval"}
        if unknown:
            raise ValueError(f"Unknown datasets: {', '.join(sorted(unknown))}")
        for name in ("tensor_parallel", "pipeline_parallel", "concurrency", "batch_sizes"):
            setattr(args, name, _csv(getattr(args, name), numeric=True))
        if args.seed < 0 or args.min_words < 0 or args.min_words > args.max_words:
            raise ValueError("seed and word limits must be nonnegative and min-words <= max-words")
        if not math.isfinite(args.temperature) or args.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
            raise ValueError("top-p must be in (0, 1]")
        if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
            raise ValueError("gpu-memory-utilization must be in (0, 1]")
        for name in ("placement", "startup", "inference", "shutdown"):
            value = getattr(args, f"{name}_timeout_s")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}-timeout-s must be finite and positive")
        if args.ray_head_port is not None and not 1 <= args.ray_head_port <= 65535:
            raise ValueError("ray-head-port must be between 1 and 65535")
        if args.driver_device_indices is not None:
            args.driver_device_indices = [int(part.strip()) for part in args.driver_device_indices.split(",")]
            if (not args.driver_device_indices or min(args.driver_device_indices) < 0
                    or len(set(args.driver_device_indices)) != len(args.driver_device_indices)):
                raise ValueError("driver-device-indices must be distinct nonnegative integers")
            if args.monitor == "none":
                raise ValueError("driver-device-indices requires an enabled driver monitor")
        args.model_kwargs = {}
        if args.model_kwargs_file is not None:
            args.model_kwargs = json.loads(args.model_kwargs_file.read_text())
            if not isinstance(args.model_kwargs, dict):
                raise ValueError("model-kwargs-file must contain a JSON object")
            json.dumps(args.model_kwargs, allow_nan=False)
        for name in ("max_model_len", "gpu_memory_utilization"):
            if getattr(args, name) is not None:
                args.model_kwargs[name] = getattr(args, name)
        args.model_dir = args.model_dir.expanduser().resolve()
        args.output_dir = args.output_dir.expanduser().resolve()
        args.cluster_profile = None
        if args.cluster_config is not None:
            from tokenpowerbench.distributed.cluster_setup import load_profile
            args.cluster_config = args.cluster_config.expanduser().resolve()
            args.cluster_profile = load_profile(args.cluster_config)
    except (ValueError, TypeError, OSError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    return args


def _load_workloads(args):
    if args.prompts_file is not None:
        prompts = json.loads(args.prompts_file.expanduser().read_text())
        if (not isinstance(prompts, list) or not prompts
                or any(not isinstance(item, str) or not item.strip() for item in prompts)):
            raise ValueError("prompts-file must contain a nonempty JSON array of nonempty strings")
        count = args.num_samples or len(prompts)
        return {"prompts_file": (prompts * math.ceil(count / len(prompts)))[:count]}
    from tokenpowerbench.data import DatasetLoader
    workloads = {}
    for name in args.datasets:
        prompts = DatasetLoader(seed=args.seed).load(
            name, num_samples=args.num_samples or 1000, min_words=args.min_words, max_words=args.max_words)
        if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError(f"Dataset {name} returned no usable prompts")
        workloads[name] = prompts
    return workloads


def _unmeasured_cluster_energy():
    return {"energy_scope": "unmeasured_cluster", "cluster_energy_j": None,
            "cluster_energy_per_token_j": None, "total_energy_j": None,
            "total_mj_per_token": None,
            "reason": "Per-node collectors and verified allocation coverage are required for cluster energy."}


def _configuration(args, model, tp, pp, concurrency, batch):
    result = dict(model_path=str(args.model_dir / model), tensor_parallel_size=tp,
                  pipeline_parallel_size=pp, concurrency=concurrency, batch_size=batch,
                  max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
                  seed=args.seed, model_kwargs=args.model_kwargs, verbose=args.verbose)
    for name in ("placement", "startup", "inference", "shutdown"):
        result[f"{name}_timeout_s"] = getattr(args, f"{name}_timeout_s")
    return result


def _run_configuration(cluster, config, prompts, args, output):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running"})
    write_json(output / "config.json", config)
    write_json(output / "prompts.json", prompts)
    engine = sensor = None
    diagnostics = []
    try:
        try:
            engine = VLLMDistributedEngine(cluster, config)
            if args.monitor != "none":
                sensor = create_monitor(args.monitor, device_indices=args.driver_device_indices)
                write_json(output / "driver_capabilities.json", sensor.capabilities)
            workers = engine.prepare()
            write_json(output / "workers.json", workers)
            # Model initialization and every replica's warmup finish before sampling.
            try:
                if sensor is not None:
                    sensor.start()
                result = engine.run_benchmark(prompts)
            finally:
                if sensor is not None:
                    _cleanup([("monitor.stop", sensor.stop),
                              ("save driver samples", lambda: write_json(output / "driver_power_samples.json", sensor.samples))],
                             diagnostics)
            if not isinstance(result, dict):
                raise RuntimeError("Distributed inference did not return a result")
            perf = result["performance_metrics"]
            if perf["total_prompts"] != len(prompts) or perf["total_tokens"] <= 0:
                raise RuntimeError("Distributed inference returned incomplete responses or no generated tokens")
            window = result["measurement_window"]
            start, end = window["start_s"], window["end_s"]
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise RuntimeError("Distributed inference returned an invalid monotonic measurement window")
            result["energy_metrics"] = _unmeasured_cluster_energy()
            result["phase_status"] = "not_measured_distributed_requests_may_overlap"
            result["driver_diagnostics"] = {"monitor_mode": args.monitor,
                                             "monitoring_scope": "local_driver_host" if sensor else "not_collected",
                                             "energy": None}
            if sensor is not None:
                # No local token attribution is available for a distributed workload.
                energy = metric_dict(sensor.compute_metrics(end - start, 0, 0, start_time=start, end_time=end))
                for name in ("total_output_tokens", "num_responses", "energy_per_token_j",
                             "gpu_mj_per_token", "total_mj_per_token"):
                    energy.pop(name, None)
                result["driver_diagnostics"].update(
                    energy=energy,
                    scope_note="Driver-host readings are diagnostic only; remote worker energy and local token attribution are unknown.")
                write_json(output / "driver_capabilities.json", sensor.capabilities)
            write_json(output / "result.json", result)
        finally:
            _cleanup([(name, resource.close) for name, resource in (("engine.close", engine), ("monitor.close", sensor))
                      if resource is not None], diagnostics)
        write_json(output / "status.json", {"status": "completed"})
        return result
    except BaseException as exc:
        status = {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                  "error": f"{type(exc).__name__}: {exc}"}
        if diagnostics:
            status["cleanup_errors"] = diagnostics
        write_json(output / "status.json", status)
        raise


def _resolve_cluster(args):
    """Use explicit CLI endpoints, then profile settings, then environment discovery."""
    profile = args.cluster_profile
    if profile is None:
        return RayClusterConfig.resolve(args.ray_head_address, args.ray_head_port)
    if args.ray_head_address is not None:
        if args.ray_head_port is not None:
            return RayClusterConfig.resolve(args.ray_head_address, args.ray_head_port)
        # An embedded CLI port wins; otherwise the profile supplies the default.
        return RayClusterConfig(head_address=args.ray_head_address, head_port=profile["head_port"])
    from tokenpowerbench.distributed.cluster_setup import resolve_profile
    if args.ray_head_port is not None:
        profile = {**profile, "head_port": args.ray_head_port}
    return resolve_profile(profile)


def _run_suite(args):
    cluster = _resolve_cluster(args)
    print(f"Ray cluster address: {cluster.ray_init_address}")
    output = args.output_dir / ("multi_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    config["ray_address"] = cluster.ray_init_address
    write_json(output / "config.json", config)
    write_json(output / "runtime.json", runtime_identity())
    info = environment()
    for name in ("ray", "cupy-cuda12x", "cupy-cuda13x"):
        try:
            info["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            info["packages"][name] = None
    write_json(output / "environment.json", info)
    write_json(output / "status.json", {"status": "running"})
    results, failures = {}, []
    try:
        workloads = _load_workloads(args)
        write_json(output / "prompts.json", workloads)
        config["prompt_sha256"] = {key: hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()
                                    for key, value in workloads.items()}
        config["prompt_policy"] = "Raw text without an implicit template; prompt-file repetition is ordered."
        write_json(output / "config.json", config)
        index = 0
        for model in args.models:
            for dataset, prompts in workloads.items():
                for tp in args.tensor_parallel:
                    for pp in args.pipeline_parallel:
                        for concurrency in args.concurrency:
                            for batch in args.batch_sizes:
                                index += 1
                                key = f"config_{index:04d}_TP{tp}_PP{pp}_C{concurrency}_B{batch}"
                                configuration = _configuration(args, model, tp, pp, concurrency, batch)
                                print(f"{key}: {model}, {dataset}, {len(prompts)} prompts")
                                try:
                                    result = _run_configuration(cluster, configuration, prompts, args, output / key)
                                    result["workload"] = {"model": model, "dataset": dataset}
                                    results[key] = result
                                except Exception as exc:
                                    failures.append({"configuration": key, "model": model, "dataset": dataset,
                                                     "error": f"{type(exc).__name__}: {exc}"})
                                    print(f"{key} failed: {exc}", file=sys.stderr)
                                write_json(output / "results.json", results)
                                write_json(output / "failures.json", failures)
        write_json(output / "status.json", {"status": "partial_failure" if results and failures else
                                            "failed" if failures or not results else "completed",
                                            "completed_configurations": len(results), "failed_configurations": len(failures)})
    except BaseException as exc:
        write_json(output / "results.json", results)
        write_json(output / "failures.json", failures)
        write_json(output / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                            "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        print(f"Results and diagnostics: {output}")
    return 1 if failures or not results else 0


def run(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--configure-cluster" in arguments:
        parser = argparse.ArgumentParser(description="Configure Ray node addresses or dynamic SLURM discovery.")
        parser.add_argument("--configure-cluster", action="store_true", required=True)
        parser.add_argument("--cluster-config", type=Path, default=Path("cluster.json"),
                            help="New profile path (default: cluster.json); existing files are preserved")
        setup = parser.parse_args(arguments)
        from tokenpowerbench.distributed.cluster_setup import configure
        return configure(setup.cluster_config.expanduser().absolute())
    args = parse_args(argv)
    try:
        print(json.dumps({"runtime": runtime_identity()}, indent=2))
        return _run_suite(args)
    except KeyboardInterrupt:
        print("Distributed benchmark interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Distributed benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
