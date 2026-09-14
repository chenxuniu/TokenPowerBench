#!/usr/bin/env python3
"""Single-node vLLM benchmark with explicit sensor scope and optional phase timing."""

import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

from tokenpowerbench.energy import create_monitor
from tokenpowerbench.engines import VLLMEngine
from tokenpowerbench.runtime import runtime_identity


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="Local model directory or Hugging Face model ID")
    p.add_argument("--engine", default="vllm", choices=["vllm"])
    p.add_argument("--dataset", default="alpaca", choices=["alpaca", "dolly", "longbench", "humaneval"])
    p.add_argument("--prompts-file", type=Path, help="JSON array of prompt strings (no dataset download)")
    p.add_argument("--num-samples", type=positive_int, default=1000)
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--max-words", type=positive_int, default=300)
    p.add_argument("--batch-sizes", default="256")
    p.add_argument("--output-tokens", type=positive_int, default=500)
    p.add_argument("--max-model-len", type=positive_int, help="Explicit model context limit for reproducible allocation")
    p.add_argument("--tensor-parallel-size", type=positive_int, help="Must match visible GPUs so monitoring scope matches inference")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--phase-profiling", action="store_true", help="Serial first-token phase profiling; requires --batch-sizes 1")
    p.add_argument("--monitor", default="auto", choices=["auto", "gpu_only", "full_node"],
                   help="auto: probe permissions; gpu_only: NVML; full_node: require IPMI; RAPL CPU is optional")
    p.add_argument("--check-monitor", action="store_true", help="Print actual sensor access and exit without loading a model")
    p.add_argument("--output-dir", type=Path, default=Path("./results"))
    args = p.parse_args(argv)
    try:
        args.batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        if not args.batch_sizes or any(x <= 0 for x in args.batch_sizes):
            raise ValueError
    except ValueError:
        p.error("--batch-sizes must contain positive integers separated by commas")
    if args.phase_profiling and args.batch_sizes != [1]:
        p.error("--phase-profiling requires --batch-sizes 1: concurrent requests mix prefill and decode power")
    if args.min_words < 0 or args.min_words > args.max_words:
        p.error("require 0 <= --min-words <= --max-words")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        p.error("--temperature must be finite and nonnegative")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization <= 1:
        p.error("--gpu-memory-utilization must be finite and in (0, 1]")
    if not args.model and not args.check_monitor:
        p.error("--model is required unless --check-monitor is used")
    return args


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_prompts(args):
    if args.prompts_file:
        prompts = json.loads(args.prompts_file.read_text())
        if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("--prompts-file must contain a nonempty JSON array of nonempty strings")
    else:
        from tokenpowerbench.data import DatasetLoader
        prompts = DatasetLoader(seed=args.seed).load(args.dataset, num_samples=args.num_samples,
                                                    min_words=args.min_words, max_words=args.max_words)
    if not prompts:
        raise ValueError("No prompts remain after filtering")
    # Save the exact ordered requests; repetitions are explicit and reproducible.
    return (prompts * math.ceil(args.num_samples / len(prompts)))[:args.num_samples]


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
    versions = {}
    for name in ("tokenpowerbench", "torch", "vllm", "nvidia-ml-py", "datasets", "transformers"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parent
    def git(*args):
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


def run(argv=None):
    args = parse_args(argv)
    monitor = None
    run_dir = None
    try:
        # Probe before model allocation; the same GPU UUID selection is used by CUDA and NVML.
        print(json.dumps({"runtime": runtime_identity()}, indent=2))
        monitor = create_monitor(args.monitor)
        print(json.dumps({"sensors": monitor.capabilities}, indent=2))
        if args.check_monitor:
            return 0
        if args.tensor_parallel_size is not None:
            devices = monitor.capabilities.get("gpu", {}).get("devices", [])
            if args.tensor_parallel_size != len(devices):
                raise ValueError("--tensor-parallel-size must equal the number of CUDA-visible GPUs; set CUDA_VISIBLE_DEVICES to select exactly those devices")
        prompts = load_prompts(args)
        run_dir = args.output_dir / ("local_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        run_dir.mkdir(parents=True, exist_ok=False)
        configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        configuration["prompt_sha256"] = hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()
        configuration["prompt_policy"] = "raw text, no implicit chat template; repeat ordered prompts to num_samples"
        write_json(run_dir / "config.json", configuration)
        write_json(run_dir / "prompts.json", prompts)
        write_json(run_dir / "environment.json", environment())
        write_json(run_dir / "runtime.json", runtime_identity())
        write_json(run_dir / "capabilities.json", monitor.capabilities)
        write_json(run_dir / "status.json", {"status": "running"})
        engine = VLLMEngine()
        engine.setup_model(resolve_model_path(args.model), phase_profiling=args.phase_profiling,
                           seed=args.seed, temperature=args.temperature,
                           max_model_len=args.max_model_len, tensor_parallel_size=args.tensor_parallel_size,
                           gpu_memory_utilization=args.gpu_memory_utilization)
        write_json(run_dir / "engine_config.json", getattr(engine, "resolved_config", {}))
        if args.phase_profiling:
            engine.run_profiled_benchmark([prompts[0]], 1, 1, min(20, args.output_tokens))
        else:
            engine.run_inference([prompts[0]], batch_size=1, max_tokens=min(20, args.output_tokens))
        results = {}
        for batch in args.batch_sizes:
            events = []
            monitor.start()
            try:
                if args.phase_profiling:
                    outputs, start, end, events = engine.run_profiled_benchmark(prompts, len(prompts), batch, args.output_tokens)
                else:
                    outputs, start, end = engine.run_benchmark(prompts, len(prompts), batch, args.output_tokens)
            finally:
                try:
                    monitor.stop()
                finally:
                    write_json(run_dir / f"batch_{batch}_power_samples.json", monitor.samples)
            total_tokens = engine.estimate_tokens(outputs)
            if len(outputs) != len(prompts) or total_tokens <= 0:
                raise RuntimeError("Inference returned incomplete responses or no generated token IDs")
            metrics = monitor.compute_metrics(end - start, total_tokens, len(outputs), start_time=start, end_time=end)
            result = {"batch_size": batch, "start_s": start, "end_s": end,
                      "total_output_tokens": total_tokens, "num_responses": len(outputs),
                      "output_tokens_per_s": total_tokens / (end - start),
                      "energy": metric_dict(metrics), "phases": phase_results(monitor, events),
                      "phase_status": "serial_host_boundaries" if events else "not_requested"}
            write_json(run_dir / f"batch_{batch}_result.json", result)
            results[f"batch_{batch}"] = result
            write_json(run_dir / "capabilities.json", monitor.capabilities)
            print(metrics.summary())
        write_json(run_dir / "results.json", results)
        write_json(run_dir / "status.json", {"status": "completed"})
        print(f"Results and raw samples: {run_dir.resolve()}")
        return 0
    except KeyboardInterrupt:
        if run_dir is not None:
            write_json(run_dir / "status.json", {"status": "interrupted"})
        print("Benchmark interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if run_dir is not None:
            write_json(run_dir / "status.json", {"status": "failed", "error": str(exc)})
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if monitor is not None:
            monitor.close()


if __name__ == "__main__":
    raise SystemExit(run())
