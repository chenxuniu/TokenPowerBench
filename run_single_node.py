#!/usr/bin/env python3
"""
Single-node LLM energy benchmark.

Usage
-----
    python run_single_node.py \
        --model /path/to/model \
        --engine vllm \
        --batch-sizes 128,256 \
        --num-samples 5000 \
        --output-tokens 500 \
        --monitor auto        # "auto" | "gpu_only" | "full_node"

Monitor modes
-------------
  auto       Automatically use full_node if RAPL is accessible, else gpu_only.
  gpu_only   GPU power via NVML only. No root required.
             Suitable for any data-center / cloud user.
  full_node  GPU + CPU (Intel RAPL) + total node (IPMI).
             Requires root or appropriate sysfs / BMC permissions.
"""

import argparse
import json
import os
import time
from pathlib import Path

from tokenpowerbench.data import DatasetLoader
from tokenpowerbench.energy import create_monitor
from tokenpowerbench.engines import VLLMEngine


def parse_args():
    p = argparse.ArgumentParser(description="Single-node LLM energy benchmark")

    p.add_argument("--model", required=True,
                   help="Path to the model directory")
    p.add_argument("--engine", default="vllm", choices=["vllm"],
                   help="Inference engine (default: vllm)")

    # Dataset
    p.add_argument("--dataset", default="alpaca",
                   choices=["alpaca", "dolly", "longbench", "humaneval"],
                   help="Dataset to use for prompts")
    p.add_argument("--num-samples", type=int, default=5000,
                   help="Number of inference requests")
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--max-words", type=int, default=300)

    # Inference
    p.add_argument("--batch-sizes", default="256",
                   help="Comma-separated list of batch sizes, e.g. '128,256,512'")
    p.add_argument("--output-tokens", type=int, default=500,
                   help="Max tokens to generate per prompt")

    # Energy monitoring
    p.add_argument("--monitor", default="auto",
                   choices=["auto", "gpu_only", "full_node"],
                   help="Energy monitor mode (default: auto)")

    # Output
    p.add_argument("--output-dir", default="./results",
                   help="Directory for result JSON files")

    return p.parse_args()


def run():
    args = parse_args()
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    loader = DatasetLoader()
    prompts = loader.load(
        args.dataset,
        num_samples=args.num_samples,
        min_words=args.min_words,
        max_words=args.max_words,
    )
    if not prompts:
        print("No prompts loaded. Exiting.")
        return

    # Setup engine
    engine = VLLMEngine()
    if not engine.available:
        print("vLLM is not installed. Run: pip install vllm")
        return

    model = engine.setup_model(args.model)
    if model is None:
        print(f"Failed to load model from {args.model}")
        return

    # Warmup
    print("Running warmup pass…")
    engine.run_inference([prompts[0]], batch_size=1, max_tokens=20)

    all_results = {}

    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch size: {batch_size}  |  monitor: {args.monitor}")
        print(f"{'='*60}")

        monitor = create_monitor(args.monitor)
        monitor.start()

        outputs, t0, t1 = engine.run_benchmark(
            prompts, args.num_samples, batch_size, args.output_tokens
        )

        # Extra idle time so monitoring captures the tail
        time.sleep(2.0)
        monitor.stop()

        duration = t1 - t0
        total_tokens = engine.estimate_tokens(outputs)
        metrics = monitor.compute_metrics(duration, total_tokens, len(outputs))

        print(metrics.summary())

        result = {
            "model": args.model,
            "engine": args.engine,
            "dataset": args.dataset,
            "batch_size": batch_size,
            "num_samples": args.num_samples,
            "output_tokens": args.output_tokens,
            "monitor_mode": args.monitor,
            # Timing
            "duration_s": duration,
            "total_output_tokens": total_tokens,
            "num_responses": len(outputs),
            # GPU (always present)
            "gpu_avg_power_w": metrics.gpu_avg_power_w,
            "gpu_energy_j": metrics.gpu_energy_j,
            "gpu_mj_per_token": metrics.gpu_mj_per_token,
            "per_gpu_power_w": metrics.per_gpu_power_w,
            # CPU / DRAM (full_node only)
            "cpu_avg_power_w": metrics.cpu_avg_power_w,
            "cpu_energy_j": metrics.cpu_energy_j,
            "dram_avg_power_w": metrics.dram_avg_power_w,
            "dram_energy_j": metrics.dram_energy_j,
            # System total via IPMI (full_node only)
            "system_avg_power_w": metrics.system_avg_power_w,
            "system_energy_j": metrics.system_energy_j,
            # Combined
            "total_energy_j": metrics.total_energy_j,
            "total_mj_per_token": metrics.total_mj_per_token,
        }
        all_results[f"batch_{batch_size}"] = result

    # Save results
    model_slug = os.path.basename(args.model.rstrip("/"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"{model_slug}_{args.engine}_b{'_'.join(str(b) for b in batch_sizes)}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    run()
