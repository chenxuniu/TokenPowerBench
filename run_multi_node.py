#!/usr/bin/env python3
"""
Multi-node distributed LLM energy benchmark (vLLM + Ray).

Usage
-----
    # On the Ray head node (after the cluster is up):
    python run_multi_node.py \
        --models Llama-3.1-405B,Llama-3.1-70B \
        --model-dir ~/models \
        --datasets alpaca \
        --batch-sizes 256 \
        --tensor-parallel 8 \
        --pipeline-parallel 2 \
        --concurrency 1 \
        --num-samples 1000 \
        --max-tokens 512 \
        --monitor auto

Ray cluster setup
-----------------
The head node IP is read from:
  1. --ray-head-address argument  (explicit)
  2. RAY_HEAD_ADDRESS env var
  3. SLURM_JOB_NODELIST            (automatic in SLURM jobs — recommended)
  4. "auto"                        (Ray service discovery fallback)

See scripts/slurm_head.sh for a complete SLURM example that sets
RAY_HEAD_ADDRESS automatically — no hardcoded IPs needed.
"""

import argparse
import json
import os
import time
from pathlib import Path

from tokenpowerbench.data import DatasetLoader
from tokenpowerbench.distributed import VLLMDistributedEngine
from tokenpowerbench.distributed.ray_cluster import RayClusterConfig
from tokenpowerbench.energy import create_monitor


def parse_args():
    p = argparse.ArgumentParser(description="Multi-node LLM energy benchmark")

    # Models
    p.add_argument("--models", required=True,
                   help="Comma-separated model names, e.g. 'Llama-3.1-405B,Llama-3.1-70B'")
    p.add_argument("--model-dir", default=os.path.expanduser("~/models"),
                   help="Base directory containing model folders (default: ~/models)")

    # Dataset
    p.add_argument("--datasets", default="alpaca",
                   help="Comma-separated dataset names (alpaca, dolly, longbench, humaneval)")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--min-words", type=int, default=5)
    p.add_argument("--max-words", type=int, default=100)

    # Distributed config
    p.add_argument("--tensor-parallel", default="8",
                   help="Comma-separated TP sizes, e.g. '4,8,16'")
    p.add_argument("--pipeline-parallel", default="2",
                   help="Comma-separated PP sizes, e.g. '1,2,4'")
    p.add_argument("--concurrency", default="1",
                   help="Comma-separated concurrency values")
    p.add_argument("--batch-sizes", default="256",
                   help="Comma-separated batch sizes")

    # Generation
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)

    # Ray cluster
    p.add_argument("--ray-head-address", default=None,
                   help="Override Ray head address (default: read from env or SLURM)")
    p.add_argument("--ray-head-port", type=int, default=6379)

    # Energy monitoring
    p.add_argument("--monitor", default="auto",
                   choices=["auto", "gpu_only", "full_node"],
                   help="Energy monitor mode (default: auto)")

    # Output
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def run():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    tp_sizes = [int(x) for x in args.tensor_parallel.split(",")]
    pp_sizes = [int(x) for x in args.pipeline_parallel.split(",")]
    concurrencies = [int(x) for x in args.concurrency.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Resolve Ray cluster config
    if args.ray_head_address:
        cluster = RayClusterConfig(
            head_address=args.ray_head_address,
            head_port=args.ray_head_port,
        )
    else:
        # from_slurm() falls back to from_env() if SLURM vars are absent
        cluster = RayClusterConfig.from_slurm()

    print(f"Ray cluster: {cluster.ray_init_address}")

    loader = DatasetLoader()
    all_results = {}

    for model_name in models:
        model_path = os.path.join(args.model_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}  — skipping.")
            continue

        model_results = {}

        for dataset_name in datasets:
            prompts = loader.load(
                dataset_name,
                num_samples=args.num_samples,
                min_words=args.min_words,
                max_words=args.max_words,
            )
            if not prompts:
                print(f"No prompts for {dataset_name} — skipping.")
                continue

            for tp in tp_sizes:
                for pp in pp_sizes:
                    for concurrency in concurrencies:
                        for batch_size in batch_sizes:
                            cfg_key = f"TP{tp}_PP{pp}_C{concurrency}_B{batch_size}"
                            print(f"\n{'='*70}")
                            print(f"Model: {model_name}  Dataset: {dataset_name}  Config: {cfg_key}")
                            print(f"{'='*70}")

                            engine_config = dict(
                                model_path=model_path,
                                tensor_parallel_size=tp,
                                pipeline_parallel_size=pp,
                                concurrency=concurrency,
                                batch_size=batch_size,
                                max_tokens=args.max_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                verbose=args.verbose,
                            )

                            engine = VLLMDistributedEngine(cluster, engine_config)
                            monitor = create_monitor(args.monitor)

                            monitor.start()
                            t0 = time.time()
                            result = engine.run_benchmark(prompts)
                            t1 = time.time()
                            time.sleep(2.0)
                            monitor.stop()

                            if result is None:
                                print(f"Benchmark failed for {cfg_key}")
                                continue

                            perf = result["performance_metrics"]
                            duration = t1 - t0
                            total_tokens = perf.get("total_tokens", 0)
                            num_responses = perf.get("total_prompts", len(prompts))

                            energy = monitor.compute_metrics(duration, total_tokens, num_responses)
                            print(energy.summary())

                            result["energy_metrics"] = {
                                "monitor_mode": args.monitor,
                                "duration_s": duration,
                                "gpu_avg_power_w": energy.gpu_avg_power_w,
                                "gpu_energy_j": energy.gpu_energy_j,
                                "gpu_mj_per_token": energy.gpu_mj_per_token,
                                "per_gpu_power_w": energy.per_gpu_power_w,
                                "cpu_avg_power_w": energy.cpu_avg_power_w,
                                "cpu_energy_j": energy.cpu_energy_j,
                                "dram_avg_power_w": energy.dram_avg_power_w,
                                "dram_energy_j": energy.dram_energy_j,
                                "system_avg_power_w": energy.system_avg_power_w,
                                "system_energy_j": energy.system_energy_j,
                                "total_energy_j": energy.total_energy_j,
                                "total_mj_per_token": energy.total_mj_per_token,
                            }

                            # Save per-config file
                            slug = model_name.replace("/", "_")
                            ts = time.strftime("%Y%m%d_%H%M%S")
                            fname = (
                                output_dir /
                                f"{slug}_{dataset_name}_tp{tp}_pp{pp}_c{concurrency}_b{batch_size}_{ts}.json"
                            )
                            with open(fname, "w") as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            print(f"Saved: {fname}")

                            model_results.setdefault(dataset_name, {})[cfg_key] = result

        if model_results:
            all_results[model_name] = model_results

    # Summary file
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"summary_{ts}.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    run()
