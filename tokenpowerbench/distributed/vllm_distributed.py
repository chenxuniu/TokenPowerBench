"""
VLLMDistributedEngine — multi-node vLLM inference via Ray Data.

Usage
-----
    from tokenpowerbench.distributed import VLLMDistributedEngine
    from tokenpowerbench.distributed.ray_cluster import RayClusterConfig

    cluster = RayClusterConfig.from_slurm()   # or from_env() / from_env()
    engine = VLLMDistributedEngine(cluster, engine_config)
    result = engine.run_benchmark(prompts)
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import ray
import numpy as np
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import SamplingParams

from .ray_cluster import RayClusterConfig
from .predictor import VLLMPredictor

_MIN_RAY_VERSION = "2.22.0"


class VLLMDistributedEngine:
    """
    Distributed vLLM inference engine.

    Parameters
    ----------
    cluster : RayClusterConfig
        How to connect to the Ray cluster (no hardcoded IPs).
    config : dict
        Engine settings:
            model_path          str   – path to the model weights
            tensor_parallel_size int
            pipeline_parallel_size int
            concurrency         int   – number of concurrent Ray workers
            batch_size          int   – prompts per worker batch
            max_tokens          int   – max output tokens
            temperature         float
            top_p               float
            verbose             bool
    """

    def __init__(self, cluster: RayClusterConfig, config: Dict[str, Any]) -> None:
        self.cluster = cluster
        self.config = config
        self.model_path = config["model_path"]
        self.tensor_parallel_size = config["tensor_parallel_size"]
        self.pipeline_parallel_size = config["pipeline_parallel_size"]
        self.concurrency = config["concurrency"]
        self.batch_size = config["batch_size"]
        self.verbose = config.get("verbose", False)

        self.sampling_params = SamplingParams(
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            max_tokens=config.get("max_tokens", 512),
            stop=["\n\n", "Human:", "Assistant:", "###"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_benchmark(self, prompts: List[str]) -> Optional[Dict[str, Any]]:
        """Run a full benchmark sweep over the given prompts.

        Returns a dict with keys: "results", "performance_metrics", "configuration".
        Returns None if the run failed.
        """
        try:
            if not self._init_ray():
                return None
            dataset = self._build_dataset(prompts)
            return self._run_inference(dataset, prompts)
        except Exception as exc:
            print(f"[VLLMDistributedEngine] Benchmark failed: {exc}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Ray initialisation
    # ------------------------------------------------------------------

    def _init_ray(self) -> bool:
        """Connect to the Ray cluster and validate resource availability."""
        if Version(ray.__version__) < Version(_MIN_RAY_VERSION):
            print(f"[VLLMDistributedEngine] Ray >= {_MIN_RAY_VERSION} required, found {ray.__version__}")
            return False

        if not ray.is_initialized():
            print(f"[VLLMDistributedEngine] Connecting to Ray at {self.cluster.ray_init_address}")
            ray.init(address=self.cluster.ray_init_address, ignore_reinit_error=True)
        else:
            print("[VLLMDistributedEngine] Ray already initialized.")

        time.sleep(2)  # wait for cluster to settle

        # Query cluster resources with retries
        total_gpus = 0
        for attempt in range(5):
            try:
                nodes = ray.nodes()
                alive = [n for n in nodes if n["Alive"]]
                resources = ray.cluster_resources()
                node_gpus = sum(int(n.get("Resources", {}).get("GPU", 0)) for n in alive)
                total_gpus = max(int(resources.get("GPU", 0)), node_gpus)
                if total_gpus > 0:
                    break
                print(f"[VLLMDistributedEngine] No GPUs detected (attempt {attempt + 1}/5), retrying…")
                time.sleep(3)
            except Exception as exc:
                print(f"[VLLMDistributedEngine] Cluster query failed (attempt {attempt + 1}): {exc}")
                time.sleep(2)

        print(f"[VLLMDistributedEngine] Cluster: {len(alive)} node(s), {total_gpus} GPU(s)")
        for n in alive:
            hostname = n.get("NodeManagerHostname", "?")
            gpus = int(n.get("Resources", {}).get("GPU", 0))
            cpus = int(n.get("Resources", {}).get("CPU", 0))
            print(f"  {hostname}: {gpus} GPUs, {cpus} CPUs")

        required = self.tensor_parallel_size * self.pipeline_parallel_size * self.concurrency
        if total_gpus < required:
            print(
                f"[VLLMDistributedEngine] Insufficient GPUs: need {required}, "
                f"have {total_gpus}. "
                f"(TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size}, "
                f"concurrency={self.concurrency})"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Dataset & inference
    # ------------------------------------------------------------------

    def _build_dataset(self, prompts: List[str]):
        print(f"[VLLMDistributedEngine] Building Ray dataset from {len(prompts)} prompts.")
        return ray.data.from_items([{"text": p} for p in prompts])

    def _run_inference(self, dataset, prompts: List[str]) -> Optional[Dict[str, Any]]:
        print(
            f"[VLLMDistributedEngine] Starting inference  "
            f"model={os.path.basename(self.model_path)}  "
            f"TP={self.tensor_parallel_size}  PP={self.pipeline_parallel_size}  "
            f"concurrency={self.concurrency}  batch={self.batch_size}"
        )
        t0 = time.time()
        try:
            resources_kwargs = self._placement_group_kwargs()
            dataset = dataset.map_batches(
                VLLMPredictor,
                fn_constructor_kwargs=dict(
                    model_path=self.model_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    pipeline_parallel_size=self.pipeline_parallel_size,
                    sampling_params=self.sampling_params,
                    verbose=self.verbose,
                ),
                concurrency=self.concurrency,
                batch_size=self.batch_size,
                **resources_kwargs,
            )
            outputs = dataset.take_all()
            t1 = time.time()
            return self._process_results(outputs, t0, t1, prompts)
        except Exception as exc:
            print(f"[VLLMDistributedEngine] Inference failed: {exc}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None

    def _placement_group_kwargs(self) -> Dict[str, Any]:
        """Build Ray placement-group scheduling strategy."""

        def _make_strategy():
            nodes = [n for n in ray.nodes() if n["Alive"]]
            total_gpus = self.tensor_parallel_size * self.pipeline_parallel_size

            if len(nodes) >= 2 and total_gpus >= 8:
                # Spread across nodes for large multi-node models
                gpus_per_node = max(
                    1,
                    int(nodes[0].get("Resources", {}).get("GPU", 4))
                )
                bundles = []
                remaining = total_gpus
                for node in nodes:
                    if remaining <= 0:
                        break
                    alloc = min(gpus_per_node, remaining)
                    bundles.extend([{"GPU": 1, "CPU": 2}] * alloc)
                    remaining -= alloc
                strategy = "SPREAD"
            else:
                bundles = [{"GPU": 1, "CPU": 1}] * total_gpus
                strategy = "PACK"

            pg = ray.util.placement_group(bundles, strategy=strategy)
            return dict(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    pg, placement_group_capture_child_tasks=True
                )
            )

        return {"num_gpus": 0, "ray_remote_args_fn": _make_strategy}

    # ------------------------------------------------------------------
    # Result processing
    # ------------------------------------------------------------------

    def _process_results(
        self,
        outputs: List[Dict],
        t0: float,
        t1: float,
        prompts: List[str],
    ) -> Dict[str, Any]:
        total_time = t1 - t0
        results = []
        total_tokens = 0
        total_proc_time = 0.0

        for i, out in enumerate(outputs):
            instruction = self._scalar(out.get("prompt", prompts[i] if i < len(prompts) else ""))
            response = self._scalar(out.get("generated_text", ""))
            proc = float(self._scalar(out.get("processing_time", 0.0)))
            tokens = len(response.split()) if isinstance(response, str) else 0
            total_tokens += tokens
            total_proc_time += proc
            results.append({
                "id": i + 1,
                "prompt": instruction,
                "response": response,
                "tokens_generated": tokens,
                "processing_time_s": proc,
                "tokens_per_second": tokens / proc if proc > 0 else 0,
            })

        metrics = {
            "total_prompts": len(outputs),
            "total_time_s": total_time,
            "total_tokens": total_tokens,
            "avg_processing_time_s": total_proc_time / len(outputs) if outputs else 0,
            "throughput_tokens_per_s": total_tokens / total_time if total_time > 0 else 0,
            "success_rate": len(results) / len(prompts) if prompts else 0,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "concurrency": self.concurrency,
            "batch_size": self.batch_size,
        }

        print(
            f"[VLLMDistributedEngine] Done: {metrics['total_prompts']} prompts, "
            f"{metrics['throughput_tokens_per_s']:.1f} tok/s, "
            f"success={metrics['success_rate']:.1%}"
        )

        return {
            "results": results,
            "performance_metrics": metrics,
            "configuration": {
                "model_path": self.model_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "concurrency": self.concurrency,
                "batch_size": self.batch_size,
            },
        }

    @staticmethod
    def _scalar(value: Any) -> Any:
        if isinstance(value, list) and len(value) > 0:
            return value[0]
        return value
