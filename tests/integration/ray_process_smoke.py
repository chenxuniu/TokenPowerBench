"""Exercise real local Ray processes with synthetic GPU resources and no CUDA/vLLM."""
import argparse
import json
from pathlib import Path
import time
import uuid
from unittest.mock import patch

import ray
from ray.cluster_utils import Cluster

from tokenpowerbench.distributed import RayClusterConfig
from tokenpowerbench.distributed import vllm_distributed as distributed


class LogicalGPUWorker:
    def ready(self):
        return {"node_id": ray.get_runtime_context().get_node_id(), "gpu_ids": ray.get_gpu_ids()}


class StubPredictor:
    def __init__(self, model_path, tensor_parallel_size, pipeline_parallel_size, sampling_params, **kwargs):
        worker = ray.remote(num_cpus=0, num_gpus=1)(LogicalGPUWorker)
        self.workers = [worker.remote() for _ in range(tensor_parallel_size * pipeline_parallel_size)]
        self.worker_info = ray.get([actor.ready.remote() for actor in self.workers], timeout=20)

    def ready(self):
        return {"synthetic_workers": self.worker_info}

    def warmup(self):
        return {"synthetic_warmup": True}

    def __call__(self, batch):
        time.sleep(0.02 if batch["request_id"][0] == 0 else 0.005)
        size = len(batch["text"])
        return {"request_id": batch["request_id"], "prompt": batch["text"],
                "generated_text": ["synthetic response"] * size,
                "input_tokens": [2] * size, "output_tokens": [3] * size,
                "batch_id": [uuid.uuid4().hex] * size,
                "batch_duration_s": [0.005] * size, "batch_size": [size] * size}

    def close(self):
        for actor in self.workers:
            ray.kill(actor, no_restart=True)
        self.workers.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ray-process-smoke"))
    args = parser.parse_args()
    output = args.output_dir / uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    cluster = Cluster()
    report = {"scope": "local Ray processes and synthetic resources; no GPU inference or physical multi-node validation", "runs": []}
    engine = None
    try:
        cluster.add_node(num_cpus=2, num_gpus=2, object_store_memory=100 * 1024 ** 2, include_dashboard=False)
        cluster.add_node(num_cpus=2, num_gpus=2, object_store_memory=100 * 1024 ** 2, include_dashboard=False)
        ray.init(address=cluster.address, log_to_driver=False)
        assert len([node for node in ray.nodes() if node["Alive"]]) == 2
        mismatch = distributed.VLLMDistributedEngine(
            RayClusterConfig(head_address="127.0.0.1:1"), {"model_path": "synthetic-model"})
        try:
            try:
                mismatch.prepare()
            except RuntimeError as error:
                assert "Cannot verify" in str(error), error
            else:
                raise AssertionError("An explicit mismatched cluster address was accepted")
        finally:
            mismatch.close()
        assert ray.is_initialized()
        report["mismatched_address_rejected"] = True
        with patch.object(distributed, "VLLMPredictor", StubPredictor):
            for width, replicas in ((3, 1), (1, 2)):
                config = {"model_path": "synthetic-model", "tensor_parallel_size": width,
                          "pipeline_parallel_size": 1, "concurrency": replicas, "batch_size": 2,
                          "placement_timeout_s": 30, "startup_timeout_s": 40,
                          "inference_timeout_s": 20, "shutdown_timeout_s": 20}
                engine = distributed.VLLMDistributedEngine(RayClusterConfig(head_address=cluster.address), config)
                prepared = engine.prepare()
                if width == 3:
                    nodes = {entry["node_id"] for entry in prepared["workers"][0]["synthetic_workers"]}
                    assert len(nodes) == 2, nodes
                result = engine.run_benchmark([f"prompt {i}" for i in range(5)])
                assert result["performance_metrics"]["total_tokens"] == 15
                assert result["performance_metrics"]["total_batches"] == 3
                assert [row["request_id"] for row in result["results"]] == list(range(5))
                repeated = engine.run_benchmark(["repeat one", "repeat two"])
                assert repeated["performance_metrics"]["total_tokens"] == 6
                engine.close()
                assert ray.is_initialized()
                deadline = time.monotonic() + 10
                while ray.available_resources().get("GPU", 0) != 4:
                    assert time.monotonic() < deadline, ray.available_resources()
                    time.sleep(0.1)
                report["runs"].append({"width": width, "replicas": replicas, "prepared": prepared,
                                       "metrics": result["performance_metrics"], "resources_after_close": ray.available_resources()})
                print("RAY_PROCESS_CASE_OK", width, replicas, flush=True)
        (output / "summary.json").write_text(json.dumps(report, indent=2))
    finally:
        if engine is not None:
            engine.close()
        ray.shutdown()
        cluster.shutdown()
    print(f"RAY_PROCESS_SMOKE_PASSED: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
