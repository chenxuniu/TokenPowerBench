"""
TokenPowerBench — GPU energy benchmarking for LLM inference.

Quick start
-----------
Single node (GPU-only, no root required):

    from tokenpowerbench.energy import create_monitor
    from tokenpowerbench.engines import VLLMEngine

    monitor = create_monitor("gpu_only")
    engine = VLLMEngine()
    engine.setup_model("/path/to/model")
    engine.run_inference(["warmup"], batch_size=1, max_tokens=5)

    monitor.start()
    outputs, t0, t1 = engine.run_benchmark(prompts, num_samples, batch_size, max_tokens)
    monitor.stop()
    metrics = monitor.compute_metrics(t1 - t0, engine.estimate_tokens(outputs), len(outputs))
    print(metrics.summary())

Full-node (root / sudo — includes CPU + DRAM + system power):

    monitor = create_monitor("full_node")   # or create_monitor("auto")
    ...

Multi-node with Ray:

    from tokenpowerbench.distributed import VLLMDistributedEngine, RayClusterConfig

    cluster = RayClusterConfig.from_slurm()  # auto-detects head IP from SLURM
    engine = VLLMDistributedEngine(cluster, config)
    result = engine.run_benchmark(prompts)
"""

__version__ = "0.2.0"
