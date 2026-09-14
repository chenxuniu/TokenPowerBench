# Python API

TokenPowerBench 1.0 provides `benchmark`, `check_environment`, and
`BenchmarkResult` from the top-level package. The Python API and CLI share the
same measurement pipeline and result format.

## Installation

Install from the repository in a Python 3.10+ environment:

```bash
git clone --branch v1.0.0 https://github.com/chenxuniu/TokenPowerBench.git
cd TokenPowerBench
python -m pip install '.[vllm,datasets]'
```

The `vllm` extra supplies inference dependencies; `datasets` supplies built-in
prompt dataset loading. For sensor access without inference, install
`python -m pip install .` and select physical NVML devices explicitly as below.
For GH200, use the [container environment](gh200-validation.md#repeating-the-checks).

## Check environment

```python
from tokenpowerbench import check_environment

report = check_environment(monitor="auto", device_indices=[0])
print(report["runtime"]["is_root"])
print(report["sensors"]["gpu"])
print(report["sensors"]["system"])
```

`check_environment(monitor="auto", device_indices=None, device_uuids=None)` probes
sensor access without loading a model and returns a dictionary with `runtime`
and `sensors`. It closes its monitor before returning.

`device_indices` selects physical NVML indices, not CUDA's remapped indices.
Alternatively pass `device_uuids=["GPU-..."]`; do not supply both selectors.
Explicit indices/UUIDs allow this check with the base NVML-only installation.
Without a selector, automatic CUDA-to-NVML UUID mapping requires a compatible
PyTorch installation. The CLI's `--check-monitor` uses automatic mapping.

| Monitor mode | Behavior |
| --- | --- |
| `gpu_only` | Probe only selected GPUs |
| `auto` | Probe GPUs and independently collect accessible IPMI/RAPL sensors |
| `full_node` | Require IPMI whole-node power; Intel RAPL CPU/DRAM remain optional |

Root status and sensor access are independent. Grace has no Intel RAPL interface,
so CPU/DRAM RAPL values remain unavailable even as root. The check does not change
permissions, and failure to initialize required GPU/IPMI sensors raises an error.

## Run a benchmark

Save this as `experiment.py`, then launch it with
`CUDA_VISIBLE_DEVICES=0 python experiment.py`. Use a local model prepared with
[a fixed revision](reproducing.md#3-download-a-fixed-model-revision).

```python
from tokenpowerbench import benchmark


def main():
    result = benchmark(
        model="models/Qwen2.5-0.5B-Instruct",
        prompts=[
            "Explain prefill and autoregressive decoding in detail.",
            "Describe GPU memory bandwidth and KV caching.",
        ],
        num_samples=3,
        batch_sizes=(1,),
        output_tokens=128,
        phase_profiling=True,
        monitor="gpu_only",
        output_dir="results/local_python_experiment",
        max_model_len=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.2,
        seed=42,
        temperature=0.0,
    )
    print(result.output_dir)
    run = result.results["batch_1"]
    print(run["total_output_tokens"], run["energy"]["gpu_energy_j"])
    for phase in run["phases"]:
        print(phase["ttft_s"], phase["decode_s"])


if __name__ == "__main__":
    main()
```

Keep the `__main__` guard: vLLM can launch subprocesses using Python's spawn
method. Set CUDA visibility before importing or initializing the inference stack.
The high-level API closes the inference engine and sensors before returning;
no context manager is required.

## Benchmark parameters

`benchmark()` accepts keyword arguments. `model` is required; it can be a local
model directory or a Hugging Face repository ID. Use a pinned local snapshot for
repeatable experiments.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `prompts` | `None` | Sequence of nonempty raw prompt strings |
| `prompts_file` | `None` | Path to a JSON array of prompt strings |
| `dataset` | `None` | Built-in dataset: `alpaca`, `dolly`, `longbench`, or `humaneval` |
| `num_samples` | `None` | Use all supplied prompts/file entries; a larger value repeats them in order |
| `batch_sizes` | `(1,)` | Batch sizes to run sequentially |
| `output_tokens` | `128` | Maximum generated tokens per request |
| `phase_profiling` | `False` | Observe serial first-token boundaries; requires `batch_sizes=(1,)` |
| `monitor` | `"auto"` | Sensor mode described above |
| `output_dir` | `"results"` | Parent directory for a unique result directory |
| `seed`, `temperature` | `42`, `0.0` | Sampling settings |
| `max_model_len` | `None` | Optional explicit context limit |
| `tensor_parallel_size` | `None` | Default to visible GPU count; an explicit value must match it |
| `gpu_memory_utilization` | `0.9` | vLLM memory fraction in `(0, 1]` |
| `min_words`, `max_words` | `2`, `300` | Built-in dataset word filters; these are not token-length bins |

Choose one prompt source. With no source, the API loads Alpaca and defaults to
1,000 requests. With supplied prompts or a prompt file, omitted `num_samples`
uses that source's length. For a built-in dataset, set the desired sample count
explicitly. Inputs are raw text without an automatic chat template.

For an ordinary batch comparison, use `batch_sizes=(1, 2)` and
`phase_profiling=False`. Model loading and warmup occur before power measurement.
All batch configurations share the same saved ordered requests. Actual output
counts can be smaller than `output_tokens` because of early stopping.

## Return values and errors

A successful call returns `BenchmarkResult`:

- `output_dir`: absolute `pathlib.Path` to the run directory.
- `results`: dictionary keyed by batch, such as `"batch_1"`, containing the same
  data written to `results.json`.
- `to_dict()`: JSON-serializable representation of the result object.

Whole-run metrics are under each batch's `energy`; request events and phase
metrics are under `phases`. GPU, CPU, DRAM, and node values remain separate.
Short or insufficiently sampled windows return `None` in Python and `null` in JSON.
A successful call does not guarantee every phase has measurable energy.

Invalid arguments raise `ValueError` or `TypeError`. Dependency, sensor, model,
phase timing, and I/O failures propagate as exceptions; `KeyboardInterrupt`
propagates to the caller. After a run directory exists, failures record failed
or interrupted status and retain samples from an active measurement. Preflight
validation or sensor-probe failures may occur before artifacts exist. The
inference engine and sensors are closed on success and failure.

Use [measurement definitions](measurement.md) to interpret scope, timing,
missing values, and temporal resolution. See the [reproduction guide](reproducing.md)
for complete commands, fixed models, and result preservation.
