# TokenPowerBench 1.0

A Python library for measuring LLM inference power, energy, throughput, and
prefill/decode timing on a single NVIDIA GPU node.

TokenPowerBench accompanies **TokenPowerBench: Benchmarking the Power Consumption
of LLM Inference**, published at **AAAI 2026**.

**Chenxu Niu, Wei Zhang, Jie Li, Yongjian Zhao, Tongyang Wang, Xi Wang, and Yong Chen.**

[AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/40535) ·
[Paper PDF](https://ojs.aaai.org/index.php/AAAI/article/download/40535/44496) ·
[arXiv](https://arxiv.org/abs/2512.03024) ·
[Python API](docs/api.md) · [Reproduction guide](docs/reproducing.md)

*Proceedings of the AAAI Conference on Artificial Intelligence*, 40(38),
32582–32590. DOI: [10.1609/aaai.v40i38.40535](https://doi.org/10.1609/aaai.v40i38.40535).

- Run experiments from Python or the command line with the same measurement pipeline.
- Measure selected GPUs through NVML, CPU/DRAM through Intel RAPL, and whole-node power through IPMI.
- Record root status and sensor access explicitly; unavailable measurements are `null` with a reason.
- Profile serial requests using first-token events to separate TTFT/prefill and decode host windows.
- Save exact prompts, configuration, environment, raw sensor readings, and per-request results.

## Installation

Use Python 3.10+ on Linux with NVIDIA GPUs and a compatible CUDA/vLLM environment.
The inference extra targets vLLM 0.10.x. On **GH200/aarch64**, use the
[container installation and commands](docs/gh200-validation.md#repeating-the-checks).

```bash
git clone https://github.com/chenxuniu/TokenPowerBench.git
cd TokenPowerBench
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[vllm,datasets]'

tokenpowerbench --version
```

For sensor access without inference, install `python -m pip install .` and use
`check_environment(device_indices=[0])`. The base package only requires
`nvidia-ml-py`. Local prompt lists and JSON files do not require the `datasets`
extra. These instructions install from GitHub; no PyPI release is required.

## Python quick start

Download a [fixed model revision](docs/reproducing.md#3-download-a-fixed-model-revision),
save this as `experiment.py`, and run `CUDA_VISIBLE_DEVICES=0 python experiment.py`:

```python
from tokenpowerbench import benchmark


def main():
    result = benchmark(
        model="models/Qwen2.5-0.5B-Instruct",
        prompts=["Explain how a language model generates tokens."],
        output_tokens=512,
        phase_profiling=True,
        monitor="gpu_only",
        max_model_len=4096,
        gpu_memory_utilization=0.2,
    )
    print(result.output_dir)
    run = result.results["batch_1"]
    print(run["energy"])
    for request in run["phases"]:
        print(request["ttft_s"], request["decode_s"])


if __name__ == "__main__":
    main()
```

`benchmark()` returns a `BenchmarkResult` containing an absolute artifact path
and a results dictionary. Supplied prompts run once each unless `num_samples`
is specified. The `__main__` guard supports vLLM worker spawning. Model loading
and warmup happen before measurement. See the [API reference](docs/api.md) for
parameters, sensor checks, batch comparisons, return values, and exceptions.

## Command-line quick start

The `tokenpowerbench` command and `python -m tokenpowerbench` are equivalent.

```bash
export CUDA_VISIBLE_DEVICES=0
tokenpowerbench --check-monitor --monitor auto

tokenpowerbench \
  --model models/Qwen2.5-0.5B-Instruct \
  --prompts-file examples/prompts.json \
  --num-samples 3 --batch-sizes 1 --output-tokens 512 \
  --max-model-len 4096 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.2 --seed 42 --temperature 0 \
  --phase-profiling --monitor gpu_only \
  --output-dir results/smoke-phases
```

For a batch comparison, omit `--phase-profiling` and use
`--num-samples 4 --batch-sizes 1,2`. For a built-in dataset, replace
`--prompts-file examples/prompts.json` with `--dataset alpaca` and choose a
sample count. Inputs are raw text without an implicit chat template.

Set `CUDA_VISIBLE_DEVICES` before launch. Monitoring maps the visible GPUs to
NVML UUIDs; `tensor_parallel_size` defaults to their count and must match it
when specified. Use an otherwise idle, dedicated node for whole-node experiments.

## Configure a multi-node cluster

Use the interactive terminal wizard to enter node IPs or select SLURM discovery:

```bash
python -m pip install '.[distributed]'
tpbench-multi --configure-cluster --cluster-config cluster.json
```

Choose **manual** to enter a head IPv4 address, optional worker addresses, and
the Ray port. The wizard saves the profile and prints the command to run on each
node. It does not SSH into nodes or start Ray. Workers join the head; the
benchmark connects to that one head address.

Choose **SLURM** to discover the head and workers from the job's allocated nodes
at launch, with an optional network interface and port. A saved SLURM profile
can be reused across allocations without recording fixed node IPs. Follow the
[multi-node setup guide](docs/multi-node.md#configure-a-cluster-profile) for
profile-based launches and commands to inspect allocated nodes.

## Root permissions and measurement scope

| Measurement | Source | Typical permission | When unavailable |
| --- | --- | --- | --- |
| Selected GPU power | NVML (`pynvml`), also used by `nvidia-smi` | Ordinary GPU access | Initialization fails if selected GPUs are inaccessible; invalid readings are missing |
| CPU package energy and optional DRAM | Intel RAPL `energy_uj` under `/sys/class/powercap` | Root or delegated sysfs read access | `null`, with a reason |
| Whole-node power | `ipmitool dcmi power reading`, polled every second | Root/local BMC device access or delegated IPMI permission | `null`, or a required-sensor error in `full_node` mode |

**Without RAPL/IPMI access, GPU power is the available measurement.** GPU energy
is never labeled as whole-node energy. IPMI measures the entire node, including
components and activity outside the selected GPUs.

Benchmark launches report process UID/EUID and `is_root`. Runs save this identity
in `runtime.json` and actual sensor availability in `capabilities.json`.
Root access alone cannot supply a sensor that the hardware does not expose:
**the GH200 Grace ARM CPU has no Intel RAPL interface**, so CPU/DRAM RAPL values
remain `null` even as root. Accessible IPMI can still provide whole-node power.

- `gpu_only`: use NVML without probing RAPL or IPMI.
- `auto`: independently probe GPU, RAPL, and IPMI access.
- `full_node`: require IPMI whole-node power; CPU/DRAM remain independently optional.

The library uses the permissions of the calling process. See the
[reproduction guide](docs/reproducing.md) for root and non-root commands.

## Prefill and decode

Phase profiling requires serial requests with `batch_sizes=(1,)`. It observes
the first generated **token ID** through incremental vLLM engine steps and
uses the same monotonic clock as the power sampler.

| Window | Start | End |
| --- | --- | --- |
| TTFT / prefill proxy | Immediately before request submission | First observed output token |
| Decode | First observed output token | Request completion |

The prefill proxy is the TTFT window and includes dispatch, queueing, scheduler,
host, and first-token processing overhead. These are host-observed boundaries,
not GPU kernel boundaries. The first output token belongs to the prefill proxy;
decode energy per token uses the remaining generated tokens. Ordinary batched
runs report aggregate energy because requests can overlap across phases.

**A phase can have valid timing and unavailable energy.** IPMI is polled once
per second, and NVML readings can themselves represent a one-second average.
GPU/IPMI windows shorter than one second or lacking sufficient samples return
`null` energy. Raw readings and reasons are saved. Inspect `engine_config.json`
for requested and effective engine settings, including prefix caching and
chunked prefill. See [measurement definitions](docs/measurement.md) for the
integration rules and temporal resolution limits.

## Reproduction and validation

Each call creates a unique directory under `output_dir` containing:

- `config.json`, `engine_config.json`, `environment.json`: workload, engine settings, and software/runtime metadata.
- `runtime.json`, `capabilities.json`: root status, device identity, sensor availability, and scope.
- `prompts.json`: exact ordered requests, with their hash in the configuration.
- `batch_*_power_samples.json`: timestamped GPU/RAPL/IPMI readings and IPMI DCMI metadata.
- `batch_*_result.json`, `results.json`: token counts, timing, energy, and optional request phase events.
- `status.json`: completion, failure, or interruption; active measurements retain raw samples on failure.

Follow the [reproduction guide](docs/reproducing.md) for fixed model revisions,
repeated experiments, result inspection, and the relationship to the paper's
figures. [GH200 validation](docs/gh200-validation.md) records the environment,
commands, measured results, and hardware limits. Functional checks do not imply
numerical reproduction of the AAAI figures. Intel RAPL requires an Intel host.

The v1.0 validated inference path is single-node vLLM. The
[multi-node guide](docs/multi-node.md) describes the `tpbench-multi` CLI, Ray
replicas, and SLURM launch commands. Distributed orchestration has local process
and mocked test coverage; GPU/NCCL execution requires validation on the target
cluster. Cluster energy and distributed prefill/decode attribution are unavailable.
The distributed interface is outside the validated single-node Python API scope.

```bash
python -m unittest discover -s tests -v
```

## Citation


If you use TokenPowerBench in your research, please cite the [AAAI 2026 paper](https://ojs.aaai.org/index.php/AAAI/article/view/40535):

```bibtex
@article{niu2026tokenpowerbench,
  title   = {{TokenPowerBench}: Benchmarking the Power Consumption of {LLM} Inference},
  author  = {Niu, Chenxu and Zhang, Wei and Li, Jie and Zhao, Yongjian and
             Wang, Tongyang and Wang, Xi and Chen, Yong},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {40},
  number  = {38},
  pages   = {32582--32590},
  year    = {2026},
  doi     = {10.1609/aaai.v40i38.40535},
  url     = {https://ojs.aaai.org/index.php/AAAI/article/view/40535}
}
```
