# TokenPowerBench

Code for **TokenPowerBench: Benchmarking the Power Consumption of LLM Inference**, AAAI 2026. [Paper](https://arxiv.org/abs/2512.03024).

The maintained single-node entry point measures inference energy with an explicit sensor scope and saves the exact prompts, configuration, environment, and raw power samples. vLLM supports optional serial first-token phase profiling. Existing batch experiments remain available; they do not claim to separate overlapping prefill/decode activity.

## Install and check access

Use a Linux NVIDIA GPU host and an isolated Python environment compatible with your chosen vLLM/CUDA release. Install the engine versions used in your experiment; the dependency lower bounds here are **not** a lockfile for the AAAI results.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[vllm]'
python run_single_node.py --check-monitor --monitor auto
```

| Measurement | Source | Typical permission | Result when unavailable |
| --- | --- | --- | --- |
| Selected GPU power | NVML (`pynvml`), the telemetry underlying `nvidia-smi` | Ordinary GPU access | Error if GPU selection/initialization fails; invalid readings are missing |
| CPU package energy, optional DRAM | Intel RAPL `energy_uj` in `/sys/class/powercap` | Root or administrator-granted sysfs read access | `null`, with the reason recorded |
| Total node power | `ipmitool dcmi power reading`, polled every second | Root/local BMC device access or delegated IPMI permission | `null`, with the reason recorded |

**Without RAPL/IPMI permissions, measure GPU power only.** GPU energy is never presented as total node energy. Every launch, including `--check-monitor`, reports the process identity before probing sensors. `runtime.json` records real UID/GID, effective UID/GID, `is_root`, platform, and architecture. `is_root` means effective UID 0 in the current process environment; actual sensor access is recorded separately in `capabilities.json`.

Root alone does not guarantee that the hardware exposes RAPL or IPMI. In particular, the **GH200 Grace ARM CPU has no Intel RAPL interface**: CPU/DRAM measurements remain `null` even as root. If its IPMI sensor is readable, whole-node power can still be measured independently.

- `--monitor gpu_only`: use NVML without probing RAPL or IPMI.
- `--monitor auto`: detect RAPL and IPMI access independently and report the available sensors.
- `--monitor full_node`: require a successful IPMI whole-node power reading; fail explicitly otherwise. CPU/DRAM RAPL measurements are independently optional.

Administrators can grant sensor access to a non-root account. The tool records the identity and permissions with which it was launched; it performs no automatic `sudo`, permission changes, or import-time installations.

## Single-node batch benchmark

```bash
python run_single_node.py \
  --model /path/to/model \
  --dataset alpaca --num-samples 1000 \
  --batch-sizes 128,256 --output-tokens 500 \
  --monitor gpu_only --seed 42 --temperature 0
```

Set `CUDA_VISIBLE_DEVICES` before launch to choose inference GPUs; monitoring maps the CUDA-visible devices to NVML UUIDs. Use an otherwise idle, dedicated node for node-level comparisons.

Use `--max-model-len` to specify the model context limit and `--gpu-memory-utilization` to set the vLLM memory fraction (default `0.9`). `--tensor-parallel-size` defaults to the number of CUDA-visible GPUs; an explicit value must equal that count so inference and monitoring cover the same devices. For a single GH200, select `CUDA_VISIBLE_DEVICES=0` and use `--tensor-parallel-size 1`.

For fixed local inputs, replace `--dataset alpaca` with `--prompts-file prompts.json`; the file is a JSON array of prompt strings. Prompts are repeated in order to reach `--num-samples`, and the exact expanded input list is saved. Dataset failures stop the run instead of silently substituting example prompts. Inputs are raw text without an implicit chat template.

## Prefill/decode timing and energy

```bash
python run_single_node.py \
  --model /path/to/model \
  --dataset alpaca --num-samples 20 \
  --batch-sizes 1 --output-tokens 256 \
  --phase-profiling --monitor auto
```

This mode uses incremental vLLM engine steps to observe the first generated **token ID**. It requires one request at a time and uses the same monotonic clock as power sampling. The engine is asked to disable prefix caching and chunked prefill; some vLLM versions override requested settings. Inspect the `requested`, `effective`, and `engine_class` fields in `engine_config.json`. Chunked prefill does not invalidate the serial host window by itself: the runner waits for one request to finish before submitting another, preventing overlap between different requests' phases.

- **TTFT:** request submission → first observed token.
- **Prefill proxy:** request submission → first observed token, the same window as TTFT; includes dispatch, queueing, scheduler, host, and first-token processing overhead.
- **Decode window:** first observed token → request completion.

`submitted_s` is recorded immediately before `add_request`; `dispatch_completed_s` records its return as a diagnostic event. A background engine may already be executing when dispatch returns, so that return cannot define GPU execution start. These are host-observed windows, not exact GPU kernel boundaries. Normal batched generation does not provide separate phase energy. The first generated token belongs to the prefill proxy; decode energy/token uses the remaining generated token IDs.

**Short phases may have timestamps but no reliable energy value.** IPMI is polled once per second, and NVML power can itself be averaged over one second on newer GPUs. The current monitor conservatively declines subsecond GPU/IPMI energy estimates and requires sufficient samples for each window. It does not manufacture a split by multiplying whole-run average power by phase duration. See [measurement definitions and resolution limits](docs/measurement.md).

## Reproduction artifacts

Each run creates a unique `results/local_<timestamp>_<id>/` directory:

- `config.json`, `engine_config.json`, `environment.json`: CLI settings, requested and effective engine configuration plus engine class, package versions, and repository state. Unavailable effective settings are `null`, never assumed to match the request.
- `runtime.json`: process UID/EUID/GID/EGID, root status, platform, and architecture; also included in `environment.json`.
- `prompts.json`: exact ordered input requests; its hash is stored in the configuration.
- `capabilities.json`: sensor availability, permissions, GPU identity, and measurement scope.
- `batch_*_power_samples.json`: timestamped raw GPU/RAPL/IPMI readings, including complete IPMI DCMI output and BMC metadata for sensor audits.
- `batch_*_result.json`, `results.json`: exact output token counts, throughput, energy, and optional per-request phase events.
- `status.json`: completion or failure; inference failures retain sampled data.

Missing sensors and insufficient samples produce `null` plus reasons, never artificial zero-watt measurements. Results use a new schema; historical scripts expecting a top-level `batch_*` JSON file should read `results.json` or adapt to the explicit `energy` object.

## Existing code and validation

`SingleNode/llm_benchmark.py --engine vllm` routes to the maintained runner. Legacy Transformers/DeepSpeed/TensorRT-LLM adapters share the corrected monitor but have no phase instrumentation. The multi-node caller receives only monitor lifecycle/time-window compatibility fixes; it has not been validated for cluster-wide energy attribution. [Historical usage](docs/legacy-usage.md) is retained for reference.

```bash
python -m unittest discover -s tests -v
```

Tests cover runtime identity, permission fallback, sensor integration, first-token events, failure cleanup, and result artifacts with mocked hardware. **Real GH200 validation** also exercised non-root/root permissions, Qwen2.5-0.5B/7B-Instruct phase runs, and monitor reuse across batch sizes 1 and 2. See the [commands, observed results, and sensor limitations](docs/gh200-validation.md). The [GH200 Dockerfile](docker/Dockerfile.gh200) uses `nvcr.io/nvidia/vllm:25.09-py3` plus `ipmitool`; container root still needs access to the host IPMI device. These are functional checks, not numerical reproduction of the AAAI figures or validation of exact GPU kernel boundaries. Intel RAPL needs separate testing on an Intel host. See [single-node review findings](docs/review-single-node.md) and the [measurement contract](docs/measurement.md).
