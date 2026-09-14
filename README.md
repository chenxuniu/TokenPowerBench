# TokenPowerBench

Code for **TokenPowerBench: Benchmarking the Power Consumption of LLM Inference**, published at **AAAI 2026**.

**Chenxu Niu, Wei Zhang, Jie Li, Yongjian Zhao, Tongyang Wang, Xi Wang, and Yong Chen.**

[AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/40535) · [Paper PDF](https://ojs.aaai.org/index.php/AAAI/article/download/40535/44496) · [arXiv](https://arxiv.org/abs/2512.03024) · [Reproduction guide](docs/reproducing.md) · [GH200 validation](docs/gh200-validation.md)

*Proceedings of the AAAI Conference on Artificial Intelligence*, 40(38), 32582–32590. DOI: [10.1609/aaai.v40i38.40535](https://doi.org/10.1609/aaai.v40i38.40535).

The maintained single-node entry point measures inference energy with an explicit sensor scope and saves the exact prompts, configuration, environment, and raw power samples. vLLM supports optional serial first-token phase profiling. Existing batch experiments remain available; they do not claim to separate overlapping prefill/decode activity.

## What changed in the maintained single-node workflow

| Area | Current behavior |
| --- | --- |
| Runtime identity and permissions | Report UID/EUID and `is_root`; independently probe GPU, IPMI, and Intel RAPL access. Missing sensors produce `null` with a reason. |
| Prefill/decode boundary | Observe the first generated token ID through incremental vLLM steps; record serial TTFT/prefill and decode host windows. |
| Energy calculation | Integrate timestamped readings over each actual window; separate IPMI node totals from component measurements and reject insufficiently sampled estimates. |
| Repeated experiments | Preserve prompts, seed, configuration, runtime, raw traces, and status; support monitor reuse across batch sizes. |
| Hardware verification | Validate root/non-root behavior and real vLLM inference on GH200; publish the commands, results, and sensor limitations. |

## Quick start: reproduce a single-node run

Use the [complete reproduction guide](docs/reproducing.md) for model preparation, root/IPMI runs, repeated experiments, result inspection, and the relationship to the paper's figures. **On GH200, use the [validated container environment](docs/gh200-validation.md#repeating-the-checks).** The native Python steps below require a compatible Linux NVIDIA GPU environment.

### 1. Get the code and install

Use a Linux NVIDIA GPU host and an isolated Python environment compatible with your chosen vLLM/CUDA release. Install the engine versions used in your experiment; the dependency lower bounds here are **not** a lockfile for the AAAI results.

```bash
git clone https://github.com/chenxuniu/TokenPowerBench.git
cd TokenPowerBench

# Select the reviewed single-node implementation, including while PR #4 is open.
git fetch origin pull/4/head
git switch --detach FETCH_HEAD

python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[vllm]'

export CUDA_VISIBLE_DEVICES=0
python run_single_node.py --check-monitor --monitor auto
```

### 2. Download the model used in the GH200 smoke tests

This pins the exact model revision used during validation. The checked-in [prompts](examples/prompts.json) are also the inputs used in the small-model tests.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    local_dir="models/Qwen2.5-0.5B-Instruct",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
)
PY
```

### 3. Run GPU-only prefill/decode profiling

```bash
python run_single_node.py \
  --model models/Qwen2.5-0.5B-Instruct \
  --prompts-file examples/prompts.json \
  --num-samples 3 --batch-sizes 1 --output-tokens 512 \
  --max-model-len 4096 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.2 --seed 42 --temperature 0 \
  --phase-profiling --monitor gpu_only \
  --output-dir results/smoke-phases
```

For an ordinary batch comparison, omit `--phase-profiling`, use `--num-samples 4 --batch-sizes 1,2`, and choose `--output-dir results/smoke-batches`. The output directory printed by the command contains `status.json`, `results.json`, per-request events, and raw samples. A short prefill should have a timing value and `null` phase energy when sensor resolution is insufficient.

These commands reproduce the **functional benchmark procedure**. Matching the paper's numerical results also requires its hardware and per-experiment model, engine, workload, and sampling settings. The [reproduction guide](docs/reproducing.md#relationship-to-the-aaai-paper) explains the supported experiments and remaining requirements.

## Root permissions and sensor scope

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
