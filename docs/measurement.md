# Single-node measurement contract

The supported path for these measurements is `run_single_node.py` with the
`tokenpowerbench` package. `SingleNode/llm_benchmark.py --engine vllm` routes to
this runner. Other legacy engine adapters and `MultipleNode/` do not acquire
the corrected phase measurements described here.

## What can be measured with your permissions

| Quantity | Source | Required access | Meaning |
| --- | --- | --- | --- |
| GPU power | NVIDIA NVML, through `pynvml` | NVIDIA device access and a supported driver/power sensor; root is normally unnecessary | Board power of the selected GPU(s), including associated circuitry |
| CPU package energy | Intel RAPL `energy_uj` files under Linux powercap sysfs | Usually root, or read access explicitly delegated by the administrator | Energy counter for each accessible CPU package |
| DRAM energy | A separate Intel RAPL DRAM domain, where exposed | The same sysfs read permissions | Energy of the exposed DRAM domain; not available on every platform |
| Total node power | `ipmitool dcmi power reading` | Usually root for local IPMI access, or administrator-configured IPMI/BMC access | Total node power reported by that server's BMC |

**没有 root、没有管理员授予的 RAPL/IPMI 读取权限时，只能测 GPU。
CPU 和整个 node 的功耗应显示为不可用，不能写成 0，也不能把 GPU 功耗称为整机功耗。**

RAPL and IPMI are independent capabilities: having access to one does not prove
access to the other. Root access also does not create a sensor on an unsupported
machine. The monitor checks actual reads, and records unavailable sensors and
their reasons. It does not change system permissions or invoke `sudo`.

Each launch prints process identity before probing sensors, including with
`--check-monitor`. Benchmark runs save the same identity in `runtime.json` and
under `runtime` in `environment.json`:

- `uid` and `gid`: real process user and group IDs.
- `euid` and `egid`: effective process user and group IDs.
- `is_root`: whether the effective UID is 0; `null` if effective IDs are
  unsupported. A real UID of 0 alone does not imply `is_root: true`.
- `platform` and `machine_architecture`: operating system and CPU architecture.

Identity and sensor availability are separate facts. In a container, effective
UID 0 describes the container process and does not guarantee access to host
devices. A non-root account may have administrator-delegated sensor access.
Read `capabilities.json` for the actual result of each sensor probe.

**GH200 uses a Grace ARM CPU, which does not expose Intel RAPL.** Its missing
RAPL CPU/DRAM measurements remain `null` even when running as root. This is a
hardware/interface limitation, not proof of insufficient privileges. An
accessible IPMI sensor can still provide the server's whole-node power.

Choose a monitor mode explicitly when comparing runs:

- `--monitor gpu_only`: read GPU power only, without probing RAPL or IPMI.
- `--monitor auto`: probe RAPL and IPMI separately and collect the accessible
  sensors. Partial coverage is reported as partial coverage.
- `--monitor full_node`: require a valid IPMI node-power reading. If IPMI is
  unavailable, fail explicitly. CPU package and DRAM RAPL counters remain
  independent optional measurements; missing RAPL does not invalidate valid
  IPMI whole-node energy.

`nvidia-smi` can be used to check GPU access and reported power independently;
the benchmark sampler uses NVML. NVIDIA documents GPU power as including
associated circuitry such as device memory, so it is not the consumption of an
individual CUDA kernel or an individual process.
[NVML device-query reference](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html)

CPU power is calculated from the change in the RAPL energy counter divided by
the elapsed time, with counter wraparound handled using `max_energy_range_uj`.
CPU totals use package domains. Core/uncore subdomains overlap with their parent
package and must not be added to package energy. DRAM is reported separately
when present.
[Linux powercap/RAPL interface](https://www.kernel.org/doc/html/latest/power/powercap/powercap.html)

The `system_*` measurements refer exclusively to IPMI. Do not add GPU, CPU, or
DRAM energy to IPMI node energy: the node reading already includes components
within the BMC's measurement boundary. A sum of available component measurements
also cannot substitute for an unavailable IPMI reading. The BMC's power boundary
and any PSU treatment should be documented for the particular server.

## Prefill and decode boundaries

Enable `--phase-profiling --batch-sizes 1` to profile one request at a time. This
mode observes successive `llm_engine.step()` outputs and marks the first output
containing a generated token ID, including tokens whose decoded text is empty.
The final event is the engine's request-finished output. Calling the blocking
`LLM.generate()` and taking a timestamp after it returns cannot measure TTFT.
The step-based interface follows the
[vLLM 0.10.0 engine API](https://docs.vllm.ai/en/v0.10.0/api/vllm/engine/llm_engine.html).
If the first observed output already contains multiple tokens, or the installed
engine lacks the required step interface, phase profiling fails explicitly:
there is no trustworthy first-token boundary to recover from that output.

Four host-clock timestamps are recorded for each request:

| Event | Definition |
| --- | --- |
| `submitted_s` | Immediately before calling `add_request`; also saved as `prefill_start_s` |
| `dispatch_completed_s` | Immediately after `add_request` returns; a diagnostic event, not GPU execution start |
| `first_token_s` | When an engine step first returns a generated token ID for this request |
| `finished_s` | When the request is reported finished |

```text
submitted       dispatch completed          first token              finished
    |                    |                       |                      |
    |<------------- TTFT / prefill proxy -------->|<------ decode ----->|
    |<----------------------- request latency ------------------------->|
```

- **TTFT** and **prefill proxy** both span submitted → first observed token.
  They include request dispatch, tokenization, queueing, scheduling, prompt
  processing, first-token sampling, and host observation overhead.
- **Decode** is first token → finished. A request that generates exactly one
  token can have a zero-length decode interval and no measurable decode energy.

The dispatch timestamp is retained for diagnostics, but does not split the
energy window: a background engine can begin executing before `add_request`
returns. Excluding dispatch time could therefore omit real prefill work.
No recorded timestamp identifies actual GPU execution start.

The prefill proxy is the practical host-observed interval for the requested
time-window experiment. It is not a pure GPU prefill-kernel duration. GPU kernel
boundaries would require separate engine/profiler instrumentation. Timing and
power samples use the same monotonic host clock so that wall-clock adjustments
do not shift the integration windows.

**中文定义：调用 `add_request` 前到主机观测到首 token，是 TTFT，也是本工具的
prefill proxy 时间窗；首 token 到请求完成是 decode。请求提交返回的时间只作辅助
记录，因为后台引擎可能在返回前已开始执行。该 prefill 窗口包含提交、排队、调度、
主机处理和首 token 采样开销，不能解释为纯 GPU kernel 时间。**

Phase mode requests `enable_prefix_caching=False`,
`enable_chunked_prefill=False`, and `max_num_seqs=1`. The actual installed vLLM
may override requested settings, including enabling chunked prefill. Inspect
`engine_config.json`: `requested` records constructor arguments, `effective`
records runtime configuration, and `engine_class` identifies the engine
implementation. Effective settings that cannot be inspected are `null`.
Do not report chunked prefill as disabled based only on constructor arguments.

Chunked prefill can still be observed within this submission-to-first-token
window when requests run serially: the next request is submitted only after
the current one finishes. There is no overlap between different requests'
prefill/decode stages. Phase mode requires batch size 1; concurrent requests
could mix their phases in the same node-level trace, preventing unique
attribution. The host interface does not certify the absence of internal
preemption or recomputation. Ordinary batched throughput benchmarks remain
available, with phase attribution marked unavailable. This implementation does
not claim validated phase support for Transformers, DeepSpeed, or TensorRT-LLM.

## Integration and temporal resolution

Power readings retain their host timestamps. Energy is estimated by integrating
the trace over the actual requested window, with elapsed times between samples
accounted for. The implementation uses piecewise linear interpolation and
trapezoidal integration, requires samples covering both endpoints, and does not
extrapolate or bridge an explicit invalid reading. Average power is the
resulting energy divided by window duration.
Startup samples and idle tails are excluded by timestamps rather than dropping
an arbitrary percentage of the trace. Raw traces are retained for inspection.
Missing readings remain missing, and unavailable metrics serialize as JSON
`null` rather than 0.

Each sensor needs at least two valid readings inside a phase before that
phase's energy is reported. If the sensor lacks adequate samples, its phase
power and energy remain unavailable even when phase timing is known. Aggregate
phase results must also retain their coverage information: do not treat a sum
over measurable requests as energy for all requests.

This sampling check is a minimum data-availability check, not a guarantee of
physical time resolution:

- IPMI is polled at the original experiment's nominal **1 second** interval.
  The command's latency and BMC refresh/averaging behavior can make the effective
  interval slower. A prefill shorter than a second will often have no usable
  node-level phase estimate. Repeated reads do not create new BMC measurements.
- NVML polling frequency is distinct from sensor averaging. NVIDIA documents
  `nvmlDeviceGetPowerUsage` as returning a 1 second average on Ampere other than
  GA100, and newer GPUs; GA100 and older architectures return instantaneous
  values. Polling every 100 ms therefore does not establish 100 ms power
  resolution on every GPU. A short prefill/decode boundary may be blurred even
  when several readings are present.
  [NVML power-usage semantics](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html)
- RAPL power is an average over the interval between counter reads. A counter
  interval spanning a phase boundary cannot establish an exact instantaneous
  change at that boundary.

The current GPU monitor applies a conservative 1 second minimum window to all
selected GPUs because it does not certify each architecture's effective sensor
resolution. Thus subsecond GPU energy is deliberately null even if 100 ms
polling collected several values. This conservative policy is saved in the
capability report; it is not a claim that GA100 hardware always averages over
1 second. Longer windows remain sampled estimates and can still include sensor
averaging across their boundaries.

For publication, report the sensor, software polling interval, known sensor
refresh/averaging interval, phase duration, and coverage. Label short-window
results as estimates subject to sensor averaging. Increasing request count can
improve statistical stability but does not recover information a sensor never
resolved. Extending the prompt to obtain a longer prefill changes the workload
and should be disclosed.

The measured energy includes idle/background draw within the measured windows.
No idle baseline is subtracted. Use a dedicated node when interpreting node
power as the cost of this workload, and record the selected GPU UUIDs so that
unrelated GPUs are not accidentally included.

## Reproduction and GPU acceptance

Start from a local model and a compatible Linux/NVIDIA environment using the
installation instructions in the README. Inspect sensor access before loading
a model:

```bash
CUDA_VISIBLE_DEVICES=0 python run_single_node.py --check-monitor --monitor auto
```

For an offline prompt source, create a JSON file containing an array of nonempty
strings, for example:

```json
[
  "Explain the difference between prefill and autoregressive decoding.",
  "Describe how a computer measures energy and average power."
]
```

Pass its path with `--prompts-file`. These short prompts are convenient for a
functional check; they are unlikely to produce an IPMI-resolved prefill. The
runner repeats the ordered prompts to reach `--num-samples` and saves the exact
request list. Without this option, the default Alpaca dataset is loaded. A
minimal GPU-only phase run is:

```bash
CUDA_VISIBLE_DEVICES=0 python run_single_node.py \
  --model /path/to/model \
  --prompts-file /path/to/prompts.json \
  --monitor gpu_only \
  --phase-profiling --batch-sizes 1 \
  --num-samples 8 --output-tokens 128 \
  --output-dir results/gpu-phases
```

To collect node measurements, use an account that can read IPMI and replace
`--monitor gpu_only` with `--monitor full_node`. CPU/DRAM measurements are added
only if compatible RAPL counters exist and are readable. On GH200, successful
IPMI access is sufficient for `full_node`; CPU/DRAM RAPL values remain `null`.
`--monitor auto` is useful when access is uncertain; inspect the capability
report before using its results. A standard batched run omits phase profiling:

```bash
CUDA_VISIBLE_DEVICES=0 python run_single_node.py \
  --model /path/to/model \
  --prompts-file /path/to/prompts.json \
  --monitor gpu_only --batch-sizes 1,8 \
  --num-samples 16 --output-tokens 128 \
  --output-dir results/batched
```

The defaults are seed 42 and temperature 0.0; they can be changed with `--seed`
and `--temperature`. Inputs are raw text without an implicit chat template. The
output limit is a maximum, so use actual generated token-ID counts in metrics.
The phase mode changes scheduling and the corrected integration changes measurement conventions. Compare new runs under the same configuration; do not silently mix historical and new results.

Allocation can be made explicit with `--max-model-len`,
`--gpu-memory-utilization` (default `0.9`, valid range `(0, 1]`), and
`--tensor-parallel-size`. Tensor parallel size defaults to the number of
CUDA-visible GPUs. An explicit value must equal that number: select the exact
inference devices using `CUDA_VISIBLE_DEVICES` so NVML observes the same GPUs.
For example, a single GH200 uses `CUDA_VISIBLE_DEVICES=0` and
`--tensor-parallel-size 1`. Choose a context limit that accommodates the input
and generated output; changing these settings changes the experiment.

The runner writes a unique `local_<timestamp>_<id>/` directory containing:

| File | Contents |
| --- | --- |
| `config.json` | Resolved CLI options and the ordered prompt list's SHA-256 |
| `engine_config.json` | `requested` constructor arguments, `effective` runtime settings, and `engine_class` |
| `prompts.json` | Exact ordered requests, including any repetitions |
| `runtime.json` | Real and effective UID/GID, `is_root`, platform, and machine architecture |
| `environment.json` | Runtime identity, package versions, platform, Git revision/status, and CUDA visibility |
| `capabilities.json` | Available sensors, selected devices, and access failures |
| `status.json` | Running, completed, or failed state, with an error for a failed run |
| `batch_1_power_samples.json` | Timestamped power trace; IPMI entries also retain complete DCMI stdout/stderr, return code, and BMC metadata |
| `batch_1_result.json` | Whole-run `energy`, per-request `phases`, and `phase_status` |
| `results.json` | Combined results for all completed batch configurations |

Batch filenames use the selected batch size. Each entry in `phases` retains the
request's events and separate `prefill_proxy_energy` and `decode_energy` fields.
**Null phase energy does not mean the first-token timing is missing.** Inspect
sensor coverage and the energy status alongside the event timestamps. An empty
decode window is recorded explicitly, and a batched run without phase profiling
has `phase_status: "not_requested"`.

IPMI samples retain acquisition start/end times and the original BMC timestamp,
statistics sampling period, reading state, and min/max/average fields under
`dcmi`. The integral uses only the instantaneous-power field. The BMC timestamp
does not share the host monotonic clock, and the statistics period does not
establish the instantaneous sensor's refresh interval. Explicitly deactivated
readings are missing values. If all readings covering a valid window are
identical, the result carries an additional warning: stable power, slow updates,
or stale BMC readings cannot be distinguished from that trace alone. Inspect
these diagnostics before claiming a physical power difference between phases.

Before accepting results on the target GPU machine:

1. Check reported runtime UID/EUID/root status and GPU UUIDs against the
   launched process and allocated devices. Confirm `gpu_only` results have
   unavailable CPU/DRAM/node values. In `auto`, a non-root account without
   delegated RAPL/IPMI access must retain only the available GPU measurements.
   Confirm `full_node` fails when IPMI is unavailable, even if RAPL is readable.
2. With the intended sensor permissions, confirm IPMI access and any available
   RAPL package names. On GH200, confirm `full_node` succeeds with readable
   IPMI while missing Intel RAPL remains `null`, including as root. Compare a
   sustained-load trace with direct `nvidia-smi` and IPMI
   observations. Check that ordinary node values above 1000 W remain in the
   trace when using a server that draws that much power.
3. Generate multiple tokens for one request and inspect its events:
   `submitted_s` ≤ `dispatch_completed_s` ≤ `first_token_s` ≤ `finished_s`.
   Confirm `prefill_start_s == submitted_s` and `prefill_proxy_s == ttft_s`.
   Dispatch completion must not be interpreted as GPU execution start. Inspect
   requested and effective engine settings alongside `engine_class`.
   First token must precede request
   completion when the output requires later decode steps. Compare the observed
   boundary with a profiler trace or the corresponding vLLM engine events on
   the exact installed version.
4. Check that output-token counts equal the lengths of generated token-ID
   sequences. Verify a one-token request does not create a fictitious decode
   interval. Confirm warmup and idle tails are outside the measured windows.
5. Inspect a short prefill: insufficiently sampled sensors should have null
   phase energy. Inspect a longer request and recompute its integrals from the
   trace, while retaining the sensor-averaging qualification above.
6. Run more than one batch configuration in the same process and confirm that
   valid GPU readings continue after the first monitor stop/start cycle. Save
   the result JSON, traces, exact command, model/data revisions, environment
   versions, server details, and any administrator sensor configuration.

The GH200 container recipe in
[`docker/Dockerfile.gh200`](../docker/Dockerfile.gh200) extends
`nvcr.io/nvidia/vllm:25.09-py3` with `ipmitool`. Source and local model weights
are mounted at run time. The recipe declares `USER root`; host sensor devices
still need to be accessible from the container. Actual non-root/root GH200
phase and batch checks, their observed results, and the sensor limitations are
recorded separately in [the GH200 validation report](gh200-validation.md).

Local tests use fake engines/sensors and exercise runtime identity, boundary
logic, permission handling, and integration arithmetic. Passing these tests
does not establish hardware validation or reproduction of the paper's figures.
Record actual target-machine commands and artifacts before accepting results.
