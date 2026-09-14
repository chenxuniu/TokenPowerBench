# Measurement definitions

TokenPowerBench 1.0 measures single-node vLLM inference through the
`tokenpowerbench` command, `python -m tokenpowerbench`, and the
[Python API](api.md). All three interfaces use the same timing, sensor, and
integration rules.

## Sensor scope and permissions

| Measurement | Source | Access | Scope |
| --- | --- | --- | --- |
| GPU power | NVIDIA NVML through `pynvml` | GPU device/driver access; normally no root | Selected physical GPUs and associated circuitry, including GPU memory |
| CPU package energy | Intel RAPL `energy_uj` | Usually root or administrator-delegated sysfs read access | Exposed CPU package domains |
| DRAM energy | Separate Intel RAPL DRAM domains | The same sysfs access | Exposed DRAM domains, where supported |
| Node power | `ipmitool dcmi power reading` | Usually root/local IPMI device access or delegated permission | The BMC's whole-node measurement boundary |

Without access to IPMI or RAPL, GPU measurements remain available. Missing sensor
values are `null`, never zero. GPU energy cannot substitute for node energy.
IPMI and RAPL are probed independently; access to one does not imply access to
the other. The package reads sensors without invoking `sudo` or changing permissions.

Root does not create a missing hardware interface. GH200 uses a Grace ARM CPU
with no Intel RAPL interface; its CPU/DRAM RAPL fields stay `null`, including
under root. Readable IPMI can still provide whole-node measurements.

| Monitor | Behavior |
| --- | --- |
| `gpu_only` | Collect NVML readings; do not probe IPMI or RAPL |
| `auto` | Collect GPU and independently probe optional IPMI/RAPL sensors |
| `full_node` | Require a valid IPMI reading; CPU/DRAM RAPL remain optional |

`--check-monitor` reports process identity and sensor availability without loading
a model. `check_environment()` provides the same information to Python callers.
Benchmark artifacts save it in `runtime.json`, `environment.json`, and
`capabilities.json`.

- `uid`/`gid` identify the real process user and group.
- `euid`/`egid` identify effective user and group.
- `is_root` means effective UID 0; it is `null` when that concept is unsupported.
- `platform` and `machine_architecture` identify the operating system and CPU architecture.

Container UID 0 describes the process inside the container; host sensor devices
must also be accessible. A non-root process can use administrator-delegated access.
The sensor probe determines availability separately from process identity.

NVML reads physical-device power, not power attributable to an individual process
or kernel. Set `CUDA_VISIBLE_DEVICES` before launch; the benchmark maps visible
CUDA devices to NVML UUIDs. Explicit tensor parallelism must equal the visible
GPU count. Other workloads on the selected GPUs affect their readings, and other
workloads anywhere on the node affect IPMI readings.
[NVML device-query reference](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html)

RAPL CPU totals use package counters. Core and uncore subdomains overlap their
parent package and are excluded from that sum. DRAM is reported separately.
Counter wraparound uses `max_energy_range_uj` where available; CPU power is energy
change divided by elapsed time.
[Linux powercap/RAPL interface](https://www.kernel.org/doc/html/latest/power/powercap/powercap.html)

`system_*` and `total_energy_j` refer to IPMI node measurements. Never add GPU,
CPU, or DRAM energy to that total: IPMI already includes components within the
BMC's boundary. `components_energy_j` sums available component measurements; it
is not a substitute for unavailable IPMI energy. Document the server's BMC power
boundary and PSU treatment when comparing machines.

## Prefill and decode timing

Enable `--phase-profiling --batch-sizes 1`, or use
`benchmark(..., phase_profiling=True, batch_sizes=(1,))`, for serial phase measurements.
The engine observes incremental outputs and identifies the first generated token
ID, including a token whose decoded text is empty. The final event is the
request-finished output. An engine without the required incremental interface,
or a first observation containing multiple tokens, raises a phase-timing error.

| Event | Host observation |
| --- | --- |
| `submitted_s` | Immediately before `add_request`; also recorded as `prefill_start_s` |
| `dispatch_completed_s` | Immediately after `add_request` returns |
| `first_token_s` | After the engine step that first exposes one generated token ID |
| `finished_s` | After the step that reports request completion |

```text
submitted       dispatch completed          first token              finished
    |                    |                       |                      |
    |<------------- TTFT / prefill proxy -------->|<------ decode ----->|
    |<----------------------- request latency ------------------------->|
```

TTFT and prefill proxy have the same submission-to-first-token duration. They
include tokenization, queueing, dispatch, scheduling, first-token processing,
and host overhead. A background engine can execute before dispatch returns;
`dispatch_completed_s` therefore does not define GPU execution start.
No host event identifies an exact GPU kernel boundary.

Decode begins at the first observed token and ends at completion. The first
generated token belongs to the prefill proxy; decode energy per token uses
`output_tokens - 1`. A request producing one token can have an empty decode window.
Actual generated token-ID counts supply token denominators; the configured output
limit is a maximum, and early stopping can produce fewer tokens.

Each request finishes before the next is submitted. Ordinary batched inference
reports whole-run measurements and `phase_status: "not_requested"`; concurrent
requests can mix prefill and decode in one power trace. Phase attribution through
this interface is available for vLLM. Preemption, recomputation, and exact kernel
boundaries require additional engine/profiler instrumentation to inspect.

Phase mode requests disabled prefix caching, disabled chunked prefill, and
`max_num_seqs=1`. vLLM may override requested settings. `engine_config.json`
records `requested`, `effective`, and `engine_class`; unavailable effective fields
are `null`. Single-request chunking still lies within the same prefill proxy
window and does not introduce overlap with another request.

## Integration and resolution

Energy is the trapezoidal integral of the timestamped power trace over the
requested window. Boundary values use linear interpolation. Samples must cover
both endpoints, and explicit invalid readings cannot be bridged. No extrapolation,
percentage-based edge trimming, or idle-power subtraction is applied. RAPL energy
uses interpolated cumulative counters with the same window coverage requirement.
All timing and sampling use the host's `time.perf_counter` monotonic clock.

| Sensor | Software polling interval | Minimum accepted window |
| --- | --- | --- |
| GPU NVML | 100 ms | 1 second, plus at least two samples inside the window |
| IPMI | 1 second | 1 second, plus at least two samples inside the window |
| Intel RAPL | 500 ms | 500 ms, plus at least two samples inside the window |

These are minimum data-availability checks. They do not certify physical sensor
resolution. Average power is accepted window energy divided by window duration.
Each sensor can independently return `null` when its coverage is insufficient.

NVIDIA documents `nvmlDeviceGetPowerUsage` as returning a one-second average on
Ampere except GA100 and on newer GPUs; GA100 and older devices return instantaneous
values. Polling every 100 ms does not establish 100 ms physical resolution. The
GPU monitor conservatively applies its one-second minimum to all devices.
Longer accepted windows can still include averaging across phase boundaries.
[NVML power-usage semantics](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html)

IPMI acquisition latency and BMC refresh/averaging can exceed the nominal polling
interval. Repeated queries do not prove that new physical measurements occurred.
RAPL intervals likewise cannot establish an instantaneous change at a boundary.
A short prefill can have valid timing and unavailable energy; extending its prompt
to obtain a longer window changes the workload and should be recorded.

IPMI samples retain acquisition start/end timestamps and their midpoint, plus a
`dcmi` object containing complete stdout/stderr, return code, BMC timestamp,
statistics period, reading state, and min/max/average fields. Integration uses
only the instantaneous-power field. BMC timestamps are separate from the host
monotonic clock; statistics periods do not establish instantaneous refresh rates.
Explicitly deactivated readings are unavailable. Constant readings across a valid
window produce a warning while retaining the integral: stable power, slow updates,
and stale readings cannot be distinguished from equal returned values alone.

## Result files and interpretation

A successful benchmark returns a `BenchmarkResult` and writes a unique
`local_<timestamp>_<id>/` directory under its output directory. The CLI prints
that location. Both interfaces use this artifact structure:

| File | Contents |
| --- | --- |
| `config.json` | Run parameters and ordered-prompt SHA-256 |
| `engine_config.json` | Requested/effective engine settings and engine class |
| `prompts.json` | Exact ordered requests, including repetitions |
| `runtime.json` | Process identity and architecture |
| `environment.json` | Runtime, versions, platform, available repository state, CUDA visibility |
| `capabilities.json` | Sensor availability, devices, and probe failures |
| `status.json` | Running, completed, failed, or interrupted state |
| `batch_<size>_power_samples.json` | Raw GPU/RAPL/IPMI trace and DCMI metadata |
| `batch_<size>_result.json` | Whole-run energy, request phases, and phase status |
| `results.json` | Combined results for completed batch configurations |

Each phase contains timestamps plus `prefill_proxy_energy` and `decode_energy`.
An empty decode window is marked explicitly. Completion does not imply every
sensor or phase is measurable. Preserve `null` fields and warnings in downstream
analysis, and report how many requests had measurable energy; summing only
measurable phases does not give total phase energy for all requests.

For an experiment, retain the exact prompts, model/data revisions, code revision,
package versions or image identity, selected GPU UUIDs, and sensor permissions.
Use an otherwise idle node. Independently recompute integrals from raw traces,
and distinguish arithmetic validation from physical sensor resolution. See the
[reproduction guide](reproducing.md) for runnable commands and
[GH200 environment and measurements](gh200-validation.md) for observed results.
