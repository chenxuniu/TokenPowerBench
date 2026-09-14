# Single-node power and prefill/decode review

Review baseline: Git commit `ec00f52`. All source line references below point to
that commit so subsequent edits do not change the evidence. This review covers
permissions, energy accounting, first-token observations, and phase windows. It
does not assess multi-node performance or establish that the paper's experimental
data have the same issues. Successful inference and sensor collection are
separate acceptance criteria from correct phase attribution.

## Findings and changes

| Priority | Baseline evidence | Effect on reproduction | Current fix |
| --- | --- | --- | --- |
| P1 | [`run_single_node.py:110–123`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/run_single_node.py#L110-L123); [`SingleNode/llm_benchmark.py:150–171`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/SingleNode/llm_benchmark.py#L150-L171) | The single-node entry points record only the start and end of complete inference, without a first-token timestamp. They cannot support a prefill/decode power split. | The maintained single-node vLLM path observes each request's first token through incremental engine steps and records both phase windows. Batch size 1 is required. |
| P1 | [`MultipleNode/vllm_engine.py:481–494`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/MultipleNode/vllm_engine.py#L481-L494) | `first_token_generated` is recorded after blocking `generate()` returns, when the entire response has already been generated. That field does not measure TTFT. | This remains an explicit limitation of the legacy multi-node path. The new single-node path timestamps the first returned token ID; no fix is claimed for the old multi-node scripts. |
| P1 | [`tokenpowerbench/energy/gpu_monitor.py:23–34, 87–109`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/tokenpowerbench/energy/gpu_monitor.py#L87-L109); [`SingleNode/power_monitor.py:377–394`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/SingleNode/power_monitor.py#L377-L394) | Samples have no retained timestamps. The purported time filtering drops 10% from each end and multiplies average power by duration. It cannot precisely exclude startup or idle tails, or align the trace with the first token. | Power samples retain monotonic-clock timestamps and are integrated over the actual requested intervals. Insufficient phase samples produce `null`, and raw traces are retained. |
| P1 | [`tokenpowerbench/energy/full_node_monitor.py:136–139, 294–298`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/tokenpowerbench/energy/full_node_monitor.py#L294-L298) | IPMI uses `_robust_mean` with a default 1000 W upper limit. Valid node readings above 1000 W on multi-GPU servers can all be discarded and ultimately reported as 0. | The generic upper limit no longer filters node readings. Sensor failures and actual 0 W readings are represented separately. |
| P1 | [`tokenpowerbench/energy/full_node_monitor.py:122–128, 191–212`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/tokenpowerbench/energy/full_node_monitor.py#L122-L128) | RAPL discovery includes package and core/uncore subdomains, but all non-DRAM domains are summed. CPU energy can be counted more than once. | CPU totals include only package domains. Independent DRAM domains are reported separately, and counter wraparound uses the sensor's reported counter range. |
| P2 | [`tokenpowerbench/energy/__init__.py:65–72`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/tokenpowerbench/energy/__init__.py#L65-L72); [`tokenpowerbench/energy/full_node_monitor.py:117–139, 237–240`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/tokenpowerbench/energy/full_node_monitor.py#L117-L139) | `auto` depends only on RAPL, missing cases where IPMI is readable but RAPL is not. `full_node` can silently lack sensors, and read failures become 0. | The process identity is reported before independent RAPL/IPMI probes. `gpu_only` skips both probes. `full_node` requires successful IPMI access; CPU/DRAM remain independently optional. Missing values are `null`, with capabilities and reasons recorded. |
| P2 | [`tokenpowerbench/energy/gpu_monitor.py:49–51`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/tokenpowerbench/energy/gpu_monitor.py#L49-L51) | NVML enumerates every GPU on the host while inference uses CUDA-visible devices. Restricting `CUDA_VISIBLE_DEVICES` can therefore still include another job's GPU in energy measurements. | NVML devices are selected using the UUIDs visible to the inference process, and their identities are saved. |
| P2 | [`SingleNode/llm_benchmark.py:88, 147–163`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/SingleNode/llm_benchmark.py#L147-L163); [`SingleNode/power_monitor.py:307–318`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/SingleNode/power_monitor.py#L307-L318) | Legacy scripts reuse a monitor, but each stop shuts down NVML without reinitializing it for the next batch. Read failures then become 0, potentially corrupting later batches' GPU energy. | The maintained monitor supports repeated start/stop cycles and separates final shutdown from stopping sampling. Legacy single-node entry points use the corrected monitor through compatibility adapters. |

P1 findings can distort the main measurement meaning or values and should be
resolved before relying on the code for reproduction. P2 findings affect
correctness under particular permission, device-selection, or repeated-run
conditions. The fixes cover `run_single_node.py`, `tokenpowerbench/`, and the
legacy single-node monitoring adapters; they are not a complete rewrite of the
historical four-engine implementation.

## The two central measurement requirements

The permission documentation must state the original acquisition method
clearly: the author polled node power through IPMI once per second and read CPU
energy counters through Intel RAPL sysfs. These interfaces usually require root
or administrator-delegated permissions. Without access to either interface, an
ordinary user obtains only NVML GPU measurements. GPU energy or a sum of
components cannot substitute for IPMI node energy, and unreadable CPU/node
values must not become 0. GPU/CPU/DRAM energy must not be added to IPMI energy,
since that would count components already inside the node measurement twice.

Each launch, including `--check-monitor`, first reports real UID/GID, effective
UID/GID, and `is_root`; benchmark runs save these in `runtime.json`. `is_root`
depends only on whether the effective UID is 0. Actual sensor capabilities are
probed and recorded separately. GH200's Grace ARM CPU does not expose Intel
RAPL, so these CPU/DRAM readings remain unavailable even as root. Readable IPMI
is sufficient to collect node energy with `full_node`; missing RAPL must not
block a valid whole-node measurement.

The original single-node implementation lacks an auditable prefill/decode
boundary. Another directory contains `first_token_generated`, but records it
at the wrong point. The corrected path defines the first token using token IDs
returned incrementally, including IDs whose decoded text is empty.
`submitted_s` is recorded before calling `add_request` and also serves as
`prefill_start_s`: the interval from that timestamp to the first observed token
is both TTFT and the prefill proxy. Decode spans first token to completion.
`dispatch_completed_s` records the return from `add_request` only as a diagnostic
event. A background engine may already execute before dispatch returns, so that
return cannot be treated as GPU execution start or used to exclude dispatch
energy. The window includes submission, queueing, scheduling, host processing,
and first-token sampling overhead; it is not pure GPU prefill-kernel time.

Phase mode requires batch size 1 and finishes each request before submitting
the next, preventing overlap between different requests' prefill/decode stages.
The engine constructor requests disabled prefix caching and chunked prefill,
with `max_num_seqs=1`. The vLLM V1 version validated on GH200 enabled chunked
prefill despite that request; prefix caching remained disabled and
`max_num_seqs=1` remained effective. Consequently, `engine_config.json` records
`requested`, `effective`, and `engine_class` separately. Constructor arguments
alone do not establish that chunked prefill is disabled. The serial
submission-to-first-token window still includes chunked prefill without phase
overlap between requests; it does not prove the absence of internal preemption
or recomputation. Ordinary batched throughput tests remain available without
unsupported phase attribution.

## Sensor resolution remains an experimental limitation

Correct timestamps do not make once-per-second IPMI sampling resolve short
prefill phases. The monitor independently counts valid in-window readings for
each sensor and declines phase energy when fewer than two are available. This
checks data availability, not whether the underlying sensor actually resolved
changes within the phase.

NVML polling cadence also differs from physical sensor resolution. NVIDIA
documents `nvmlDeviceGetPowerUsage` as returning one-second average power on
Ampere other than GA100 and on newer architectures; GA100 and earlier
architectures return instantaneous values. Polling every 100 ms can therefore
still mix short phases through the sensor's averaging window. RAPL counter
differences likewise represent averages over the interval between reads.
[NVIDIA power-query documentation](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html)

The GPU monitor currently applies a conservative policy: until a device's
effective resolution is certified, all GPU energy windows shorter than one
second return `null`, even if they contain multiple readings. The capability
report explains this policy. It does not assert that GA100 hardware necessarily
uses one-second averaging. Phase timestamps remain available.

New results should retain phase durations, sample counts, sensor types, and
sampling configuration, making unavailable estimates explicit. Repetition can
reduce random variation but cannot recover temporal changes the hardware did
not observe. Longer prompts may improve IPMI coverage of prefill, but they
change the workload and must be documented.

## Validation status

Local mock tests cover engine events, process identity, sensor permissions,
counter wraparound, window integration, and missing values. On 2026-09-14 UTC,
real non-root vLLM phase inference completed on GH200, and the permission matrix
passed: non-root `auto` reported GPU capability, non-root `full_node` failed
explicitly on IPMI permissions, and root `full_node` read IPMI while CPU/DRAM
RAPL remained unavailable. Short-prefill timestamps were recorded, with GPU
phase energy returning `null` under the resolution policy.

The root long-input phase run and ordinary batch-size 1 and 2 regression runs
also completed. Independent integration of the raw traces matched the reported
results. Long-input prefill lasted approximately 1.03 seconds, still with too
few IPMI readings inside each window, and every IPMI reading in that run was
515 W. These functional checks therefore do not establish a physical difference
between node prefill and decode power. See the
[GH200 validation report](gh200-validation.md) for the environment, results,
and sensor limitations. Intel RAPL hardware behavior remains unvalidated because
Grace does not provide that interface. Equivalence between host first-token
observations and exact GPU kernel-profiler boundaries is also unvalidated.
Passing functional checks does not remove sensor-averaging limitations.

Executable single-node acceptance steps and the complete measurement contract
are in [measurement.md](measurement.md). These changes do not establish
reproduction of the paper's figures. Findings in the repository baseline cannot
determine which scripts, records, or postprocessing the author actually used for
the original experiments.
