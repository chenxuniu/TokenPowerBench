# GH200 functional validation

Validation date: **2026-09-14 UTC**. These checks exercise process identity,
sensor permissions, real vLLM first-token events, and energy integration. Their
workloads and results do **not** reproduce the paper's figures.

## Environment

| Item | Observed configuration |
| --- | --- |
| GPU used | One NVIDIA GH200 144G HBM3e; GPU 0 selected from a two-GPU node |
| CPU architecture | NVIDIA Grace, ARM/aarch64; no Intel RAPL interface |
| NVIDIA driver | `580.173.02` |
| Base image | `nvcr.io/nvidia/vllm:25.09-py3` |
| Image additions | `ipmitool`, through [`docker/Dockerfile.gh200`](../docker/Dockerfile.gh200) |
| PyTorch | `2.9.0a0+50eac811a6.nv25.9` |
| vLLM | `0.10.1.1+381074ae.nv25.9.cu130` |
| Models | Local Qwen2.5-0.5B-Instruct and Qwen2.5-7B-Instruct weights |

The downloaded model metadata records revisions
`7ae557604adf67be50417f59c2c2f167def9a775` for `Qwen/Qwen2.5-0.5B-Instruct` and
`a09a35458c702b33eeacc393d103063234e8bc28` for `Qwen/Qwen2.5-7B-Instruct`.
See [model preparation](reproducing.md) for pinned downloads.

Only the selected GPU is included in GPU energy. IPMI reports the whole node,
including components beyond that GPU. The node's two-GPU topology therefore
matters when comparing these scopes.

## Completed checks

`--check-monitor` printed process identity before attempting sensor access.
The local IPMI device was owned by root with mode `0600`; the non-root failure
was a real device-permission failure. Passing `/dev/ipmi0` into the container
was sufficient for root IPMI access; `--privileged` was not required.

| Process / mode | Exit code | Observed result |
| --- | --- | --- |
| Non-root, `--check-monitor --monitor auto` | `0` | GPU available; IPMI unavailable; Intel RAPL unavailable |
| Non-root, `--check-monitor --monitor full_node` | `1` | Expected explicit failure because IPMI was unreadable |
| Root, `--check-monitor --monitor full_node` | `0` | IPMI available; CPU/DRAM still unavailable because Intel RAPL is absent |

This confirms that `is_root` and sensor availability are independent:
`full_node` requires IPMI, while CPU/DRAM RAPL measurements remain optional.
The absence of Intel RAPL on Grace is not corrected by root permissions.

A real **non-root, GPU-only phase run** completed three serial Qwen2.5-0.5B-Instruct
requests. The saved runtime reported `is_root: false`.

| Quantity | Observed value |
| --- | --- |
| Requests completed | `3` |
| Total generated tokens | `1536` |
| Measured run duration | `9.908 s` |
| Selected-GPU energy over the measured run | `1563.68 J` |
| TTFT / prefill proxy, request order | `28.1 ms`, `48.4 ms`, `30.0 ms` |
| Decode duration, request order | `3.245 s`, `3.406 s`, `3.150 s` |
| Prefill GPU energy | `null` for every request, as expected for these short windows |

The first-token observations preceded completion and separated the serial
requests' prefill proxy and decode windows. TTFT and prefill proxy both span
`submitted_s` to `first_token_s`. `dispatch_completed_s` is an auxiliary host
event, not a GPU execution-start measurement. Values above are rounded; the
saved JSON events and raw samples retain greater precision.

The installed vLLM enabled chunked prefill despite the request to disable it:

| Setting | Requested | Effective |
| --- | --- | --- |
| `enable_chunked_prefill` | `false` | `true` |
| `enable_prefix_caching` | `false` | `false` |
| `max_num_seqs` | `1` | `1` |

`engine_config.json` records both configurations and the engine class. The
runner finished each request before submitting the next, so chunking did not
introduce overlap between different requests' phases. These observations still
describe host windows, including queueing and host overhead; they do not
identify exact GPU kernel boundaries.

## Completed root run and trace audit

The root `full_node` phase run completed with Qwen2.5-7B-Instruct and long
inputs. The saved runtime reported `is_root: true`; IPMI was available and
Intel RAPL CPU/DRAM metrics remained `null`.

| Quantity | Observed value |
| --- | --- |
| Requests completed | `3` |
| Input tokens per request | `30035` |
| Output tokens per request / total | `512` / `1536` |
| Measured run duration | `13.819793 s` |
| Selected-GPU energy | `6604.168904 J` |
| Integrated IPMI node reading | `7117.193194 J` |

| Request | TTFT / prefill proxy (s) | GPU prefill estimate (J) | Decode (s) | GPU decode estimate (J) | IPMI samples inside prefill |
| --- | --- | --- | --- | --- | --- |
| 1 | `1.024868` | `577.871339` | `3.573980` | `1640.632151` | `0` |
| 2 | `1.032839` | `548.751090` | `3.546484` | `1633.389224` | `1` |
| 3 | `1.034173` | `547.950077` | `3.606620` | `1655.213967` | `1` |

An independent audit of the saved root-run trace and reported integrals
passed, with maximum absolute energy difference **`1.82e-12 J`**. This verifies
the arithmetic on the recorded readings, not the physical sensor's ability to
resolve a phase boundary.

All **15 IPMI readings across 13.906 seconds were exactly 515 W**, while the
selected GPU's reported power ranged from **395.910 to 679.596 W**. The IPMI
prefill sample counts were `0`, `1`, and `1`, below the required two readings;
all three prefill node-energy values correctly remained `null`. The valid
whole-run IPMI integral is therefore the integral of a constant returned
reading. It does not demonstrate a measured physical difference between node
prefill and decode power.

The approximately 1.03-second prefill windows passed the GPU monitor's
minimum-duration and sample-coverage checks, so GPU energy values were
reported. They remain heavily affected by NVML's approximately one-second
averaging window: the reported prefill/decode estimates cannot establish a
sharp physical GPU power transition either. Temporal resolution is a separate
limitation from correct timestamps and integration.

## Completed ordinary batch and monitor-reuse check

A non-root GPU-only Qwen2.5-0.5B-Instruct run completed batch sizes 1 and 2
sequentially in the same process:

| Batch size | Responses | Output tokens | Duration (s) | GPU energy (J) | Raw / in-window GPU samples |
| --- | --- | --- | --- | --- | --- |
| 1 | `4` | `2048` | `15.334865` | `2392.069863` | `155` / `153` |
| 2 | `4` | `2048` | `6.917567` | `1099.098412` | `71` / `69` |

An independent recomputation used explicit endpoint interpolation followed by
trapezoids over the clipped raw trace, without calling the production
integrator. The absolute differences were `1.364e-12 J` and `0 J` for batch
sizes 1 and 2 respectively. Each trace had finite, strictly increasing
timestamps; the second trace started after the first ended and contained no
retained samples from the earlier batch. Neither batch reported NVML sampling
errors.

Both batch results retained `phase_status: "not_requested"` and an empty
`phases` list. CPU, DRAM, node, and total-node energy stayed `null`; the scope
was `selected_gpus`. These checks confirm monitor reuse and scope handling for
this tested configuration.

## Completed IPMI diagnostics follow-up

The final sampler retains complete DCMI responses, including the BMC timestamp,
statistics period, and reading state. A further root Qwen2.5-0.5B-Instruct phase
run exercised this code with **20 requests / 10240 output tokens** over
**74.896266 seconds**. It reported **11744.381971 J** from the selected GPU and
**39356.756111 J** from IPMI.

Independent integration of all 750 GPU and 76 IPMI samples reproduced the
whole-run and all 20 decode-window estimates. The maximum absolute difference
was `1.46e-11 J` for GPU energy and `0 J` for IPMI energy.

All 76 IPMI responses succeeded and reported an activated reading state. Their
BMC timestamps advanced by one second on each query; the median host polling
interval was **0.999985 seconds**. The returned instantaneous power nevertheless
had only two values: 517 W for 33 readings, then 532 W for 43 readings. The
statistics period was retained as reported, without treating it as the power
sensor's refresh period. This trace does not establish the physical sensor's
refresh or averaging behavior.

The 20 short prefill windows retained timestamps and null energy. Nineteen
decode windows had identical IPMI readings throughout their coverage and
correctly received the new constant-reading warning; the remaining window
crossed the change in the returned power. Numeric integrals remain estimates
over the returned readings, with these diagnostics attached.

## Validation boundaries

The completed runs and independent integration audits establish functional
behavior. No physical node-level prefill/decode power difference has been
validated. Intel RAPL hardware behavior remains untested on this ARM machine,
and equivalence to exact GPU kernel-profiler boundaries remains unvalidated.
All 75 automated tests passed locally and in the GH200 container. The Python
wheel built successfully and its installed single-node command was checked
outside the source checkout. The test containers exited and both GPUs returned
to zero allocated memory after validation.

## Repeating the checks

The following Bash recipe uses a task-specific validation directory containing
`code/` (this checkout), `models/` (local model directories), and `prompts.json`
(a JSON array of prompt strings). Install the NVIDIA container runtime on the
host and use an account authorized to run Docker. Local model weights must
already be present. Replace the validation-directory placeholder with your own
path.

```bash
TPB_VALIDATION=/path/to/tokenpowerbench-validation
TPB_IMAGE=tokenpowerbench-gh200:25.09
TPB_UID="$(id -u)"
TPB_GID="$(id -g)"
TPB_USER_HOME="$(getent passwd "$TPB_UID" | cut -d: -f6)"
TPB_ROOT_HOME="$(getent passwd 0 | cut -d: -f6)"

test -n "$TPB_USER_HOME" && test -n "$TPB_ROOT_HOME"
mkdir -p "$TPB_VALIDATION/results" \
  "$TPB_VALIDATION/nonroot-home" "$TPB_VALIDATION/root-home" \
  "$TPB_VALIDATION/cache/nonroot" "$TPB_VALIDATION/cache/root"
cp "$TPB_VALIDATION/code/examples/prompts.json" "$TPB_VALIDATION/prompts.json"

sudo docker build \
  -f "$TPB_VALIDATION/code/docker/Dockerfile.gh200" \
  -t "$TPB_IMAGE" "$TPB_VALIDATION/code"
```

The helper below switches the **container process** between the current user's
numeric UID/GID and root. `sudo docker` grants access to the Docker daemon;
`--user` determines the identity recorded by the benchmark.

```bash
tpb_run() {
  local tpb_mode="$1"
  shift
  local tpb_identity tpb_user_home tpb_home_mount
  case "$tpb_mode" in
    nonroot)
      tpb_identity="$TPB_UID:$TPB_GID"
      tpb_user_home="$TPB_USER_HOME"
      tpb_home_mount="$TPB_VALIDATION/nonroot-home"
      ;;
    root)
      tpb_identity=0:0
      tpb_user_home="$TPB_ROOT_HOME"
      tpb_home_mount="$TPB_VALIDATION/root-home"
      ;;
    *) return 2 ;;
  esac
  sudo docker run --rm \
    --gpus '"device=0"' --shm-size=16g \
    --user "$tpb_identity" --device /dev/ipmi0 \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v "$tpb_home_mount:$tpb_user_home" \
    -v "$TPB_VALIDATION/code:/workspace:ro" \
    -v "$TPB_VALIDATION/models:/models:ro" \
    -v "$TPB_VALIDATION/prompts.json:/inputs/prompts.json:ro" \
    -v "$TPB_VALIDATION/results:/results" \
    -v "$TPB_VALIDATION/cache/$tpb_mode:/cache" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e HF_HUB_OFFLINE=1 \
    -e XDG_CACHE_HOME=/cache/xdg \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    -e TRITON_CACHE_DIR=/cache/triton \
    -e VLLM_CACHE_ROOT=/cache/vllm \
    -e HF_HOME=/cache/huggingface \
    --workdir /workspace "$TPB_IMAGE" run_single_node.py "$@"
}

tpb_run nonroot --check-monitor --monitor auto
tpb_run nonroot --check-monitor --monitor full_node
tpb_run root --check-monitor --monitor full_node
```

The middle command is expected to return exit code `1` on the tested device's
permissions. Run the matrix interactively, or account for that expected failure
in a script using `set -e`. On a different server, delegated device permissions
may allow the non-root `full_node` check to succeed. If the host has no
`/dev/ipmi0`, omit its device mapping for a GPU-only check; that does not create
IPMI availability.

The passwd/group and writable-home mounts address two problems encountered
during validation: a bare numeric container UID without a passwd entry caused
PyTorch's user lookup to fail, and FlashInfer needed a writable user home.
The helper mounts only a dedicated empty directory at the passwd-resolved home
path; it does not mount the host user's actual home. Non-root and root caches
are kept in separate task directories. No `HOME` environment override is used.

The small-model phase and batch checks used the two raw inputs in
[`examples/prompts.json`](../examples/prompts.json), copied during setup above.
The runner repeats them in order to reach `--num-samples`.

```bash
tpb_run nonroot \
  --model /models/Qwen2.5-0.5B-Instruct \
  --prompts-file /inputs/prompts.json \
  --monitor gpu_only --phase-profiling --batch-sizes 1 \
  --num-samples 3 --output-tokens 512 \
  --max-model-len 4096 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.2 \
  --seed 42 --temperature 0 \
  --output-dir /results/nonroot-phases
```

These prompts are a functional smoke workload, not the paper's workload.
The output-token limit is a maximum:
early stopping can produce fewer than 512 tokens. Read actual token-ID counts
from the results. Short prefill windows should retain their timing and return
`null` GPU/IPMI phase energy under the documented resolution policy.

To check ordinary batching, omit `--phase-profiling` and use
`--batch-sizes 1,2 --num-samples 4`, with a separate output directory. To check
whole-node monitoring, launch the helper as `root` and use
`--monitor full_node`; Intel RAPL CPU/DRAM values should remain `null` on Grace.
A long-prefill experiment also needs a sufficiently long local prompt and an
appropriate `--max-model-len`; for the larger model, mount its local weights
as `/models/Qwen2.5-7B-Instruct`. Changing the prompt or model changes the
workload and must be recorded. The completed long-prefill run above illustrates
that a longer window can yield numeric GPU estimates while still leaving node
prefill energy unavailable.

Keep each run's `runtime.json`, `capabilities.json`, `engine_config.json`,
`environment.json`, `prompts.json`, result JSON, and raw power samples together.
Before interpreting a phase estimate, verify coverage and recompute its
integral from the saved trace as described in
[the measurement contract](measurement.md). Raw artifacts can contain local
paths, process IDs, and GPU UUIDs; the tables here intentionally report only
the hardware/software context needed to interpret the checks.
