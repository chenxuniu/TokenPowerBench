# GH200 environment and measurements

TokenPowerBench supports single-node vLLM inference on GH200 through the
container configuration below. The reference measurements dated **2026-09-14 UTC**
exercise GPU and node monitoring, process permissions, first-token timing, and
ordinary batching. They use functional workloads and do not reproduce the paper's
numerical results.

## Environment

| Item | Configuration |
| --- | --- |
| GPU selection | GPU 0: NVIDIA GH200 144G HBM3e, selected from a two-GPU node |
| CPU | NVIDIA Grace, ARM/aarch64; Intel RAPL unavailable |
| NVIDIA driver | `580.173.02` |
| NVIDIA base image | `nvcr.io/nvidia/vllm:25.09-py3` |
| Package image | [`docker/Dockerfile.gh200`](../docker/Dockerfile.gh200): TokenPowerBench and `ipmitool` |
| PyTorch | `2.9.0a0+50eac811a6.nv25.9` |
| vLLM | `0.10.1.1+381074ae.nv25.9.cu130` |
| Qwen2.5-0.5B-Instruct revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Qwen2.5-7B-Instruct revision | `a09a35458c702b33eeacc393d103063234e8bc28` |

The image installs the Python package while preserving the supplied NVIDIA
inference dependencies. GPU metrics cover the selected device; IPMI covers the
whole node, including the second GPU and other components. Intel RAPL CPU/DRAM
metrics are unavailable on Grace, including under root.

## Repeating the checks

Use Linux with the NVIDIA container runtime and an account authorized to run
Docker. Prepare a directory containing `code/` (the package source), `models/`
(pinned local snapshots), and `prompts.json`. The
[reproduction guide](reproducing.md#3-download-a-fixed-model-revision) provides
model downloads and this directory layout. Run the following in Bash, replacing
the directory placeholder:

```bash
TPB_VALIDATION=/path/to/tokenpowerbench-validation
TPB_IMAGE=tokenpowerbench:1.0.0-gh200
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

The helper runs the installed library from `/results` without mounting source
code. `sudo docker` provides daemon access; `--user` determines the benchmark
process's identity. The host passwd/group entries and separate writable home/cache
mounts support Python user lookup and inference compilation caches.

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
    -v "$TPB_VALIDATION/models:/models:ro" \
    -v "$TPB_VALIDATION/prompts.json:/inputs/prompts.json:ro" \
    -v "$TPB_VALIDATION/results:/results" \
    -v "$TPB_VALIDATION/cache/$tpb_mode:/cache" \
    -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
    -e XDG_CACHE_HOME=/cache/xdg \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    -e TRITON_CACHE_DIR=/cache/triton \
    -e VLLM_CACHE_ROOT=/cache/vllm \
    -e HF_HOME=/cache/huggingface \
    --workdir /results --entrypoint python "$TPB_IMAGE" -m tokenpowerbench "$@"
}

tpb_run nonroot --check-monitor --monitor auto
tpb_run nonroot --check-monitor --monitor full_node
tpb_run root --check-monitor --monitor full_node
```

The helper maps only a dedicated writable directory at each process's home path;
it does not mount the host user's actual home. Root and non-root caches remain
separate. Keep compilation-cache mount paths stable: cached kernels can contain
absolute paths. Use a fresh cache directory when changing container mount paths
or the inference software stack. If `/dev/ipmi0` is absent, omit that device mapping for GPU-only use.
Host sensor permissions still apply inside the container; `--privileged` is not
required for this setup.

| Process / monitor | Reference exit code | Sensor result |
| --- | --- | --- |
| Non-root / `auto` | `0` | GPU available; IPMI and Intel RAPL unavailable |
| Non-root / `full_node` | `1` | IPMI device unreadable |
| Root / `full_node` | `0` | IPMI available; Intel RAPL unavailable |

The reference IPMI device had mode `0600` and was owned by root. A non-root
account with delegated access may instead pass `full_node`. Account for an
expected permission failure when running these checks in a script with `set -e`.

The example prompts are the two raw strings in
[`examples/prompts.json`](../examples/prompts.json). For serial phase profiling:

```bash
tpb_run nonroot \
  --model /models/Qwen2.5-0.5B-Instruct \
  --prompts-file /inputs/prompts.json \
  --monitor gpu_only --phase-profiling --batch-sizes 1 \
  --num-samples 3 --output-tokens 512 \
  --max-model-len 4096 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.2 --seed 42 --temperature 0 \
  --output-dir /results/nonroot-phases
```

For ordinary batching, omit `--phase-profiling`, use
`--batch-sizes 1,2 --num-samples 4`, and choose a separate output directory.
For node measurements, use `tpb_run root` and `--monitor full_node`. CPU/DRAM
remain `null` on Grace. The output-token limit is a maximum; report actual counts.

## Installed Python library verification

TokenPowerBench **1.0.0** was installed into the image and imported from Python's
`dist-packages`, with `/tmp` as the working directory. Inference runs mounted
models, inputs, results, and caches; they did not mount the package source.
All 21 installed Python source files matched the corresponding package sources.
The NVIDIA PyTorch and vLLM versions in the environment table were preserved.
The installed package passed **93 unit tests** on GH200. A separate clean base
installation also passed all 93 tests without inference dependencies.

One non-root Python process called `benchmark()` twice: first with serial phase
profiling, then with batch sizes 1 and 2. Both calls used the two example prompts
and Qwen2.5-0.5B-Instruct, with maximum model length 4096, memory utilization 0.2,
seed 42, and temperature 0. A separate root process used the installed
`tokenpowerbench` command with IPMI monitoring.

| Interface / workload | Requests / output tokens | Duration (s) | GPU energy (J) | IPMI node energy (J) |
| --- | --- | --- | --- | --- |
| Python, non-root, serial phases | `2` / `1024` | `6.915274` | `1081.333894` | `null` |
| Python, non-root, batch 1 | `2` / `256` | `1.972889` | `301.633332` | `null` |
| Python, non-root, batch 2 | `2` / `256` | `1.047514` | `162.493322` | `null` |
| CLI, root, serial phases | `2` / `1024` | `7.928388` | `1230.533193` | `4038.019955` |

Each Python call returned with **zero active multiprocessing children**. NVML
reported 655.0 MiB of device memory used before the calls, 691.4 MiB after the
first return, and 693.4 MiB after the second. CUDA runtime allocations can remain
in the calling process after model workers exit. After all containers exited,
`nvidia-smi` reported 0 MiB and 0% utilization on both GPUs.

The Python phase run recorded TTFT values of **33.1 and 61.2 ms**; the root CLI
run recorded **29.5 and 49.1 ms**. All four prefill windows retained timing and
`null` GPU/IPMI energy under the resolution policy. Decode energy was available;
CPU/DRAM RAPL remained `null` on Grace. The command and module entry points both
reported version `1.0.0`, and installed runs recorded no unrelated Git revision.

Independent integration of the saved readings matched all 8 numeric GPU windows
and 3 numeric IPMI windows within **2.28e-13 J**. The root trace contained 81 GPU
samples and 9 IPMI responses with 8 distinct BMC timestamps; the final boundary
reading repeated a BMC timestamp. Host samples remained strictly ordered. These
checks verify saved-window arithmetic and preserve sensor diagnostics; BMC
timestamps alone do not establish physical power-sensor refresh or phase resolution.

## Reference phase and batch measurements

All runs below used one selected GPU, seed 42, and temperature 0. The small-model
runs used maximum model length 4096 and memory utilization 0.2. The long-context
7B run used maximum model length 32768 and memory utilization 0.5.

| Workload | Requests / output tokens | Duration (s) | GPU energy (J) | IPMI node energy (J) |
| --- | --- | --- | --- | --- |
| 0.5B, serial, non-root GPU-only | `3` / `1536` | `9.908252` | `1563.680421` | `null` |
| 7B, serial, root node monitoring | `3` / `1536` | `13.819793` | `6604.168904` | `7117.193194` |
| 0.5B, serial, root node monitoring | `20` / `10240` | `74.896266` | `11744.381971` | `39356.756111` |

The non-root small-model TTFT values were **28.1, 48.4, and 30.0 ms**, with decode
durations **3.245, 3.406, and 3.150 s**. All three prefill GPU energy values were
`null` under the sensor resolution policy. Saved identity was `is_root: false`.

The long-context workload supplied **30035 input tokens per request** and generated
512 tokens per request. Its host windows and estimates were:

| Request | TTFT / prefill proxy (s) | GPU prefill estimate (J) | Decode (s) | GPU decode estimate (J) | IPMI samples inside prefill |
| --- | --- | --- | --- | --- | --- |
| 1 | `1.024868` | `577.871339` | `3.573980` | `1640.632151` | `0` |
| 2 | `1.032839` | `548.751090` | `3.546484` | `1633.389224` | `1` |
| 3 | `1.034173` | `547.950077` | `3.606620` | `1655.213967` | `1` |

All three node prefill estimates were `null` because fewer than two IPMI samples
fell inside each window. All 15 IPMI readings across 13.906 seconds were **515 W**;
GPU readings ranged from **395.910 to 679.596 W**. These different sensor traces
cannot establish a physical node-level prefill/decode power transition. The
approximately one-second GPU prefill windows also remain affected by NVML averaging.

Ordinary GPU-only batching used four requests per batch configuration in one process:

| Batch size | Output tokens | Duration (s) | GPU energy (J) | Raw / in-window GPU samples |
| --- | --- | --- | --- | --- |
| 1 | `2048` | `15.334865` | `2392.069863` | `155` / `153` |
| 2 | `2048` | `6.917567` | `1099.098412` | `71` / `69` |

Both results had an empty `phases` list and `phase_status: "not_requested"`.
CPU, DRAM, and node energy remained `null`. Each measurement had its own trace,
strictly increasing timestamps, and no NVML sampling errors.

## Engine configuration and sensor quality

The vLLM configuration for serial phase runs was:

| Setting | Requested | Effective |
| --- | --- | --- |
| `enable_chunked_prefill` | `false` | `true` |
| `enable_prefix_caching` | `false` | `false` |
| `max_num_seqs` | `1` | `1` |

`engine_config.json` preserves both configurations. Each request finished before
the next was submitted, so chunking did not overlap different requests' phases.
TTFT/prefill proxy spans submission to the first host-observed token;
`dispatch_completed_s` is a diagnostic timestamp rather than GPU execution start.
These events include scheduling, queueing, and host overhead.

The 20-request node run retained **750 GPU samples and 76 complete IPMI responses**.
All BMC timestamps were unique and advanced by one second, from 02:10:59 to
02:12:14 UTC. Median host polling interval was **0.999985 s**. Nevertheless,
instantaneous IPMI power took only two values: **517 W for 33 readings**, then
**532 W for 43 readings**. This does not determine physical sensor refresh rate.
All responses reported an activated reading state. Statistics periods are saved
as reported and are not treated as instantaneous refresh periods.

All 20 short prefill windows retained timing and null energy. Nineteen decode
windows contained constant IPMI readings and carried the constant-reading
warning; one crossed the change in returned power. These numeric values are
integrals of returned readings with their diagnostics attached.

Independent integration of the raw traces reproduced the long-context totals
within **1.82e-12 J**, ordinary batch totals within **1.364e-12 J**, and the
20-request GPU/IPMI totals and decode windows within **1.46e-11 J**. This verifies
arithmetic, not physical phase resolution. Intel RAPL requires validation on
compatible hardware; exact GPU kernel boundaries require a profiler.

Keep complete run directories, model/code revisions, package versions or image
identity, and hardware/sensor configuration together. See
[measurement definitions](measurement.md) for energy scope, temporal resolution,
and DCMI interpretation, and [the Python API](api.md) for library usage.
