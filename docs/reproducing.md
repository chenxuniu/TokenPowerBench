# Reproducing a single-node run

TokenPowerBench 1.0 runs single-node experiments with fixed model weights,
saved prompts, explicit sensor scope, and inspectable results. The examples are
functional workloads; they do not reproduce the published figures or their numbers.
The paper is [TokenPowerBench: Benchmarking the Power Consumption of LLM Inference
(AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/40535).

## 1. Get the package source

```bash
git clone --branch v1.0.0 https://github.com/chenxuniu/TokenPowerBench.git
cd TokenPowerBench
mkdir -p results/local_reproduction_environment
git rev-parse HEAD > results/local_reproduction_environment/code-commit.txt
```

Archive the recorded commit SHA and use that revision for later reruns. Keep any
local changes alongside it. Run the installation and model preparation commands
from this directory; the installed CLI works from other directories too.

## 2. Prepare the environment

Use Linux, Python 3.10 or newer, a supported NVIDIA GPU, and a working NVIDIA driver.
For GH200/aarch64, use the [GH200 package image](gh200-validation.md#repeating-the-checks),
based on `nvcr.io/nvidia/vllm:25.09-py3`.
That image supplies the tested NVIDIA software stack. Native installations can
resolve different package versions; record them with each experiment.

For native inference, install the library with vLLM and dataset support:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[vllm,datasets]'
python -m pip freeze > results/local_reproduction_environment/pip-freeze.txt
nvidia-smi > results/local_reproduction_environment/nvidia-smi.txt
```

For NVML monitoring without inference dependencies, install `python -m pip install .`
and use [the Python monitoring API](api.md#check-environment) with explicit physical
GPU indices or UUIDs.

Use the same GPU selection for monitoring and inference. The commands below use
the module CLI; `tokenpowerbench` accepts the same flags. They select one GPU and
set tensor parallelism to one. Keep the node free of other workloads:
IPMI includes the entire node, even when inference uses only one of its GPUs.

## 3. Download a fixed model revision

This public small model and exact revision were used in GH200 validation:

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

The larger validated model is `Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28`. To download it, change the repository,
revision, and local directory together. A different model or prompt changes the
workload. Save model revisions with the run; the CLI receives a local path and
does not automatically recover its Hugging Face revision.

For the Docker path, the following prepares the directory layout used by the
GH200 guide and downloads weights without allocating a GPU. It needs an
account authorized to use Docker and network access during this preparation step.

```bash
TPB_VALIDATION="$PWD/validation-gh200"
mkdir -p "$TPB_VALIDATION/code" "$TPB_VALIDATION/models" "$TPB_VALIDATION/cache/download"
git archive HEAD | tar -x -C "$TPB_VALIDATION/code"
cp examples/prompts.json "$TPB_VALIDATION/prompts.json"
git rev-parse HEAD > "$TPB_VALIDATION/code-commit.txt"
sudo docker run --rm -i --user "$(id -u):$(id -g)" \
  -v "$TPB_VALIDATION/models:/models" \
  -v "$TPB_VALIDATION/cache/download:/cache" -e HF_HOME=/cache/huggingface \
  --entrypoint python nvcr.io/nvidia/vllm:25.09-py3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    local_dir="/models/Qwen2.5-0.5B-Instruct",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
)
PY
```

`git archive` includes committed files only. Continue with the GH200 guide's image
build and `tpb_run` helper, using this `TPB_VALIDATION`. It runs the installed
package with separate writable home and compilation-cache directories.

## 4. Check sensor access

The native commands below print UID/EUID, `is_root`, and actual sensor availability
without loading a model:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tokenpowerbench --check-monitor --monitor auto
CUDA_VISIBLE_DEVICES=0 python -m tokenpowerbench --check-monitor --monitor gpu_only
```

`gpu_only` uses NVML. `auto` also probes IPMI and Intel RAPL, retaining available
sensors. `full_node` requires readable IPMI; CPU/DRAM RAPL are optional. Local IPMI
and RAPL commonly require root or administrator-delegated access. Root does not
create a missing hardware interface: Grace has no Intel RAPL, so CPU/DRAM stay
`null` on the tested GH200 node even under root.

On a host where you are authorized to use root, check with the explicit virtual
environment interpreter so `sudo` does not accidentally select another Python:

```bash
sudo env CUDA_VISIBLE_DEVICES=0 "$PWD/.venv/bin/python" \
  -m tokenpowerbench --check-monitor --monitor full_node
```

For containers, use the GH200 guide's `tpb_run nonroot` and `tpb_run root` checks.
`sudo docker` selects access to the daemon; the container's `--user` selects the
identity recorded by the benchmark. Map the actual IPMI device as documented.

## 5. Run ordinary batching and serial phases

The checked-in [example prompts](../examples/prompts.json) are two raw strings.
The runner repeats them in order to reach the requested sample count. It adds no
chat template. `--output-tokens 512` is a maximum; early stopping can return fewer
tokens. Model loading and a warmup request are outside the measured run.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python -m tokenpowerbench \
  --model models/Qwen2.5-0.5B-Instruct --prompts-file examples/prompts.json \
  --monitor gpu_only --batch-sizes 1,2 --num-samples 4 --output-tokens 512 \
  --max-model-len 4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.2 \
  --seed 42 --temperature 0 --output-dir results/local_reproduce_batches

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python -m tokenpowerbench \
  --model models/Qwen2.5-0.5B-Instruct --prompts-file examples/prompts.json \
  --monitor gpu_only --phase-profiling --batch-sizes 1 --num-samples 3 \
  --output-tokens 512 --max-model-len 4096 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.2 --seed 42 --temperature 0 \
  --output-dir results/local_reproduce_phases
```

For native whole-node measurement, run the second command through
`sudo env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 "$PWD/.venv/bin/python"`, followed
by `-m tokenpowerbench` and the same arguments, replacing `gpu_only` with
`full_node` and choosing a separate output directory. Expect root-owned outputs.
For Docker, the equivalent invocation starts with `tpb_run nonroot` or `tpb_run root`;
use `/models/Qwen2.5-0.5B-Instruct`, `/inputs/prompts.json`, and `/results/...` paths.

Phase profiling requires batch size one. TTFT/prefill proxy spans submission to
the first host-observed token; decode spans that token to completion. Queueing,
tokenization, scheduling, and host overhead are included. `dispatch_completed_s`
does not identify GPU execution start. Read the actual engine settings: the tested
vLLM version enabled chunked prefill despite the requested `false` value.

The GPU monitor polls at 100 ms but rejects windows shorter than one second;
IPMI polls once per second and requires at least two samples within a window.
Both require full boundary coverage. Short prefill energy is therefore normally
`null`. Numeric phase integrals remain sensor-based estimates subject to averaging
and BMC update behavior. See the [measurement contract](measurement.md).

## 6. Inspect and preserve a run

The CLI prints the exact output directory. Substitute it below:

```bash
TPB_RUN=results/local_reproduce_phases/local_REPLACE_WITH_PRINTED_DIRECTORY
python - "$TPB_RUN" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
assert json.loads((run / "status.json").read_text())["status"] == "completed"
for path in sorted(run.glob("batch_*_result.json")):
    result = json.loads(path.read_text())
    assert result["num_responses"] > 0 and result["total_output_tokens"] > 0
    print(path.name, result["energy"]["energy_scope"], result["phase_status"])
    print("GPU J:", result["energy"]["gpu_energy_j"],
          "node J:", result["energy"]["system_energy_j"])
    for event in result["phases"]:
        print("TTFT s:", event["ttft_s"], "decode s:", event["decode_s"])
PY
```

Check `runtime.json`, sensor availability and warnings in `capabilities.json`, and
requested/effective settings in `engine_config.json`. Completion does not promise
every energy value is measurable. Missing values remain `null`, never zero.
`total_energy_j` refers to IPMI node energy; do not add GPU energy to it.

Archive the entire run directory, including raw samples, exact ordered
`prompts.json`, `config.json`, and `environment.json`, plus the code SHA, package
freeze/container image identity, model revision, and hardware inventory. Docker
source archives have no `.git`, so preserve the host-side `code-commit.txt` too.
Rerun using the saved `prompts.json`, original sample count, seed, temperature,
model revision, hardware, and effective configuration. Matching seeds alone does
not guarantee identical timings or energy across environments.

## Relationship to the AAAI paper

[Section 4](https://arxiv.org/html/2512.03024v1#S4) reports an eight-node cluster;
each node has four H100 94 GB GPUs, two Xeon Gold 6426Y CPUs, and 512 GB RAM.
The hardware and coverage differ from the GH200 functional validation:

| Item | Paper setup | GH200 validation |
| --- | --- | --- |
| GPU configuration | 8 nodes × 4 H100 94 GB | One GH200 144G HBM3e selected from a two-GPU node |
| CPU | Two Intel Xeon Gold 6426Y per node | NVIDIA Grace/aarch64; Intel RAPL unavailable |
| Node RAM | 512 GB per node | Not recorded in the validation data |

The paper's figure labels identify these experimental dimensions. The package
supports the following corresponding procedures; it does not
supply a complete original configuration for each figure.

| Paper experiment | Procedure and supported scope |
| --- | --- |
| Figure 2: phase energy across models/engines | Repeat serial phase runs per model; the CLI supports vLLM only. Cross-engine equivalence is unvalidated. |
| Figure 3: contexts 0–2K, 2–5K, 5–10K tokens | Prepare and save a separate prompt file per bin using each model's tokenizer; verify actual `input_tokens` in phase results. CLI word filters do not implement token bins. |
| Figure 4: batches 32–1024 | Use ordinary mode with `--batch-sizes 32,64,128,256,512,1024` and enough requests, such as `--num-samples 1024`; this example sweep requires sufficient memory. |
| Figures 5–6: parallelism and quantization | Multi-node TP/PP and quantization comparisons are outside the supported single-node interface. |

For context sweeps, record exact bin endpoints, tokenizer revision, special-token
handling, dataset revision, and selected examples. Increase `--max-model-len` to
fit input plus requested output; do not substitute word counts or padded duplicates
for the original dataset without recording that workload change. Use identical
saved requests when comparing batches, and omit `--phase-profiling` for batch sweeps.
Published-number reproduction additionally requires the original per-figure
models, prompts, software, precision, parallelism, and run settings; do not infer
missing settings from these smoke examples.
