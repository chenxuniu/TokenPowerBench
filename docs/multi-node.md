# Multi-node inference

The distributed CLI runs vLLM replicas on an existing Ray cluster or a SLURM
allocation. It supports tensor parallelism (TP), pipeline parallelism (PP), and
multiple independent replicas. Validation covers static checks, mocked tests,
and local Ray 2.48 CPU processes with synthetic placement resources. Multi-node
GPU inference, NCCL communication, and SLURM integration require validation on
the target cluster.

Cluster energy is not collected. `cluster_energy_j`, cluster energy per token,
and whole-cluster totals remain `null`. Optional monitoring covers the driver
host only and is saved under `driver_diagnostics`. Distributed requests can
overlap, so this interface does not report prefill/decode energy attribution.

## Prepare every node

Use Linux and the same Python, CUDA-compatible PyTorch/vLLM, Ray, and
TokenPowerBench installation on every node. From a shared repository checkout:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install '.[distributed]'
python -m pip freeze > environment-multi-node.txt
```

The distributed extra includes Ray's compiled-graph dependencies. Check that
CuPy matches the cluster's CUDA environment; installing into an existing NVIDIA
container can require preserving its supplied PyTorch/vLLM packages. A package
installation alone does not validate that combination.

Prepare pinned model snapshots at the same absolute paths on every participating
GPU node, with identical tokenizer/configuration files and weights. A shared
filesystem is the simplest arrangement. A CPU-only driver does not need local
model weights; `--model-dir` constructs the absolute paths used by workers. Use the
[model preparation instructions](reproducing.md#3-download-a-fixed-model-revision)
and save the exact revisions. Check available GPU memory for TP × PP ranks per
replica and `concurrency` replicas; all replicas keep their models resident.

Save prompts as a JSON array of raw strings. `--prompts-file` and `--datasets`
are mutually exclusive. Prompt-file requests repeat in order when `--num-samples`
exceeds the file length; omitted `--num-samples` uses every entry. With a dataset,
the default is 1,000 requests. No chat template, quantization mode, or custom
stop sequence is silently added.

## Configure a cluster profile

The installed `tpbench-multi` command and `python run_multi_node.py` share the
same options. To open the interactive terminal setup:

```bash
tpbench-multi --configure-cluster --cluster-config cluster.json
```

`--cluster-config` selects the JSON output path; the wizard defaults to
`cluster.json` when it is omitted. Existing files are preserved; choose another
path when creating a different profile. Setup does not require `--models` or start
inference. Run setup as a separate command with only the optional profile path;
benchmark arguments belong to the subsequent run. Select one of two connection methods:

| Method | Information entered | What happens at launch |
| --- | --- | --- |
| Manual | Head IPv4 address, optional worker IPv4 addresses, Ray port | The driver connects to the saved head address |
| SLURM | Ray port and optional common network interface | The head and workers are resolved from the allocation's node list |

A manual setup session looks like this; replace the example addresses with
addresses reachable between your nodes:

```text
$ tpbench-multi --configure-cluster --cluster-config cluster.json
TokenPowerBench cluster setup
Cluster mode [1=manual, 2=slurm] (1): 1
Head node IPv4 address: 192.0.2.10
Worker node IPv4 addresses (comma-separated, optional): 192.0.2.11
Ray head port (6379): 6379
```

The saved profile is an inspectable JSON file:

```json
{
  "schema_version": 1,
  "mode": "manual",
  "head_address": "192.0.2.10",
  "head_port": 6379,
  "worker_addresses": ["192.0.2.11"],
  "network_interface": null
}
```

For noninteractive setup, copy and edit the checked-in
[manual profile](../examples/cluster.manual.json) or
[SLURM profile](../examples/cluster.slurm.json). Replace the manual example IPs
with your own reachable node addresses, then pass the edited file with
`--cluster-config`. Preserve the schema's field names and types.

For manual setup, the wizard prints separate Ray startup commands for the head
and each listed worker. Run them in terminals on their corresponding machines,
using the same software environment. The wizard does not use SSH, start services,
or allocate nodes. Worker addresses help organize the startup instructions;
workers join the head, and `ray.init` receives the head address rather than a
list of machine IPs. The saved worker list is an inventory for those instructions;
it does not restrict placement, and all registered cluster resources remain
eligible. After Ray is ready, use the saved profile with the
[driver command](#use-an-existing-ray-cluster).

For SLURM setup, the profile stores dynamic discovery settings. Inside an
allocation with both `SLURM_JOB_ID` and `SLURM_JOB_NODELIST`, the wizard shows
the expanded node list. Outside an allocation,
it provides guidance for inspecting the nodes after scheduling. The first
allocated node is the head and the remaining nodes are workers. This profile
does not pin a particular set of machines and can be reused for later jobs.

For a separate SLURM profile, run:

```bash
tpbench-multi --configure-cluster --cluster-config cluster-slurm.json
```

Select `2` at the mode prompt, choose the port, and optionally enter an interface
at `Network interface (optional, e.g. ib0):`.

The SLURM submission and controller scripts load `--cluster-config`, including
the saved port and interface. The worker count and resource request remain
explicit in the submission command below.

## Submit one SLURM allocation

Run the submission helper from an account authorized to submit GPU jobs.
For two nodes with four GPUs each, this example runs one eight-rank model
replica with TP 4 and PP 2:

```bash
export TPB_PROJECT_DIR="$PWD"
export TPB_PYTHON="$PWD/.venv/bin/python"
export TPB_PARTITION=your_gpu_partition
export TPB_GPUS_PER_NODE=4
export TPB_CPUS_PER_NODE=16
export TPB_TIME=01:00:00

bash scripts/submit_jobs.sh 1 \
  --cluster-config cluster-slurm.json \
  --models Qwen2.5-7B-Instruct --model-dir /shared/models \
  --prompts-file "$PWD/examples/prompts.json" --num-samples 4 \
  --tensor-parallel 4 --pipeline-parallel 2 --concurrency 1 \
  --batch-sizes 1 --max-tokens 128 --max-model-len 4096 \
  --gpu-memory-utilization 0.2 --seed 42 --temperature 0 --top-p 1 \
  --monitor none --output-dir /shared/results/tokenpowerbench
```

The positional number is the worker count, excluding the head. The helper
submits one exclusive allocation containing all nodes. It creates log directories
before submission, launches each Ray service with `srun`, and waits for the
allocated node count and GPU resources before starting inference. A unique
per-launch resource marker is checked before workers join and again across the
complete cluster, preventing accidental attachment to another Ray service. The example
model must support the chosen TP size; use a smaller topology if it does not.
This is a functional configuration, not a published-paper setting.
Choose SLURM discovery when creating the `cluster-slurm.json` used by this example.
The scripts can also run without a profile using the environment settings below.
The SLURM launcher always selects the first allocated node as head and rejects
`--ray-head-address`; it cannot target a different manual head. An explicit
`--ray-head-port` can override the profile's port. A SLURM profile ignores stale
`RAY_HEAD_ADDRESS` and `RAY_ADDRESS` values from the submission environment.
A profile with `network_interface: null` selects automatic interface discovery
and ignores inherited `TPB_NETWORK_INTERFACE`; specify a name in the profile to
select a particular interface consistently across launches.

| Environment variable | Default / meaning |
| --- | --- |
| `TPB_PROJECT_DIR` | Checkout containing scripts and `run_multi_node.py`; must be shared |
| `TPB_PYTHON` | Selected Python interpreter; resolved to an absolute path available on every node |
| `TPB_PARTITION`, `TPB_ACCOUNT` | Optional site partition and account |
| `TPB_GPUS_PER_NODE`, `TPB_CPUS_PER_NODE` | `4`, `16`; homogeneous allocation resources |
| `TPB_MEMORY` | `0`: all node memory; accepts SLURM memory values |
| `TPB_TIME` | `01:00:00` allocation limit |
| `TPB_LOG_DIR` | `<checkout>/logs`; batch, per-node Ray, and cluster-readiness logs |
| `TPB_NETWORK_INTERFACE` | Optional common interface name used to select one routable IPv4 address per node |
| `RAY_HEAD_PORT` | `6379`; must be reachable from every allocated node |
| `TPB_CLUSTER_STARTUP_TIMEOUT_S` | `180` seconds for each launcher readiness stage |

Without an interface override, each SLURM node name must resolve to one routable
IPv4 address. When launching without a profile, an existing `RAY_HEAD_ADDRESS`
must match the allocated head's selected interface. The launcher sets each node's
`VLLM_HOST_IP` and preserves SLURM's
CUDA visibility. Ray and NCCL communication also require the cluster's relevant
ports/interfaces to be reachable; configure those according to site policy.

Activate modules or the Python environment before submission. The scripts do not
source interactive shell startup files. They use node-local, per-job compilation
caches and preserve model download caches. On completion, failure, or a trapped
signal, the controller terminates its own driver and `srun` services; SLURM owns the allocation
and remaining process cleanup. Inspect node logs if a service fails to start.

### Inspect SLURM nodes and interfaces

Inside an active allocation, inspect the scheduler's node list before starting
Ray services:

```bash
scontrol show hostnames "$SLURM_JOB_NODELIST"
```

The first printed hostname is the head. To inspect each node's non-loopback IPv4
interfaces through the allocation:

```bash
while IFS= read -r node; do
  printf '%s\n' "$node"
  srun --nodes=1 --ntasks=1 --nodelist="$node" \
    ip -o -4 address show scope global
done < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
```

Nodes can have several addresses for management, storage, and compute networks.
Select the interface that connects all participating nodes. Enter its common
name in the wizard, or set `TPB_NETWORK_INTERFACE` for an environment-based
launch. A saved interface is honored by both the launcher and direct profile
resolution, which queries the allocated head through `srun`. For example,
inspect a site's `ib0` interface with:

```bash
srun --nodes=1 --ntasks=1 --nodelist=allocated_node_name \
  ip -o -4 address show dev ib0 scope global
```

Replace the node and interface placeholders with actual allocation values. If
`SLURM_JOB_NODELIST` is unset, use the site's `salloc` procedure or submit the
complete benchmark allocation; a login node cannot discover a future allocation.

## Use an existing Ray cluster

Start the cluster through your site's supported procedure, then run the driver.
The installed `tpbench-multi` command accepts the same arguments as
`python run_multi_node.py`:

```bash
tpbench-multi --cluster-config cluster.json \
  --models Qwen2.5-7B-Instruct --model-dir /shared/models \
  --prompts-file examples/prompts.json --num-samples 4 \
  --tensor-parallel 4 --pipeline-parallel 2 --concurrency 1 \
  --batch-sizes 1 --max-tokens 128 --seed 42 --monitor none \
  --output-dir results
```

Use a manual profile containing the actual head address for this example.
For a direct `tpbench-multi` run, explicit `--ray-head-address` and
`--ray-head-port` options override the profile;
the profile overrides environment settings. Without a profile, resolution uses
explicit options, then `RAY_HEAD_ADDRESS`/`RAY_ADDRESS`, then the first SLURM
allocation hostname, then Ray's existing-cluster discovery. The CLI does not
start a local cluster automatically.

When using the engine from a Python process already connected to Ray, an explicit
native address must match that connection. Use `RayClusterConfig(head_address="auto")`
to intentionally reuse it. Worker preparation records the requested address and
the connected GCS address separately.

`--concurrency` means independent model replicas; `--batch-sizes` controls how
many prompts are passed together to a replica. TP × PP × concurrency must fit the
available GPU placement resources. Placement, model startup, inference, and
shutdown timeouts default to 120, 900, 3600, and 30 seconds, respectively; set
`--placement-timeout-s`, `--startup-timeout-s`, `--inference-timeout-s`, or
`--shutdown-timeout-s` for the workload.

Use `--model-kwargs-file settings.json` for an explicit JSON object of supported
vLLM model options. Record those options alongside the model revision. The
`--max-model-len` and `--gpu-memory-utilization` flags provide direct overrides.

## Interpret results

Each suite writes a unique `multi_<timestamp>_<id>/` directory with configuration,
ordered prompts, environment, process identity, status, results, and failures.
Each configuration has its own status, prompts, worker metadata, and result file.
Model initialization and replica warmup finish before the measured inference
window. Report generated token-ID counts rather than configured token limits.

For optional driver diagnostics, select `--monitor gpu_only`, `auto`, or
`full_node`; `--driver-device-indices 0,1` explicitly selects physical NVML devices
on that host. IPMI and RAPL permissions follow the
[measurement definitions](measurement.md#sensor-scope-and-permissions).
Driver raw traces and capabilities are saved per configuration. They do not
measure remote workers, and their energy is not divided by all distributed tokens.

Check both suite and configuration status. A suite with some failed configurations
returns a nonzero exit code and records `partial_failure`. Preserve the complete
directory, scheduler logs, model revisions, and the software environment. Successful
mocked tests cannot establish distributed GPU correctness or cluster energy accuracy.

For deployment details, consult the official
[Ray SLURM guide](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html),
[SLURM sbatch reference](https://slurm.schedmd.com/sbatch.html), and
[SLURM srun reference](https://slurm.schedmd.com/srun.html).

## Test orchestration without GPUs

The unit suite uses mocked inference and scheduler interfaces:

```bash
python -m unittest discover -s tests -v
```

An optional process test starts two local Ray nodes with synthetic GPU resource
labels and stub inference actors. It exercises placement across logical nodes,
multiple replicas, partial final batches, repeated runs, and resource release.
In a separate CPU test environment:

```bash
python -m pip install . 'ray==2.48.0' packaging
python tests/integration/ray_process_smoke.py
```

This test writes a summary under `results/ray-process-smoke/` and shuts down its
own local services. It does not execute CUDA kernels, load a vLLM model, invoke
SLURM, measure power, or communicate between physical hosts.
