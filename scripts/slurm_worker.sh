#!/usr/bin/env bash
# Internal per-node service, launched by slurm_head.sh through srun.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a SLURM allocation.}"
: "${TPB_PROJECT_DIR:?Set TPB_PROJECT_DIR to the shared checkout.}"
source "$TPB_PROJECT_DIR/scripts/slurm_common.sh"
tpb_setup
role="${1:-worker}"
[[ "$role" == head || "$role" == worker ]] || tpb_fail "Role must be head or worker."
: "${RAY_HEAD_ADDRESS:?The controller must provide RAY_HEAD_ADDRESS.}"
: "${RAY_HEAD_PORT:?The controller must provide RAY_HEAD_PORT.}"
: "${TPB_CLUSTER_RESOURCE:?The controller must provide its launch resource marker.}"
[[ "$TPB_CLUSTER_RESOURCE" =~ ^tpb_[0-9_]+$ ]] || tpb_fail "Invalid launch resource marker."
export RAY_ADDRESS="$RAY_HEAD_ADDRESS:$RAY_HEAD_PORT"
VLLM_HOST_IP="$(tpb_node_ip)"
export VLLM_HOST_IP
args=(start --node-ip-address="$VLLM_HOST_IP" --num-cpus="$TPB_ALLOCATED_CPUS"
  --num-gpus="$TPB_VISIBLE_GPUS" --disable-usage-stats --block
  --resources="{\"$TPB_CLUSTER_RESOURCE\":1}")
if [[ "$role" == head ]]; then
  [[ "$VLLM_HOST_IP" == "$RAY_HEAD_ADDRESS" ]] || tpb_fail "Head address differs from this node's selected interface."
  args+=(--head --port="$RAY_HEAD_PORT" --include-dashboard=false --temp-dir="$RAY_TMPDIR")
else
  args+=(--address="$RAY_ADDRESS")
fi
# Ray's blocking service owns its child processes; SLURM owns this job step.
exec "$TPB_RAY" "${args[@]}"
