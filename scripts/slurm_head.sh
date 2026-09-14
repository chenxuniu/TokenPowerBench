#!/usr/bin/env bash
# The batch controller runs on the allocation's first node.
# Submit with scripts/submit_jobs.sh so source paths and resources are explicit.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a SLURM allocation.}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required.}"
: "${TPB_PROJECT_DIR:=${SLURM_SUBMIT_DIR:-}}"
[[ -f "$TPB_PROJECT_DIR/scripts/slurm_common.sh" ]] || {
  echo "Set TPB_PROJECT_DIR to the shared TokenPowerBench checkout." >&2; exit 2;
}
source "$TPB_PROJECT_DIR/scripts/slurm_common.sh"
(( $# > 0 )) || tpb_fail "Supply run_multi_node.py arguments, including --models."
tpb_apply_cluster_profile "$@"
set -- "${TPB_BENCHMARK_ARGS[@]}"
tpb_setup

nodes_text="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
nodes=()
while IFS= read -r node; do [[ -z "$node" ]] || nodes+=("$node"); done <<< "$nodes_text"
(( ${#nodes[@]} > 0 )) || tpb_fail "SLURM returned no allocated nodes."
[[ "${SLURMD_NODENAME:-$(hostname -s)}" == "${nodes[0]}" ]] ||
  tpb_fail "Run the batch controller on the allocation's first node."

detected_head_ip="$(tpb_node_ip)"
RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-$detected_head_ip}"
[[ "$RAY_HEAD_ADDRESS" == "$detected_head_ip" ]] || tpb_fail "RAY_HEAD_ADDRESS differs from the selected head interface."
export RAY_HEAD_ADDRESS
export VLLM_HOST_IP="$RAY_HEAD_ADDRESS"
export RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
[[ "$RAY_HEAD_PORT" =~ ^[1-9][0-9]*$ ]] && (( RAY_HEAD_PORT <= 65535 )) || tpb_fail "RAY_HEAD_PORT must be 1..65535."
export RAY_ADDRESS="$RAY_HEAD_ADDRESS:$RAY_HEAD_PORT"
export TPB_CLUSTER_STARTUP_TIMEOUT_S="${TPB_CLUSTER_STARTUP_TIMEOUT_S:-180}"
[[ "$TPB_CLUSTER_STARTUP_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || tpb_fail "Startup timeout must be a positive integer."
export TPB_LOG_DIR="${TPB_LOG_DIR:-$TPB_PROJECT_DIR/logs}"
mkdir -p -- "$TPB_LOG_DIR"
export TPB_GPUS_PER_NODE="$TPB_VISIBLE_GPUS"
export TPB_CPUS_PER_NODE="$TPB_ALLOCATED_CPUS"
export TPB_CLUSTER_RESOURCE="tpb_${SLURM_JOB_ID}_$$_${RANDOM}"

# Refuse to attach workers to a pre-existing service on the requested head port.
"$TPB_PYTHON" - "$RAY_HEAD_ADDRESS" "$RAY_HEAD_PORT" <<'PYTHON'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.bind((sys.argv[1], int(sys.argv[2])))
PYTHON

step_pids=()
driver_pid=""
cleanup() {
  local status=$? pid running deadline
  trap - EXIT INT TERM
  if [[ -n "$driver_pid" ]]; then step_pids+=("$driver_pid"); fi
  for pid in "${step_pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  deadline=$((SECONDS + 15))
  while (( SECONDS < deadline )); do
    running=0
    for pid in "${step_pids[@]}"; do kill -0 "$pid" 2>/dev/null && running=1; done
    (( running )) || break
    sleep 1
  done
  for pid in "${step_pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_node() {
  local node="$1" role="$2"
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="$node" --exact \
    --cpus-per-task="$TPB_CPUS_PER_NODE" --gres="gpu:$TPB_GPUS_PER_NODE" \
    --export=ALL --chdir="$TPB_PROJECT_DIR" \
    bash "$TPB_PROJECT_DIR/scripts/slurm_worker.sh" "$role" \
    > "$TPB_LOG_DIR/ray_${SLURM_JOB_ID}_${node}.log" 2>&1 &
  step_pids+=("$!")
}
start_node "${nodes[0]}" head

# Bound head startup before workers attempt to connect. No fixed startup delay.
timeout "$TPB_CLUSTER_STARTUP_TIMEOUT_S" "$TPB_PYTHON" - "$RAY_HEAD_ADDRESS" "$RAY_HEAD_PORT" <<'PYTHON'
import socket
import sys
import time

while True:
    try:
        with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2):
            break
    except OSError:
        time.sleep(1)
PYTHON
# Verify this launch's marker before any workers can join the head.
wait_cluster() {
  timeout "$TPB_CLUSTER_STARTUP_TIMEOUT_S" "$TPB_PYTHON" - \
    "$RAY_ADDRESS" "$1" "$TPB_GPUS_PER_NODE" "$TPB_CLUSTER_RESOURCE" \
    "$TPB_LOG_DIR/cluster_${SLURM_JOB_ID}.json" <<'PYTHON'
import json
from pathlib import Path
import sys
import time
import ray

address, expected_nodes, gpus_per_node, marker, output = sys.argv[1:]
expected_nodes, gpus_per_node = int(expected_nodes), int(gpus_per_node)
ray.init(address=address, log_to_driver=False)
try:
    while True:
        alive = [node for node in ray.nodes() if node.get("Alive")]
        if any(node.get("Resources", {}).get(marker) != 1 for node in alive):
            raise RuntimeError("Ray contains a node that does not belong to this launcher")
        if len(alive) > expected_nodes:
            raise RuntimeError("Ray contains nodes outside the requested allocation")
        if len(alive) == expected_nodes and all(node.get("Resources", {}).get("GPU", 0) == gpus_per_node for node in alive):
            Path(output).write_text(json.dumps({"address": address, "launch_resource": marker, "nodes": alive}, indent=2))
            break
        time.sleep(1)
finally:
    ray.shutdown()
PYTHON
}
wait_cluster 1
kill -0 "${step_pids[0]}" 2>/dev/null || tpb_fail "The owned Ray head exited; workers were not started."
for ((i = 1; i < ${#nodes[@]}; i++)); do start_node "${nodes[i]}" worker; done
wait_cluster "${#nodes[@]}"
for pid in "${step_pids[@]}"; do kill -0 "$pid" 2>/dev/null || tpb_fail "A Ray node exited during startup; inspect node logs."; done
echo "Ray ready at $RAY_ADDRESS on ${#nodes[@]} nodes."
"$TPB_PYTHON" "$TPB_PROJECT_DIR/run_multi_node.py" "$@" \
  --ray-head-address "$RAY_HEAD_ADDRESS" --ray-head-port "$RAY_HEAD_PORT" &
driver_pid=$!
wait "$driver_pid"
driver_pid=""
