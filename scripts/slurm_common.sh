#!/usr/bin/env bash
# Shared setup for the SLURM controller and per-node Ray services.

tpb_fail() { echo "TokenPowerBench: $*" >&2; exit 2; }

tpb_apply_cluster_profile() {
  : "${TPB_PYTHON:=$(command -v python)}"
  [[ "$TPB_PYTHON" == /* && -x "$TPB_PYTHON" ]] || tpb_fail "TPB_PYTHON must be an absolute executable path."
  export TPB_PYTHON
  local settings index profile_path value
  local values=()
  settings="$("$TPB_PYTHON" - "$TPB_PROJECT_DIR" "$@" <<'PYTHON'
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
parser.add_argument("--cluster-config", type=Path)
parser.add_argument("--ray-head-port", type=int)
parser.add_argument("--ray-head-address")
parser.add_argument("--configure-cluster", action="store_true")
args, _ = parser.parse_known_args(sys.argv[2:])
if args.configure_cluster:
    parser.error("Run --configure-cluster interactively before submitting a SLURM job")
if args.ray_head_address is not None:
    parser.error("SLURM launchers select the head from the allocation; use tpbench-multi directly for an explicit Ray address")
profile = {}
profile_path = ""
try:
    if args.cluster_config is not None:
        from tokenpowerbench.distributed.cluster_setup import load_profile
        profile_path = str(args.cluster_config.expanduser().resolve(strict=True))
        profile = load_profile(profile_path)
        if profile["mode"] != "slurm":
            raise ValueError("SLURM launchers require a profile with mode 'slurm'")
    port = args.ray_head_port
    if port is None:
        port = profile.get("head_port", os.environ.get("RAY_HEAD_PORT", "6379"))
    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError("Ray head port must be 1..65535")
except (ValueError, TypeError, OSError) as error:
    parser.error(str(error))
print(port)
print((profile["network_interface"] or "") if profile else os.environ.get("TPB_NETWORK_INTERFACE", ""))
print(profile_path)
PYTHON
)" || return $?
  while IFS= read -r value; do values+=("$value"); done <<< "$settings"
  export RAY_HEAD_PORT="${values[0]}"
  export TPB_NETWORK_INTERFACE="${values[1]:-}"
  profile_path="${values[2]:-}"
  TPB_BENCHMARK_ARGS=("$@")
  # Keep a profile path valid after sbatch changes to the shared project directory.
  if [[ -n "$profile_path" ]]; then
    # A dynamic allocation profile takes precedence over saved address hints.
    unset RAY_HEAD_ADDRESS RAY_ADDRESS
    for ((index = 0; index < ${#TPB_BENCHMARK_ARGS[@]}; index++)); do
      case "${TPB_BENCHMARK_ARGS[index]}" in
        --cluster-config) TPB_BENCHMARK_ARGS[index + 1]="$profile_path" ;;
        --cluster-config=*) TPB_BENCHMARK_ARGS[index]="--cluster-config=$profile_path" ;;
      esac
    done
  fi
}

tpb_setup() {
  : "${TPB_PROJECT_DIR:?Set TPB_PROJECT_DIR to a shared checkout.}"
  [[ -f "$TPB_PROJECT_DIR/run_multi_node.py" ]] || tpb_fail "Project path is not available on this node."
  cd -- "$TPB_PROJECT_DIR"
  : "${TPB_PYTHON:=$(command -v python)}"
  [[ "$TPB_PYTHON" == /* && -x "$TPB_PYTHON" ]] || tpb_fail "TPB_PYTHON must be an absolute executable path shared by all nodes."
  export TPB_PYTHON
  TPB_RAY="$(dirname -- "$TPB_PYTHON")/ray"
  [[ -x "$TPB_RAY" ]] || tpb_fail "Install .[distributed] into the selected Python environment on every node."
  TPB_ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-${TPB_CPUS_PER_NODE:-}}"
  [[ "$TPB_ALLOCATED_CPUS" =~ ^[1-9][0-9]*$ ]] || tpb_fail "Set a positive SLURM --cpus-per-task allocation."
  TPB_VISIBLE_GPUS="$("$TPB_PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
  [[ "$TPB_VISIBLE_GPUS" =~ ^[1-9][0-9]*$ ]] || tpb_fail "No CUDA-visible GPUs in this SLURM task."
  if [[ -n "${TPB_GPUS_PER_NODE:-}" && "$TPB_VISIBLE_GPUS" != "$TPB_GPUS_PER_NODE" ]]; then
    tpb_fail "CUDA-visible GPU count ($TPB_VISIBLE_GPUS) differs from the requested per-node count ($TPB_GPUS_PER_NODE)."
  fi
  # Keep SLURM's CUDA visibility unchanged; UUIDs and remapped indices are valid.
  local node_cache="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/tpbench-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}-${SLURMD_NODENAME:-$(hostname -s)}"
  export RAY_TMPDIR="$node_cache/ray"
  export VLLM_CACHE_ROOT="$node_cache/vllm"
  export TORCHINDUCTOR_CACHE_DIR="$node_cache/torchinductor"
  export TRITON_CACHE_DIR="$node_cache/triton"
  mkdir -p -- "$RAY_TMPDIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  export RAY_USAGE_STATS_ENABLED=0
}

tpb_node_ip() {
  "$TPB_PYTHON" - "${SLURMD_NODENAME:-$(hostname -s)}" "${TPB_NETWORK_INTERFACE:-}" <<'PYTHON'
import ipaddress
import json
import socket
import subprocess
import sys

hostname, interface = sys.argv[1:]
if interface:
    data = json.loads(subprocess.check_output(["ip", "-j", "-4", "address", "show", "dev", interface], text=True))
    addresses = {item["local"] for device in data for item in device.get("addr_info", []) if item.get("scope") == "global"}
else:
    addresses = {entry[4][0] for entry in socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM)}
addresses = {address for address in addresses if not any((ipaddress.ip_address(address).is_loopback, ipaddress.ip_address(address).is_unspecified, ipaddress.ip_address(address).is_link_local))}
if len(addresses) != 1:
    raise SystemExit("Select a single routable IPv4 address with TPB_NETWORK_INTERFACE; candidates: " + repr(sorted(addresses)))
print(addresses.pop())
PYTHON
}
