#!/usr/bin/env bash
# Submit one allocation containing the Ray head and every worker.
# Usage: bash scripts/submit_jobs.sh NUM_WORKERS --models MODEL [benchmark options]
set -euo pipefail

usage() {
  echo "Usage: $0 NUM_WORKERS [--] --models MODEL [run_multi_node.py options]"
  echo "Set TPB_PARTITION, TPB_ACCOUNT, TPB_GPUS_PER_NODE (4), TPB_CPUS_PER_NODE (16),"
  echo "TPB_MEMORY (0 = all node memory), TPB_TIME (01:00:00), and TPB_PYTHON as needed."
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ ! "${1:-}" =~ ^[0-9]+$ ]]; then usage >&2; exit 2; fi
workers=$((10#$1))
shift
if [[ "${1:-}" == -- ]]; then shift; fi
if (( $# == 0 )); then usage >&2; exit 2; fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TPB_PROJECT_DIR="${TPB_PROJECT_DIR:-$(cd -- "$script_dir/.." && pwd)}"
TPB_PROJECT_DIR="$(cd -- "$TPB_PROJECT_DIR" && pwd)"
python_path="$(command -v "${TPB_PYTHON:-python}")"
[[ -x "$python_path" ]] || { echo "TPB_PYTHON must name an executable interpreter." >&2; exit 2; }
export TPB_PYTHON="$(cd -- "$(dirname -- "$python_path")" && pwd)/$(basename -- "$python_path")"
source "$TPB_PROJECT_DIR/scripts/slurm_common.sh"
tpb_apply_cluster_profile "$@"
set -- "${TPB_BENCHMARK_ARGS[@]}"
export TPB_GPUS_PER_NODE="${TPB_GPUS_PER_NODE:-4}"
export TPB_CPUS_PER_NODE="${TPB_CPUS_PER_NODE:-16}"
for count in "$TPB_GPUS_PER_NODE" "$TPB_CPUS_PER_NODE"; do
  [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "GPU and CPU counts must be positive integers." >&2; exit 2; }
done
export TPB_LOG_DIR="${TPB_LOG_DIR:-$TPB_PROJECT_DIR/logs}"
mkdir -p -- "$TPB_LOG_DIR"
TPB_LOG_DIR="$(cd -- "$TPB_LOG_DIR" && pwd)"
options=(--parsable --job-name=tpbench --nodes="$((workers + 1))" --ntasks-per-node=1
  --cpus-per-task="$TPB_CPUS_PER_NODE" --gres="gpu:$TPB_GPUS_PER_NODE"
  --exclusive --mem="${TPB_MEMORY:-0}" --time="${TPB_TIME:-01:00:00}"
  --chdir="$TPB_PROJECT_DIR" --export=ALL
  --output="$TPB_LOG_DIR/%x_%j.out" --error="$TPB_LOG_DIR/%x_%j.err")
[[ -z "${TPB_PARTITION:-}" ]] || options+=(--partition="$TPB_PARTITION")
[[ -z "${TPB_ACCOUNT:-}" ]] || options+=(--account="$TPB_ACCOUNT")
job_id="$(sbatch "${options[@]}" "$TPB_PROJECT_DIR/scripts/slurm_head.sh" "$@")"
echo "Submitted allocation $job_id: one head and $workers worker(s)."
echo "Logs: $TPB_LOG_DIR"
