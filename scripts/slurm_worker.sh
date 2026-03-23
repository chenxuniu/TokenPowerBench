#!/bin/bash
# ============================================================
# Ray Worker Node — SLURM job script
#
# Reads the head node address from the RAY_HEAD_ADDRESS
# environment variable, which should be set before submitting
# this job (see submit_jobs.sh) or passed via --export.
#
# If RAY_HEAD_ADDRESS is not set, the script exits with an error
# so that misconfigured jobs fail fast rather than silently.
# ============================================================

#SBATCH --job-name=tpbench-worker
#SBATCH --partition=h100          # <-- change to your partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4              # <-- GPUs per worker node
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/worker_%j.out
#SBATCH --error=logs/worker_%j.err
#SBATCH --exclusive

set -euo pipefail
mkdir -p logs

echo "=== TokenPowerBench — Ray Worker Node ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Date   : $(date)"

# ------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------
source ~/.bashrc
# source ~/venvs/tpbench/bin/activate

cd "$(dirname "$0")/.."

# ------------------------------------------------------------------
# Validate that the head address was passed
# ------------------------------------------------------------------
if [ -z "${RAY_HEAD_ADDRESS:-}" ]; then
    echo "ERROR: RAY_HEAD_ADDRESS is not set."
    echo "  Pass it via: sbatch --export=RAY_HEAD_ADDRESS=<ip> slurm_worker.sh"
    echo "  or set it in the environment before calling sbatch."
    exit 1
fi

RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
RAY_ADDRESS="$RAY_HEAD_ADDRESS:$RAY_HEAD_PORT"
echo "Connecting to Ray head at: $RAY_ADDRESS"

# ------------------------------------------------------------------
# Clean slate
# ------------------------------------------------------------------
ray stop --force 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
rm -rf ~/.cache/vllm/torch_compile_cache \
        ~/.cache/torch/compile_cache \
        ~/.cache/triton
sleep 5

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_ON_NODE - 1)))
fi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------------
# Connect to head with retries
# ------------------------------------------------------------------
MAX_RETRIES=10
RETRY_INTERVAL=20

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "Attempt $attempt/$MAX_RETRIES: connecting to $RAY_ADDRESS …"
    if ray start --address="$RAY_ADDRESS" \
            --num-cpus="${SLURM_CPUS_ON_NODE:-8}" \
            --num-gpus="${SLURM_GPUS_ON_NODE:-4}" \
            --disable-usage-stats; then
        echo "Connected to Ray cluster."
        break
    fi
    if [ "$attempt" -eq "$MAX_RETRIES" ]; then
        echo "ERROR: Could not connect after $MAX_RETRIES attempts. Exiting."
        exit 1
    fi
    echo "Retrying in ${RETRY_INTERVAL}s…"
    sleep "$RETRY_INTERVAL"
done

# ------------------------------------------------------------------
# Keep alive until the SLURM job is cancelled or the head disappears
# ------------------------------------------------------------------
echo "Worker is active. Monitoring every 30 s…"
while ray status &>/dev/null; do
    sleep 30
done
echo "Ray head is gone. Worker exiting."
ray stop --force
echo "=== Worker node done ==="
