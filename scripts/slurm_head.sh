#!/bin/bash
# ============================================================
# Ray Head Node — SLURM job script
#
# No hardcoded IPs. The head node IP is auto-detected from
# the SLURM environment and exported as RAY_HEAD_ADDRESS so
# that worker nodes and Python code can connect without any
# manual configuration.
#
# Customise the #SBATCH lines and the benchmark parameters
# at the bottom of this file. Everything else is generic.
# ============================================================

#SBATCH --job-name=tpbench-head
#SBATCH --partition=h100          # <-- change to your partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4              # <-- GPUs per head node
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/head_%j.out
#SBATCH --error=logs/head_%j.err
#SBATCH --exclusive

set -euo pipefail
mkdir -p logs

echo "=== TokenPowerBench — Ray Head Node ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Date   : $(date)"

# ------------------------------------------------------------------
# Environment setup — adjust paths to match your cluster
# ------------------------------------------------------------------
source ~/.bashrc
# Activate your virtual environment, e.g.:
# source ~/venvs/tpbench/bin/activate

cd "$(dirname "$0")/.."   # project root

# ------------------------------------------------------------------
# Auto-detect this node's IP — no hardcoding needed
# ------------------------------------------------------------------
HEAD_IP=$(hostname --ip-address | awk '{print $1}')
export RAY_HEAD_ADDRESS="$HEAD_IP"
export RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"

echo "Head IP : $HEAD_IP"
echo "Ray address will be: $HEAD_IP:$RAY_HEAD_PORT"

# ------------------------------------------------------------------
# Clean slate
# ------------------------------------------------------------------
echo "Cleaning Ray processes and compile caches…"
ray stop --force 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
rm -rf ~/.cache/vllm/torch_compile_cache \
        ~/.cache/torch/compile_cache \
        ~/.cache/triton
sleep 5

# Ensure CUDA_VISIBLE_DEVICES is set
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_ON_NODE - 1)))
fi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------------
# Start Ray head
# ------------------------------------------------------------------
ray start --head \
    --node-ip-address="$HEAD_IP" \
    --port="$RAY_HEAD_PORT" \
    --num-cpus="${SLURM_CPUS_ON_NODE:-8}" \
    --num-gpus="${SLURM_GPUS_ON_NODE:-4}" \
    --disable-usage-stats \
    --object-store-memory=3000000000

echo "Ray head started."

# Wait for workers to connect (adjust to match your worker job time)
echo "Waiting 90 s for workers to connect…"
sleep 90

echo "--- Cluster status ---"
ray status

# ------------------------------------------------------------------
# Run benchmark
# ============================================================
# Edit the parameters below to configure your experiment.
# See run_multi_node.py --help for all options.
# ============================================================
python run_multi_node.py \
    --models "Llama-3.1-405B" \
    --model-dir "$HOME/models" \
    --datasets "alpaca" \
    --batch-sizes "256" \
    --tensor-parallel "8" \
    --pipeline-parallel "2" \
    --concurrency "1" \
    --num-samples 1000 \
    --max-tokens 512 \
    --monitor auto \
    --output-dir "./results"

# ------------------------------------------------------------------
# Teardown
# ------------------------------------------------------------------
echo "Stopping Ray cluster…"
ray stop --force
echo "=== Head node done ==="
