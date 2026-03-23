#!/bin/bash
# ============================================================
# Submit Ray head + worker jobs to SLURM.
#
# The head job auto-detects its own IP and writes it to a
# shared file. Worker jobs read that file before joining.
#
# Usage:
#   bash scripts/submit_jobs.sh [num_workers]
#
# Example (1 head + 3 workers):
#   bash scripts/submit_jobs.sh 3
# ============================================================

set -euo pipefail

NUM_WORKERS="${1:-3}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
COORD_FILE="/tmp/ray_head_address_${USER}.txt"

echo "Submitting head job…"

# Submit head node — it will write its IP to COORD_FILE
HEAD_JOB=$(sbatch \
    --parsable \
    --export=ALL,RAY_COORD_FILE="$COORD_FILE" \
    "$SCRIPTS_DIR/slurm_head.sh")

echo "Head job ID: $HEAD_JOB"

# Wait for the head to write its IP (up to 120 s)
echo "Waiting for head node to start and write its IP…"
for i in $(seq 1 24); do
    if [ -f "$COORD_FILE" ]; then
        HEAD_IP=$(cat "$COORD_FILE")
        echo "Head IP: $HEAD_IP"
        break
    fi
    sleep 5
done

if [ -z "${HEAD_IP:-}" ]; then
    echo "ERROR: Head did not write IP within 120 s. Check head job logs."
    exit 1
fi

# Submit workers with the head address passed as an env variable
echo "Submitting $NUM_WORKERS worker job(s)…"
for i in $(seq 1 "$NUM_WORKERS"); do
    WORKER_JOB=$(sbatch \
        --parsable \
        --dependency="after:${HEAD_JOB}" \
        --export=ALL,RAY_HEAD_ADDRESS="$HEAD_IP" \
        "$SCRIPTS_DIR/slurm_worker.sh")
    echo "Worker $i job ID: $WORKER_JOB"
done

echo ""
echo "All jobs submitted."
echo "  Head   : $HEAD_JOB"
echo "  Workers: use 'squeue -u $USER' to monitor"
