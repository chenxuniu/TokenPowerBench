#!/usr/bin/env bash
# Per-node worker entrypoint; the controller supplies its allocation and address.
set -euo pipefail
project="${TPB_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$project" ]] || { echo "Use scripts/submit_jobs.sh to allocate the head and workers together." >&2; exit 2; }
exec bash "$project/scripts/slurm_worker.sh" worker "$@"
