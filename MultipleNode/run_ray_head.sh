#!/usr/bin/env bash
# Submit with bash MultipleNode/run_ray_head.sh NUM_WORKERS [benchmark options].
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  project="${TPB_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
  exec bash "$project/scripts/slurm_head.sh" "$@"
fi
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project/scripts/submit_jobs.sh" "$@"
