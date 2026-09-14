#!/usr/bin/env bash
# Submit the complete allocation through the shared launcher.
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project/scripts/submit_jobs.sh" "$@"
