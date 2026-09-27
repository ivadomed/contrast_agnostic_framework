#!/bin/bash
# Independent audit of the two ispy2 nnU-Net raw datasets (see 02_03_verify_raw.py).
# Opens ~3.6k volumes -> run_job, not the login node.
#
#   bash 02_03_verify_raw.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
run_job --name ispy2_verify_raw --gpus 0 --cpus 4 --mem 16G --time 02:00:00 \
    --log "${LOG_DIR}/verify_raw_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/02_03_verify_raw.py"
