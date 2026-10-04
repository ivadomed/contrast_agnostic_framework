#!/bin/bash
# Create isles2022 partition + 3-fold splits via run_job (CPU).  bash 01_01_create_splits.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/benchmark/02_tasks/brain_stroke/isles2022/4_splits_isles2022/logs"
mkdir -p "${LOG_DIR}"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
run_job --name isles2022_splits --gpus 0 --cpus 1 --mem 4G --time 00:15:00 \
    --log "${LOG_DIR}/splits_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/01_01_create_splits.py"
