#!/bin/bash
# Patient-level test partition + 3-fold CV (writes 4_splits_pansegdata/) via run_job (CPU).  bash 01_01_create_splits.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/benchmark/02_tasks/pancreas_disease/pansegdata/4_splits_pansegdata/logs"
mkdir -p "${LOG_DIR}"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
LOG="${LOG_DIR}/splits_$(date +%Y%m%d_%H%M%S).log"
run_job --name pansegdata_splits --gpus 0 --cpus 1 --mem 4G --time 00:10:00 --log "${LOG}" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/01_01_create_splits.py"
tail -8 "${LOG}"
