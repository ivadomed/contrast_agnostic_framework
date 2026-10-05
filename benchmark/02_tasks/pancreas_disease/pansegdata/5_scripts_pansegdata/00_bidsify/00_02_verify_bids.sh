#!/bin/bash
# Verify the BIDS leaf against the source zips (byte-exact images, label voxel counts, grids), via run_job (CPU).  bash 00_02_verify_bids.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/benchmark/02_tasks/pancreas_disease/pansegdata/1_BIDS_pansegdata/logs"
mkdir -p "${LOG_DIR}"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
LOG="${LOG_DIR}/verify_$(date +%Y%m%d_%H%M%S).log"
run_job --name pansegdata_verify_bids --gpus 0 --cpus 2 --mem 12G --time 00:45:00 \
    --log "${LOG}" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/00_02_verify_bids.py"
echo "---- ${LOG}"; tail -8 "${LOG}"
grep -q "ALL CHECKS PASSED" "${LOG}"
