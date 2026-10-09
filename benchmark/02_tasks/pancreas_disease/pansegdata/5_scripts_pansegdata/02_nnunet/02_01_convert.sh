#!/bin/bash
# BIDS -> nnU-Net raw (Dataset150 T1WCE + Dataset151 T2W, canonical LPS) via run_job (CPU).  bash 02_01_convert.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/benchmark/02_tasks/pancreas_disease/pansegdata/2_nnUNet_pansegdata/logs"
mkdir -p "${LOG_DIR}"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
LOG="${LOG_DIR}/convert_$(date +%Y%m%d_%H%M%S).log"
run_job --name pansegdata_convert --gpus 0 --cpus 4 --mem 24G --time 02:00:00 --log "${LOG}" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/02_01_convert.py"
tail -6 "${LOG}"
