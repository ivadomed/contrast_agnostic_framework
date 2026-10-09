#!/bin/bash
# Audit the PanSegData BIDS leaf (shape/spacing/axcodes/mask volume/physics contrast) via run_job (CPU).  bash 01_00_audit_raw.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/benchmark/02_tasks/pancreas_disease/pansegdata/0_raw_pansegdata/logs"
mkdir -p "${LOG_DIR}"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
run_job --name pansegdata_audit --gpus 0 --cpus 2 --mem 12G --time 01:00:00 \
    --log "${LOG_DIR}/audit_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/01_00_audit_raw.py"
