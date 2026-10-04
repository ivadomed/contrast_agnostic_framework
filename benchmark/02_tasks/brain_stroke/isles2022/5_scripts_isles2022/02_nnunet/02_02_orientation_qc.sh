#!/bin/bash
# Orientation/alignment QC PNG via run_job (CPU).  bash 02_02_orientation_qc.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/benchmark/02_tasks/brain_stroke/isles2022/2_nnUNet_isles2022/logs"
mkdir -p "${LOG_DIR}"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
run_job --name isles2022_orient_qc --gpus 0 --cpus 2 --mem 8G --time 00:20:00 \
    --log "${LOG_DIR}/orient_qc_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/02_02_orientation_qc.py"
