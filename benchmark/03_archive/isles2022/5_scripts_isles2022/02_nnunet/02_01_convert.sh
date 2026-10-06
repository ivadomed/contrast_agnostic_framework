#!/bin/bash
# BIDS -> nnU-Net raw (Dataset140 DWI + Dataset141 FLAIR) via run_job (CPU).  bash 02_01_convert.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/benchmark/02_tasks/brain_stroke/isles2022/2_nnUNet_isles2022/logs"
mkdir -p "${LOG_DIR}"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
run_job --name isles2022_convert --gpus 0 --cpus 4 --mem 24G --time 02:00:00 \
    --log "${LOG_DIR}/convert_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/02_01_convert.py"
