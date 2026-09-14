#!/bin/bash
# Build the Duke-Breast-Cancer-MRI (MAMA-MIA expert-mask subset) test-only
# nnU-Net set (291 patients, t1wce/DCE only). Submitted through run_job:
# ~582 volumes (pre + post-contrast) are copied/reoriented -- well past the
# login-node exception in CLAUDE.md.
#
#   bash 02_01_convert_test_t1wce.sh
#
# Prerequisite: datasets/duke-breast-mri/0_raw_duke-breast-mri/download_duke.py
# has completed (see 0_raw_duke-breast-mri/ for the manifest).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/datasets/duke-breast-mri/2_nnUNet_duke-breast-mri/logs"
mkdir -p "${LOG_DIR}"

source "${REPO_ROOT}/scripts/job_runner/run_job.sh"

run_job --name duke_test_t1wce --gpus 0 --cpus 4 --mem 16G --time 02:00:00 \
    --log "${LOG_DIR}/convert_test_t1wce_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/02_01_convert_test_t1wce.py"
