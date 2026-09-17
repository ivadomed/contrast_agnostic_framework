#!/bin/bash
# Derive duke-breast-mri's unilateral (single-breast) crop of the existing
# bilateral t1wce/precontrast test volumes (291 cases x 2 items = 582 volumes
# read, cropped, re-gzipped) -- past the login-node exception, submit via
# run_job.
#
#   bash 02_03_derive_unilateral_crop.sh
#
# Prerequisite: both 02_01_convert_test_t1wce.sh and
# 02_02_convert_test_precontrast.sh have completed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/datasets/duke-breast-mri/2_nnUNet_duke-breast-mri/logs"
mkdir -p "${LOG_DIR}"

source "${REPO_ROOT}/scripts/job_runner/run_job.sh"

run_job --name duke_unilateral_crop --gpus 0 --cpus 4 --mem 16G --time 01:00:00 \
    --log "${LOG_DIR}/derive_unilateral_crop_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/02_03_derive_unilateral_crop.py"
