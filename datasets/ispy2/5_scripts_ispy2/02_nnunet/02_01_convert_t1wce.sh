#!/bin/bash
# ispy2 -> nnU-Net raw, t1wce modality (see 02_00_convert_lib.py for the design).
# Submitted through run_job: ~1.7k volumes are copied and every mask is
# decompressed/rewritten, well past the login-node exception in CLAUDE.md.
#
#   bash 02_01_convert_t1wce.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/datasets/ispy2/5_scripts_ispy2/02_nnunet/logs"
mkdir -p "${LOG_DIR}"

source "${REPO_ROOT}/scripts/job_runner/run_job.sh"

run_job --name ispy2_convert_t1wce --gpus 0 --cpus 4 --mem 16G --time 04:00:00 \
    --log "${LOG_DIR}/convert_t1wce_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/02_01_convert_t1wce.py"
