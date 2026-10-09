#!/usr/bin/env bash
# Build the FOV-matched PDDCA CT test set (see the .py header for the FOV/resolution
# rationale, the licence status, the 40-of-48 count and the verified label semantics).
#
#   bash 01_01_prepare_ct.sh
#
# Expects the three v1.4.1 archives as ${PDDCA_ZIP_DIR}/part{1,2,3}.zip.
# Source: https://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part{1,2,3}.zip
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
PD_STAGE="${PD_STAGE:-${SCRATCH:-/scratch/p/paulh}/pddca}"
export PDDCA_ZIP_DIR="${PDDCA_ZIP_DIR:-${PD_STAGE}/0_raw}"
ls "${PDDCA_ZIP_DIR}"/part*.zip >/dev/null 2>&1 || { echo "missing part*.zip in ${PDDCA_ZIP_DIR}" >&2; exit 1; }
LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}" "${BIDS_ROOT}"

run_job --name "pddca_prepare" --gpus 0 --cpus 8 --mem 96G --time 03:00:00 \
    --slot 0 --wait --log "${LOG_DIR}/prepare_$(date +%Y%m%d_%H%M%S).log" -- bash -c "
        export PDDCA_ZIP_DIR='${PDDCA_ZIP_DIR}' BIDS_ROOT='${BIDS_ROOT}' nnUNet_raw='${nnUNet_raw}'
        export PDDCA_WORKERS='${PDDCA_WORKERS:-6}'
        cd '${PROJECT_ROOT}'
        .venv/bin/python '${HERE}/01_01_prepare_ct.py'
    "
