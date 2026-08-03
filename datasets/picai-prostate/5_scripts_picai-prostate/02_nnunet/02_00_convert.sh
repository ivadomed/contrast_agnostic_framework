#!/usr/bin/env bash
# Build the T2W nnU-Net raw dataset (Dataset080_PICAI_T2W) from BIDS.
# Thin wrapper around 02_00_convert.py — small CPU-only job (hard-linking NIfTIs +
# writing dataset.json), dispatched through run_job.
#
# Run AFTER 00_utils/00_01_bidsify.sh and 01_create_splits/01_01_create_splits.sh.
# Usage: bash 02_00_convert.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
[ -f "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh" ] && [ -d /scratch/p/paulh ] && \
    source "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh"
cd "${PROJECT_ROOT}"

mkdir -p "${RESULTS_DIR}/_logs"
run_job --name picai_convert_t2w --gpus 0 --cpus 4 --mem 16G --time 00:30:00 \
    --log "${RESULTS_DIR}/_logs/picai_convert_t2w.log" --wait -- bash -c "
    export BIDS_ROOT='${BIDS_ROOT}'
    export SPLITS_DIR='${SPLITS_DIR}'
    export nnUNet_raw='${nnUNet_raw}'
    cd '${PROJECT_ROOT}'
    .venv/bin/python '${SCRIPT_DIR}/02_00_convert.py' --contrast t2w
"
