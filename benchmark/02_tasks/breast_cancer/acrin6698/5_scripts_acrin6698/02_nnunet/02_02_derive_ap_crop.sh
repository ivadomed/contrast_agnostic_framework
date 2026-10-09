#!/bin/bash
# Skin-anchored A-P crop of acrin6698's unilateral DWI item (dwi -> dwi_uniap): the
# axial full-chest DWI (~330 mm A-P) cut to the I-SPY2 unilateral training geometry
# (Paul, 2026-10-01). Skin detected on the b0 reference (refB0_dwi, written by 02_01) --
# fat is dark at b800. Shared logic: benchmark/00_commun_scripts/00_00_utils/derive_ap_crop.py.
#   bash 02_02_derive_ap_crop.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
RAW="${REPO_ROOT}/benchmark/02_tasks/breast_cancer/acrin6698/2_nnUNet_acrin6698/raw"
mkdir -p "${RAW}/../logs"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
run_job --name acrin_ap_crop --gpus 0 --cpus 2 --mem 12G --time 01:00:00 \
    --log "${RAW}/../logs/derive_ap_crop_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    bash -c "cd '${REPO_ROOT}/benchmark/00_commun_scripts/00_00_utils' && '${REPO_ROOT}/.venv/bin/python' derive_ap_crop.py \
        --raw '${RAW}' --items dwi --out-names dwi_uniap --ref-dir '${RAW}/refB0_dwi'"
