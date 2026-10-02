#!/bin/bash
# Skin-anchored A-P crop of duke's unilateral items (t1wce_uni, precontrast_uni ->
# t1wce_uniap, precontrast_uniap) -- duke's full-chest A-P extent (~350 mm) cut to the
# I-SPY2 unilateral training geometry (Paul, 2026-10-01). Shared logic:
# benchmark/00_commun_scripts/00_00_utils/derive_ap_crop.py. Skin detected on t1wce_uni.
#   bash 02_04_derive_ap_crop.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
RAW="${REPO_ROOT}/benchmark/02_tasks/breast_cancer/duke-breast-mri/2_nnUNet_duke-breast-mri/raw"
mkdir -p "${RAW}/../logs"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
run_job --name duke_ap_crop --gpus 0 --cpus 2 --mem 12G --time 01:00:00 \
    --log "${RAW}/../logs/derive_ap_crop_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    bash -c "cd '${REPO_ROOT}/benchmark/00_commun_scripts/00_00_utils' && '${REPO_ROOT}/.venv/bin/python' derive_ap_crop.py \
        --raw '${RAW}' --items t1wce_uni precontrast_uni --out-names t1wce_uniap precontrast_uniap"
