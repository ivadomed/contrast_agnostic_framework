#!/usr/bin/env bash
# FOV/geometry audit: this companion's test items vs the SOURCE's TRAINING images (shared 00_00_utils/fov_audit.py), via run_job (CPU).
# Run BEFORE choosing a crop and again after cropping. MISMATCH lines mean: crop/resample to the training geometry before predicting.
#   bash 02_02_fov_audit.sh [SOURCE_CONTRAST=t1wce] [ITEM_DIR_SUFFIX=]        (ITEM_DIR_SUFFIX e.g. _crop to audit cropped items)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"; cd "${PROJECT_ROOT}"
SC="${1:-t1wce}"; SUF="${2:-}"
case "${SC}" in t1wce) DSN=Dataset150_PanSegData_T1WCE;; t2w) DSN=Dataset151_PanSegData_T2W;; *) echo "SC must be t1wce|t2w" >&2; exit 1;; esac
TRAIN_DIR="${PANSEG_NNUNET_RAW}/${DSN}/imagesTr"
mkdir -p "${RESULTS_DIR}/_logs"
run_job --name totalsegmri-pancreas_fov_audit --gpus 0 --cpus 2 --mem 8G --time 00:20:00 --log "${RESULTS_DIR}/_logs/fov_audit_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    .venv/bin/python benchmark/00_commun_scripts/00_00_utils/fov_audit.py --train-dir "${TRAIN_DIR}" --test-dir "${nnUNet_raw}/imagesTs_t1gre${SUF}" "${nnUNet_raw}/imagesTs_t2like${SUF}"
tail -25 "$(ls -t "${RESULTS_DIR}"/_logs/fov_audit_*.log | head -1)"
