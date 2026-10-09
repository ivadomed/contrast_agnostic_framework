#!/usr/bin/env bash
# nnUNet plan & preprocess (with --verify_dataset_integrity) for toothfairy2, then
# install this dataset's own 3-fold splits.
#
#   bash 02_01_plan_and_preprocess.sh
#
# Only 3d_fullres is preprocessed (-c 3d_fullres): this project trains 3D ONLY —
# train_common.sh hardcodes `nnUNetv2_train <id> 3d_fullres <fold>` — so
# nnUNetPlans_2d / _3d_lowres would be pure wasted CPU and disk (~480 cases here).
# PLANNING still covers every configuration; only the preprocessing pass is limited.
#
# Splits are copied AFTER preprocessing and byte-verified, because
# nnUNetv2_plan_and_preprocess writes into the same dir and can clobber a
# pre-placed splits file.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}"

ID="${DATASET_ID_CBCT}"
DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${ID}_" | head -1)"
[ -n "${DS_NAME}" ] || { echo "no raw dir for dataset id ${ID} in ${nnUNet_raw}" >&2; exit 1; }

run_job --name "tf2_preprocess" --gpus 0 --cpus 16 --mem 128G --time 12:00:00 \
    --slot 0 --wait --log "${LOG_DIR}/preprocess_$(date +%Y%m%d_%H%M%S).log" -- bash -c "
        export nnUNet_raw='${nnUNet_raw}' nnUNet_preprocessed='${nnUNet_preprocessed}'
        export nnUNet_results='${nnUNet_results}'
        cd '${PROJECT_ROOT}'
        .venv/bin/nnUNetv2_plan_and_preprocess -d ${ID} -c 3d_fullres --verify_dataset_integrity
    "

SRC="${SPLITS_DIR}/splits_final.json"
DST="${nnUNet_preprocessed}/${DS_NAME}/splits_final.json"
[ -f "${SRC}" ] || { echo "missing ${SRC} — run 01_create_splits/01_01_create_splits.sh first" >&2; exit 1; }
cp "${SRC}" "${DST}"
cmp -s "${SRC}" "${DST}" || { echo "splits copy differs: ${DST}" >&2; exit 1; }
echo "Installed + byte-verified splits: ${SRC} -> ${DST}"
