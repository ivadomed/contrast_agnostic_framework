#!/usr/bin/env bash
# nnUNet plan & preprocess (3d_fullres only, --verify_dataset_integrity) for both isles2022 training
# datasets, then install our patient-level 3-fold splits (copied AFTER preprocessing, byte-verified, because
# plan_and_preprocess writes into the same dir and can clobber a pre-placed file). Both datasets share one
# splits_final.json (identical case ids).   bash 02_03_plan_and_preprocess.sh [dwi|flair]
# CPU-only. 3D only: train_common.sh hardcodes 3d_fullres.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}"
CONTRASTS=("${@:-}"); [ -z "${CONTRASTS[0]:-}" ] && CONTRASTS=(dwi flair)
declare -A DS_ID=( [dwi]="${DATASET_ID_DWI}" [flair]="${DATASET_ID_FLAIR}" )
for C in "${CONTRASTS[@]}"; do
    ID="${DS_ID[$C]}"
    DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${ID}_" | head -1)"
    [ -n "${DS_NAME}" ] || { echo "no raw dir for dataset id ${ID} in ${nnUNet_raw}" >&2; exit 1; }
    run_job --name "isles2022_preprocess_${C}" --gpus 0 --cpus 8 --mem 32G --time 04:00:00 \
        --slot 0 --wait --log "${LOG_DIR}/preprocess_${C}_$(date +%Y%m%d_%H%M%S).log" -- bash -c "
            export nnUNet_raw='${nnUNet_raw}' nnUNet_preprocessed='${nnUNet_preprocessed}' nnUNet_results='${nnUNet_results}'
            cd '${PROJECT_ROOT}'
            .venv/bin/nnUNetv2_plan_and_preprocess -d ${ID} -c 3d_fullres --verify_dataset_integrity
        "
    SRC="${SPLITS_DIR}/splits_final.json"; DST="${nnUNet_preprocessed}/${DS_NAME}/splits_final.json"
    cp "${SRC}" "${DST}"; cmp -s "${SRC}" "${DST}" || { echo "splits copy differs: ${DST}" >&2; exit 1; }
    echo "Installed + byte-verified splits: ${SRC} -> ${DST}"
done
