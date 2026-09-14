#!/usr/bin/env bash
# nnUNet plan & preprocess (with --verify_dataset_integrity) for BOTH ispy2
# training datasets, then (re)install our own patient-level, FOV-stratified
# 3-fold splits.
#
#   bash 02_04_plan_and_preprocess.sh            # both
#   bash 02_04_plan_and_preprocess.sh t1wce      # one modality only
#
# Two things this does differently from ambl's 03_0X_preprocess_*.sh:
#  1. The two ispy2 datasets do NOT share one splits_final.json. A case here is a
#     (patient, FOV) pair, and t1wce has bilateral variants for only the 122
#     natively-bilateral patients, so the t1wce case set is a strict subset of the
#     t2w one. 4_splits_ispy2/ therefore carries splits_final_t1wce.json (580
#     cases) and splits_final_t2w.json (952 cases) -- the patient-level partition
#     underneath is identical, only the case expansion differs. splits_final.json
#     in that dir is the t2w one (byte-identical to splits_final_t2w.json).
#  2. Only the 3d_fullres configuration is preprocessed (-c 3d_fullres). This
#     project trains 3D ONLY: the shared driver
#     datasets/00_commun_scripts/00_01_train/train_common.sh hardcodes
#     `nnUNetv2_train ${DATASET_ID} 3d_fullres ${FOLD}` and builds
#     TRAINER_DIR=.../${TRAINER}__nnUNetPlans__3d_fullres, so nnUNetPlans_2d and
#     nnUNetPlans_3d_lowres are never read by anything downstream (the TamIA
#     transfer script already excludes nnUNetPlans_2d when copying). ambl paid to
#     generate all three for 84 cases; ispy2 has ~1530 training cases across the
#     two Datasets, so that waste is no longer tolerable in wall-clock or in disk
#     on a quota-managed filesystem. PLANNING still covers every configuration
#     (nnUNetPlans.json is written in full) -- only the preprocessing pass is
#     restricted.
#  3. Splits are copied AFTER preprocessing and then byte-verified, because
#     nnUNetv2_plan_and_preprocess writes into the same preprocessed dir and can
#     clobber a pre-placed splits file.
#
# CPU-only (preprocessing needs no GPU) -> avoids the GPU queue entirely.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}"
MODALITIES=("${@:-}")
[ -z "${MODALITIES[0]:-}" ] && MODALITIES=(t1wce t2w)

declare -A DS_ID=( [t1wce]="${DATASET_ID_T1WCE:-100}" [t2w]="${DATASET_ID_T2W:-101}" )

for MOD in "${MODALITIES[@]}"; do
    ID="${DS_ID[$MOD]}"
    DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${ID}_" | head -1)"
    [ -n "${DS_NAME}" ] || { echo "no raw dir for dataset id ${ID} in ${nnUNet_raw}" >&2; exit 1; }

    run_job --name "ispy2_preprocess_${MOD}" --gpus 0 --cpus 8 --mem 48G --time 08:00:00 \
        --slot 0 --wait --log "${LOG_DIR}/preprocess_${MOD}_$(date +%Y%m%d_%H%M%S).log" -- bash -c "
            export nnUNet_raw='${nnUNet_raw}'
            export nnUNet_preprocessed='${nnUNet_preprocessed}'
            export nnUNet_results='${nnUNet_results}'
            cd '${PROJECT_ROOT}'
            .venv/bin/nnUNetv2_plan_and_preprocess -d ${ID} -c 3d_fullres --verify_dataset_integrity
        "

    SRC="${SPLITS_DIR}/splits_final_${MOD}.json"
    DST="${nnUNet_preprocessed}/${DS_NAME}/splits_final.json"
    cp "${SRC}" "${DST}"
    cmp -s "${SRC}" "${DST}" || { echo "splits copy differs: ${DST}" >&2; exit 1; }
    echo "Installed + byte-verified splits: ${SRC} -> ${DST}"
done
