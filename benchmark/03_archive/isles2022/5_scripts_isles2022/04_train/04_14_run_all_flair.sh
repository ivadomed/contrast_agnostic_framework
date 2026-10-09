#!/usr/bin/env bash
# Launch the 6 usual experiments on isles2022 FLAIR, 3 folds (0 1 2). See
# 04_07_run_all_dwi.sh for the first training modality.
#
# Usage:
#   bash 04_14_run_all_flair.sh
#   bash 04_14_run_all_flair.sh --start-from synthseg_EM
source "$(dirname "$0")/../00_utils/env_flair.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

METHOD_SCRIPTS=(
    "${HERE}/04_08_train_flair_baseline.sh"
    "${HERE}/04_09_train_flair_auglab_default.sh"
    "${HERE}/04_10_train_flair_synthseg_noEM.sh"
    "${HERE}/04_11_train_flair_synthseg_EM.sh"
    "${HERE}/04_12_train_flair_srcsm.sh"
    "${HERE}/04_13_train_flair_auglabAug_v26_6_2_dualval.sh"
)

source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_01_train/run_all_train_common.sh" "$@"
