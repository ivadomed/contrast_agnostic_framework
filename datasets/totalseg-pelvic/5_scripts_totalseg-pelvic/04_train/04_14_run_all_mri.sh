#!/usr/bin/env bash
# Launch the 6 usual experiments on totalseg-pelvic MRI (Dataset131), 3 folds (0 1 2).
# Same shape as 04_07_run_all_ct.sh — see that file's header for backend notes.
#
# Usage:
#   bash 04_14_run_all_mri.sh
#   bash 04_14_run_all_mri.sh --start-from synthseg_EM
source "$(dirname "$0")/../00_utils/env_mri.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

METHOD_SCRIPTS=(
    "${HERE}/04_08_train_mri_baseline.sh"
    "${HERE}/04_09_train_mri_auglab_default.sh"
    "${HERE}/04_10_train_mri_synthseg_noEM.sh"
    "${HERE}/04_11_train_mri_synthseg_EM.sh"
    "${HERE}/04_12_train_mri_srcsm.sh"
    "${HERE}/04_13_train_mri_auglabAug_v26_6_2_dualval.sh"
)

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/run_all_train_common.sh" "$@"
