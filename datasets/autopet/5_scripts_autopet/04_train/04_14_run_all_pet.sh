#!/usr/bin/env bash
# Launch the 6 usual experiments on autopet PET (Dataset121), 3 folds (0 1 2).
# Same shape as 04_07_run_all_ct.sh — see its header for the TamIA pack-mode note.
#
# Usage:
#   bash 04_14_run_all_pet.sh
#   bash 04_14_run_all_pet.sh --start-from synthseg_EM
source "$(dirname "$0")/../00_utils/env_pet.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

METHOD_SCRIPTS=(
    "${HERE}/04_08_train_pet_baseline.sh"
    "${HERE}/04_09_train_pet_auglab_default.sh"
    "${HERE}/04_10_train_pet_synthseg_noEM.sh"
    "${HERE}/04_11_train_pet_synthseg_EM.sh"
    "${HERE}/04_12_train_pet_srcsm.sh"
    "${HERE}/04_13_train_pet_auglabAug_v26_6_2_dualval.sh"
)

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/run_all_train_common.sh" "$@"
