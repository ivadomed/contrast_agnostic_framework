#!/usr/bin/env bash
# Launch the 6 usual experiments on open-ms FLAIR (Dataset070), 3 folds (0 1 2).
# Thin launcher — just lists the 6 FLAIR per-method wrappers in order and hands them to the
# shared runner (00_01_train/run_all_train_common.sh), which caps folds at 0 1 2.
#
# Usage:
#   bash 04_12_run_all_flair.sh                     # launch all 6
#   bash 04_12_run_all_flair.sh --start-from synthseg_EM   # resume the suite from a method
source "$(dirname "$0")/../00_utils/env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

# The 6 methods (order: baseline → auglab_default → synthseg_noEM → synthseg_EM → v26 alone → OURS)
METHOD_SCRIPTS=(
    "${HERE}/04_01_train_baseline.sh"
    "${HERE}/04_04_train_auglab_default.sh"
    "${HERE}/04_05_train_synthseg_noEM.sh"
    "${HERE}/04_02_train_synthseg_EM.sh"
    "${HERE}/04_06_train_v26_6_2_train050_val100.sh"
    "${HERE}/04_03_train_auglabAug_v26_6_2_train025_val100.sh"
    # srcsm = added 7th comparison arm (AugLab-category); can also be run standalone via 04_20_train_srcsm.sh
    "${HERE}/04_20_train_srcsm.sh"
)

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/run_all_train_common.sh" "$@"
