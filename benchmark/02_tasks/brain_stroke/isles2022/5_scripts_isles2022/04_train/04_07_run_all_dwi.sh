#!/usr/bin/env bash
# Launch the 6 usual experiments on isles2022 DWI, 3 folds (0 1 2). Thin launcher —
# just lists the 6 per-method wrappers in order and hands them to the shared runner
# (00_01_train/run_all_train_common.sh), which caps folds at 0 1 2. See
# 04_14_run_all_flair.sh for the second training modality.
#
# Prereqs (run once, in order): 02_nnunet/'s dwi converter,
# 03_preprocess/'s dwi preprocess script.
#
# Usage:
#   bash 04_07_run_all_dwi.sh                          # launch all 6
#   bash 04_07_run_all_dwi.sh --start-from synthseg_EM   # resume the suite from a method
source "$(dirname "$0")/../00_utils/env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

# The 6 methods (order: baseline -> auglab_default -> synthseg_noEM -> synthseg_EM -> srcsm -> OURS dualval)
METHOD_SCRIPTS=(
    "${HERE}/04_01_train_dwi_baseline.sh"
    "${HERE}/04_02_train_dwi_auglab_default.sh"
    "${HERE}/04_03_train_dwi_synthseg_noEM.sh"
    "${HERE}/04_04_train_dwi_synthseg_EM.sh"
    "${HERE}/04_05_train_dwi_srcsm.sh"
    "${HERE}/04_06_train_dwi_auglabAug_v26_6_2_dualval.sh"
)

source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_01_train/run_all_train_common.sh" "$@"
