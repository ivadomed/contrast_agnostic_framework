#!/usr/bin/env bash
# Launch the 6 usual experiments on ispy2 T1WCE, 3 folds (0 1 2). Thin launcher —
# just lists the 6 per-method wrappers in order and hands them to the shared runner
# (00_01_train/run_all_train_common.sh), which caps folds at 0 1 2. See
# 04_14_run_all_t2w.sh for the second training modality.
#
# Prereqs (run once, in order): 02_nnunet/'s t1wce converter,
# 03_preprocess/'s t1wce preprocess script.
#
# Usage:
#   bash 04_07_run_all_t1wce.sh                          # launch all 6
#   bash 04_07_run_all_t1wce.sh --start-from synthseg_EM   # resume the suite from a method
source "$(dirname "$0")/../00_utils/env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

# The 6 methods (order: baseline -> auglab_default -> synthseg_noEM -> synthseg_EM -> srcsm -> OURS dualval)
METHOD_SCRIPTS=(
    "${HERE}/04_01_train_t1wce_baseline.sh"
    "${HERE}/04_02_train_t1wce_auglab_default.sh"
    "${HERE}/04_03_train_t1wce_synthseg_noEM.sh"
    "${HERE}/04_04_train_t1wce_synthseg_EM.sh"
    "${HERE}/04_05_train_t1wce_srcsm.sh"
    "${HERE}/04_06_train_t1wce_auglabAug_v26_6_2_dualval.sh"
)

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/run_all_train_common.sh" "$@"
