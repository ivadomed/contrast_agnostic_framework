#!/usr/bin/env bash
# Launch the 6 usual experiments on atlas-liver-hcc T1w (Dataset080), 3 folds (0 1 2).
# Thin launcher — just lists the 6 per-method wrappers in order and hands them to the
# shared runner (00_01_train/run_all_train_common.sh), which caps folds at 0 1 2.
# Single-modality dataset (no second training contrast) -> this is the ONLY run-all
# launcher for atlas-liver-hcc, unlike open-ms/chaos/brats which have one per contrast.
#
# Prereqs (run once, in order): 02_nnunet/02_00_convert.py, 03_preprocess/03_00_preprocess.sh.
#
# Usage:
#   bash 04_07_run_all_t1w.sh                     # launch all 6
#   bash 04_07_run_all_t1w.sh --start-from synthseg_EM   # resume the suite from a method
source "$(dirname "$0")/../00_utils/env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

# The 6 methods (order: baseline -> auglab_default -> synthseg_noEM -> synthseg_EM -> srcsm -> OURS dualval)
METHOD_SCRIPTS=(
    "${HERE}/04_01_train_baseline.sh"
    "${HERE}/04_02_train_auglab_default.sh"
    "${HERE}/04_03_train_synthseg_noEM.sh"
    "${HERE}/04_04_train_synthseg_EM.sh"
    "${HERE}/04_05_train_srcsm.sh"
    "${HERE}/04_06_train_auglabAug_v26_6_2_dualval.sh"
)

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/run_all_train_common.sh" "$@"
