#!/usr/bin/env bash
# Launch the 6 usual experiments on open-ms T1w (Dataset071), 3 folds (0 1 2).
# Thin launcher — just lists the 6 T1w per-method wrappers in order and hands them to the
# shared runner (00_01_train/run_all_train_common.sh), which caps folds at 0 1 2.
#
# Prereqs (run once, in order): 02_nnunet/02_01_convert_t1w.sh, 03_preprocess/03_01_preprocess_t1w.sh.
#
# Usage:
#   bash 04_19_run_all_t1w.sh                     # launch all 6
#   bash 04_19_run_all_t1w.sh --start-from synthseg_EM   # resume the suite from a method
source "$(dirname "$0")/../00_utils/env_t1w.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

# The 6 methods (order: baseline → auglab_default → synthseg_noEM → synthseg_EM → v26 alone → OURS)
METHOD_SCRIPTS=(
    "${HERE}/04_13_train_t1w_baseline.sh"
    "${HERE}/04_14_train_t1w_auglab_default.sh"
    "${HERE}/04_15_train_t1w_synthseg_noEM.sh"
    "${HERE}/04_16_train_t1w_synthseg_EM.sh"
    "${HERE}/04_17_train_t1w_v26_6_2_train050_val100.sh"
    "${HERE}/04_18_train_t1w_auglabAug_v26_6_2_train025_val100.sh"
)

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/run_all_train_common.sh" "$@"
