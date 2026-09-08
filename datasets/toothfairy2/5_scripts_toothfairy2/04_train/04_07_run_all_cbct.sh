#!/usr/bin/env bash
# Launch the 6 usual experiments on toothfairy2 CBCT (Dataset110), 3 folds (0 1 2).
# Thin launcher — lists the 6 per-method wrappers in the standard order and hands
# them to the shared runner (00_01_train/run_all_train_common.sh), which caps folds.
#
# This is the one-job-per-fold path (Vulcan/Killarney/romane). On TamIA, whose GPUs
# are allocated by WHOLE NODE, use 04_12_tamia_pack_suite.sh instead — running this
# there would burn 3 of every 4 GPUs idle.
#
# Usage:
#   bash 04_07_run_all_cbct.sh                          # launch all 6
#   bash 04_07_run_all_cbct.sh --start-from synthseg_EM # resume the suite from a method
source "$(dirname "$0")/../00_utils/env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Order: baseline → auglab_default → synthseg_noEM → synthseg_EM → srcsm → OURS
METHOD_SCRIPTS=(
    "${HERE}/04_01_train_baseline.sh"
    "${HERE}/04_02_train_auglab_default.sh"
    "${HERE}/04_03_train_synthseg_noEM.sh"
    "${HERE}/04_04_train_synthseg_EM.sh"
    "${HERE}/04_05_train_srcsm.sh"
    "${HERE}/04_06_train_auglabAug_v26_6_2_dualval.sh"
)

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/run_all_train_common.sh" "$@"
