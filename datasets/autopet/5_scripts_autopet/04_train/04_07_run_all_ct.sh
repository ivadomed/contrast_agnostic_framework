#!/usr/bin/env bash
# Launch the 6 usual experiments on autopet CT (Dataset120), 3 folds (0 1 2).
# Thin launcher — lists the 6 per-method wrappers in the standard order and hands them to
# the shared runner (00_01_train/run_all_train_common.sh), which caps folds.
#
# This is the one-job-per-fold path (Vulcan/Killarney). On TamIA (whole-node H100
# allocation, the project's standing default for Slurm jobs — see CLAUDE.md "Slurm jobs
# go to TamIA, not Vulcan"), use the pack-mode orchestrator instead once a sizing probe
# has measured real per-fold cost on this dataset's whole-body volumes (see
# 04_08_tamia_sizing_probe.sh) — running this driver there would burn 3 of every 4 GPUs
# idle, exactly the failure mode CLAUDE.md warns about for TamIA.
#
# Usage:
#   bash 04_07_run_all_ct.sh                          # launch all 6
#   bash 04_07_run_all_ct.sh --start-from synthseg_EM # resume the suite from a method
source "$(dirname "$0")/../00_utils/env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Order: baseline -> auglab_default -> synthseg_noEM -> synthseg_EM -> srcsm -> OURS
METHOD_SCRIPTS=(
    "${HERE}/04_01_train_ct_baseline.sh"
    "${HERE}/04_02_train_ct_auglab_default.sh"
    "${HERE}/04_03_train_ct_synthseg_noEM.sh"
    "${HERE}/04_04_train_ct_synthseg_EM.sh"
    "${HERE}/04_05_train_ct_srcsm.sh"
    "${HERE}/04_06_train_ct_auglabAug_v26_6_2_dualval.sh"
)

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/run_all_train_common.sh" "$@"
