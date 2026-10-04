#!/usr/bin/env bash
# Shared training template for isles2022 — sourced by 04_XX_train_<contrast>_<method>.sh, NOT invoked
# directly. Thin shim: sources env.sh, sets isles2022 defaults, delegates to the shared driver
# benchmark/00_commun_scripts/00_01_train/train_common.sh (see that file for the contract, per-method
# env vars and RESUME NOTES).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

DATASET_ID_DEFAULT="${DATASET_ID_DWI}"   # real wrappers always pass DATASET_ID explicitly

# EPOCH POLICY: 1000 (provisional, matches ispy2). Cohort is small (~132 train cases/fold, 3D volumes
# ~112x112x73) so overfitting is the risk, not under-training. NOT yet confirmed by Paul and NOT timed:
# re-decide after a TamIA sizing probe (per-epoch cost + VRAM, N folds sharing one GPU).
NNUNET_NUM_EPOCHS_DEFAULT="1000"

source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_01_train/train_common.sh" "$@"
