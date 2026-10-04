#!/usr/bin/env bash
# Shared training template for isles2022 — sourced by 04_XX_train_<contrast>_<method>.sh, NOT invoked
# directly. Thin shim: sources env.sh, sets isles2022 defaults, delegates to the shared driver
# benchmark/00_commun_scripts/00_01_train/train_common.sh (see that file for the contract, per-method
# env vars and RESUME NOTES).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

DATASET_ID_DEFAULT="${DATASET_ID_DWI}"   # real wrappers always pass DATASET_ID explicitly

# EPOCH POLICY: 2000 (Paul, 2026-10-04: "probably best for brain pathology", same as on-harmony/open-ms). Small cohort (~132 train cases/fold,
# 3D volumes ~112x112x73): re-time on a TamIA sizing probe before trusting RUN_JOB_TIME_DEFAULT / pack chain length.
NNUNET_NUM_EPOCHS_DEFAULT="2000"

source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_01_train/train_common.sh" "$@"
