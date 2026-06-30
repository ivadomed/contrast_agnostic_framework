#!/usr/bin/env bash
# Shared training template for brats2024-glioma — sourced by 04_0X_train_<method>.sh,
# NOT invoked directly. This is now a thin shim: it sources env.sh, sets the
# brats-specific defaults, and delegates to the shared driver
# datasets/00_commun_scripts/00_01_train/train_common.sh (see that file for the full
# contract, the per-method env vars, and the RESUME NOTES).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

DATASET_ID_DEFAULT="050"           # 4-channel; T1n wrappers override to 051
NNUNET_NUM_EPOCHS_DEFAULT="500"
# (no NNUNET_ITERS_PER_EPOCH_DEFAULT → keep nnU-Net's built-in iters/epoch default)

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/train_common.sh" "$@"
