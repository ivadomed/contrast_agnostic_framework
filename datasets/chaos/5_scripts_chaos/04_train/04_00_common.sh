#!/usr/bin/env bash
# Shared training template for chaos — sourced by 04_0X_train_<method>.sh, NOT invoked
# directly. This is now a thin shim: it sources env.sh, sets the chaos-specific defaults,
# and delegates to the shared driver
# datasets/00_commun_scripts/00_01_train/train_common.sh (see that file for the full
# contract, the per-method env vars, and the RESUME NOTES).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

DATASET_ID_DEFAULT="060"
NNUNET_NUM_EPOCHS_DEFAULT="250"
NNUNET_ITERS_PER_EPOCH_DEFAULT="150"   # chaos uses shorter epochs

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/train_common.sh" "$@"
