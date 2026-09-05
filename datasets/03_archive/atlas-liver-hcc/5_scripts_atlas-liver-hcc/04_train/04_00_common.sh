#!/usr/bin/env bash
# Shared training template for atlas-liver-hcc — sourced by 04_0X_train_<method>.sh, NOT
# invoked directly. Thin shim: sources env.sh, sets atlas-liver-hcc defaults, delegates to
# the shared driver datasets/00_commun_scripts/00_01_train/train_common.sh (see that file
# for the full contract, per-method env vars, and RESUME NOTES).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

DATASET_ID_DEFAULT="080"
NNUNET_NUM_EPOCHS_DEFAULT="2000"        # same horizon as open-ms/on-harmony
# iters-per-epoch: leave nnU-Net's built-in default (250) — do not set the default var.

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/train_common.sh" "$@"
