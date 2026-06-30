#!/usr/bin/env bash
# Shared training template for on-harmony — sourced by 04_0X_train_<method>.sh,
# NOT invoked directly. Thin shim: sources env.sh, sets the on-harmony-specific
# defaults, and delegates to the shared driver
# datasets/00_commun_scripts/00_01_train/train_common.sh (see that file for the full
# contract, the per-method env vars, and the RESUME NOTES).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

DATASET_ID_DEFAULT="031"          # 31-class T1w; T2w wrappers override via DATASET_ID
NNUNET_NUM_EPOCHS_DEFAULT="500"
# (no NNUNET_ITERS_PER_EPOCH_DEFAULT → keep nnU-Net's built-in iters/epoch default)
# No NNUNET_RESULTS_BASE here: nnUNet-family methods inherit env.sh's nnUNet_results
# (= .../01_predictions/<model>/<contrast>/nnUNet). auglab per-method scripts override
# NNUNET_RESULTS_BASE → .../<contrast>/auglab. Exactly the chaos category split.

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/train_common.sh" "$@"
