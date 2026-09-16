#!/usr/bin/env bash
# Shared training template for totalseg-pelvic — sourced by 04_XX_train_<method>.sh, NOT
# invoked directly. Thin shim: sources env.sh (or env_mri.sh, already done by the wrapper
# before this), sets the totalseg-pelvic defaults, and delegates to the shared driver
# datasets/00_commun_scripts/00_01_train/train_common.sh (see that file for the full
# contract, per-method env vars, and RESUME NOTES).
set -euo pipefail
cd "${PROJECT_ROOT:?source 00_utils/env.sh (or env_mri.sh) BEFORE this shim}"

# DATASET_ID is set by each wrapper (130=ct, 131=mri) — no single default here since
# both modalities share this shim.
: "${DATASET_ID:?each 04_0X_train_<method>[_mri].sh wrapper must set DATASET_ID}"

# EPOCH POLICY: 200 (see 00_utils/env.sh — chaos-analog default).
NNUNET_NUM_EPOCHS_DEFAULT="200"
# iters-per-epoch: leave nnU-Net's built-in default — do not set the default var.

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/train_common.sh" "$@"
