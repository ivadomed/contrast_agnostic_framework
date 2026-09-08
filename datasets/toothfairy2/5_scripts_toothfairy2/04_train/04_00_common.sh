#!/usr/bin/env bash
# Shared training template for toothfairy2 — sourced by 04_XX_train_<method>.sh, NOT
# invoked directly. Thin shim: sources env.sh, sets the toothfairy2 defaults, and
# delegates to the shared driver datasets/00_commun_scripts/00_01_train/train_common.sh
# (see that file for the full contract, per-method env vars, and RESUME NOTES).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

# env.sh's DATASET_ID_CBCT is the single source of truth for the nnU-Net id.
DATASET_ID_DEFAULT="${DATASET_ID_CBCT}"

# EPOCH POLICY: 1000 — see 00_utils/env.sh's rationale (a ~408-case training pool,
# matching the horizon ispy2 settled on for a cohort of that size; NOT a copy of
# chaos's 200, which is tuned to a 20-case cohort).
NNUNET_NUM_EPOCHS_DEFAULT="1000"
# iters-per-epoch: leave nnU-Net's built-in default (250) — do not set the default var.

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/train_common.sh" "$@"
