#!/usr/bin/env bash
# Shared training template for open-ms — sourced by 04_0X_train_<method>.sh, NOT invoked
# directly. Thin shim: sources env.sh, sets open-ms defaults, delegates to the shared
# driver benchmark/00_commun_scripts/00_01_train/train_common.sh (see that file for the
# full contract, per-method env vars, and RESUME NOTES).
#
# NOTE (2026-09-27): unlike every other headline task, open-ms's headline OURS method
# (val000/val100) is currently trained as TWO independent runs, not one DualVal run
# (see open_ms/trainers/auglab_dualval.py) -- 2x GPU cost. Backfilling DualVal here is
# a deliberate future decision requiring a retrain, not done in this pass.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

DATASET_ID_DEFAULT="070"
NNUNET_NUM_EPOCHS_DEFAULT="2000"        # long horizon for MS lesion segmentation
# iters-per-epoch: leave nnU-Net's built-in default (250) — do not set the default var.

source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_01_train/train_common.sh" "$@"
