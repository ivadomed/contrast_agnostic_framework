#!/usr/bin/env bash
# Shared training template for ispy2 — sourced by 04_0X_train_<contrast>_<method>.sh,
# NOT invoked directly. Thin shim: sources env.sh, sets ispy2 defaults, delegates to
# the shared driver datasets/00_commun_scripts/00_01_train/train_common.sh (see that
# file for the full contract, per-method env vars, and RESUME NOTES).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

# Parameterized (unlike ambl, which repeats 090/091 literally in ~40 files) —
# env.sh's DATASET_ID_T1WCE/DATASET_ID_T2W are the single source of truth.
DATASET_ID_DEFAULT="${DATASET_ID_T1WCE}"

# EPOCH POLICY: 1000 — a deliberate, user-set decision for this retrain
# (2026-09-04 pivot: ispy2 replaces ambl as the training set), NOT ambl's 2000.
# ispy2's cohort (~560 patients, ~1120 nnU-Net cases counting both FOV variants)
# is far larger than ambl's 63 training cases/fold, which is exactly why ambl
# overfit badly (baseline beat every augmented method in-domain on both
# contrasts) — 1000 epochs hedges against repeating that on a still-modest,
# if much larger, dataset. Re-time via a TamIA sizing probe before trusting
# RUN_JOB_TIME_DEFAULT in env.sh/env_t2w.sh; don't assume ambl's per-epoch cost
# transfers (different volumes, different patch size from nnU-Net's own plan).
NNUNET_NUM_EPOCHS_DEFAULT="1000"
# iters-per-epoch: leave nnU-Net's built-in default (250) — do not set the default var.

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/train_common.sh" "$@"
