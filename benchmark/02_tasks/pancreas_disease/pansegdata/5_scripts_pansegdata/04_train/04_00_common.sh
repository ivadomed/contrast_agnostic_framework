#!/usr/bin/env bash
# Shared training template for pansegdata — sourced by 04_XX_train_<contrast>_<method>.sh, NOT invoked
# directly. Thin shim: sources env.sh, sets pansegdata defaults, delegates to the shared driver
# benchmark/00_commun_scripts/00_01_train/train_common.sh (see that file for the contract, per-method
# env vars and RESUME NOTES).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

DATASET_ID_DEFAULT="${DATASET_ID_T1WCE}"   # real wrappers always pass DATASET_ID explicitly

# EPOCH POLICY: 2000. Decided by analogy with the project's epoch table (2026-10-04, not a user-specified number for this dataset): a disease
# cohort (pancreatic cystic lesions / suspected PDAC) with <= ~300 training cases (113-114 train cases per fold, 170 in the pool) -> 2000,
# same as isles2022 / on-harmony / open-ms. Re-time from a real sizing probe on the target cluster (Killarney) before trusting
# RUN_JOB_TIME_DEFAULT; if the probe makes 2000 epochs wildly expensive, lower it explicitly and say so.
NNUNET_NUM_EPOCHS_DEFAULT="2000"

source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_01_train/train_common.sh" "$@"
