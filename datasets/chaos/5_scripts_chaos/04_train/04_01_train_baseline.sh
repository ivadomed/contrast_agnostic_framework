#!/usr/bin/env bash
# Train (or resume) the CHAOS MR T1-in baseline (real data, no synthesis).
# Usage: bash 04_01_train_baseline.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerCHAOSBaseline"
DATASET_ID="060"
# 4 folds run concurrently (2 per GPU) sharing 64 CPUs → 16 DA workers/fold.
DA_WORKERS="${DA_WORKERS:-16}"
LOG_DIR="/tmp/nnunet_chaos_baseline"
source "$(dirname "$0")/04_00_common.sh" "$@"
