#!/usr/bin/env bash

source "$(dirname "$0")/../00_utils/env.sh"
# Train or resume SynthSeg-A (BrainGenerator mode A).
# Usage: bash 03_train_synthseg_a.sh [RUN_ID]
METHOD="synthseg_a"
TRAINER="nnUNetTrainerOnHarmonySynthSegA"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_synthseg_a"
# 4 folds × 2 threads each = 8 TF threads total, avoids CPU contention.
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
source "$(dirname "$0")/04_00_common.sh" "$@"
