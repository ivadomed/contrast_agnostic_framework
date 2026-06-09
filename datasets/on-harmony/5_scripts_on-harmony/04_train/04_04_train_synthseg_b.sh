#!/usr/bin/env bash

source "$(dirname "$0")/../00_utils/env.sh"
# Train or resume SynthSeg-B (BrainGenerator mode B / EM mixture).
# Usage: bash 03_train_synthseg_b.sh [RUN_ID]
METHOD="synthseg_b"
TRAINER="nnUNetTrainerOnHarmonySynthSegB"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_synthseg_b"
# 4 folds × 2 threads each = 8 TF threads total, avoids CPU contention.
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
source "$(dirname "$0")/04_00_common.sh" "$@"
