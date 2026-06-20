#!/usr/bin/env bash
# T2w contrast override — source this instead of env.sh for T2w experiments.
# Sets TRAINING_CONTRAST and nnUNet_results BEFORE sourcing env.sh so that when
# 04_00_common.sh re-sources env.sh the conditional assignments (${VAR:-default})
# preserve these values rather than overwriting them with the t1n defaults.

export TRAINING_CONTRAST="t2w"
export nnUNet_results="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/nnUNet"

# T2w training jobs run for 2500 epochs; the longest auglab experiments take ~90s/epoch
# (~62h total). Override the default 24h Slurm time limit cluster-wide for this contrast.
export RUN_JOB_TIME_DEFAULT="6-23:00:00"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
