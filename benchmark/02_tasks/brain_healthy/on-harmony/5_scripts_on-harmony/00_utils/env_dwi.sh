#!/usr/bin/env bash
# dwi_ap (b~0 EPI, NOT diffusion-weighted) override — source this instead of env.sh for dwi_ap experiments
# (3rd training modality, added 2026-09-21). Mirrors env_t2w.sh exactly.
#
# Usage (from a step subdir):
#   source "$(dirname "$0")/../00_utils/env_dwi.sh"

export TRAINING_CONTRAST="dwi_ap"

# Same epoch budget/time as T1w/T2w (2000 epochs) — 3 days headroom.
export RUN_JOB_TIME_DEFAULT="3-00:00:00"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
