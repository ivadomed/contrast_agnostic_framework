#!/usr/bin/env bash
# T2w contrast override — source this instead of env.sh for T2w experiments.
# Sets TRAINING_CONTRAST and RUN_JOB_TIME_DEFAULT BEFORE sourcing env.sh so that when
# 04_00_common.sh re-sources env.sh the conditional assignments (${VAR:-default})
# preserve these values rather than overwriting them with the T1w defaults.
#
# Usage (from a step subdir):
#   source "$(dirname "$0")/../00_utils/env_t2w.sh"

export TRAINING_CONTRAST="T2w"

# T2w training jobs run for 2000 epochs (same as T1w) — 3 days is enough headroom.
export RUN_JOB_TIME_DEFAULT="3-00:00:00"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
