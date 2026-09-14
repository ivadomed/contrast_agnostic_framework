#!/usr/bin/env bash
# T2w contrast override — source this instead of env.sh for T2w experiments.
# Sets TRAINING_CONTRAST and nnUNet_results BEFORE sourcing env.sh so that when
# 04_00_common.sh re-sources env.sh the conditional assignments (${VAR:-default})
# preserve these values rather than overwriting them with the t1wce defaults.
# Same pattern as ambl/brats2024-glioma's env_t2w.sh / chaos's env_t2spir.sh.

export TRAINING_CONTRAST="t2w"
export nnUNet_results="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/8_results_ispy2/01_predictions/ispy2_model/t2w/nnUNet"

# Same 1000-epoch horizon as t1wce (see env.sh) — DEFERRED to a TamIA sizing
# probe: re-time per-modality once real epoch costs are measured on ispy2's
# actual (much larger than ambl's) cohort — don't assume ambl's ~14-35s/epoch
# transfers, per CLAUDE.md's "measure, don't assume" rule.
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-48:00:00}"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
