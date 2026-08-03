#!/usr/bin/env bash
# Shared training template for picai-prostate — sourced by 04_XX_train_<modality>_<method>.sh,
# NOT invoked directly. Thin shim: sources env.sh (ADC wrappers pre-source env_adc.sh, which
# sets TRAINING_CONTRAST first so env.sh's `:-` default keeps it), applies the tamia
# scratch-path override, fills in the picai-prostate defaults, and delegates to the shared
# driver datasets/00_commun_scripts/00_01_train/train_common.sh (see that file for the full
# contract, per-method env vars, and RESUME NOTES).
#
# Per-method wrapper sets: METHOD, TRAINER, CATEGORY (nnUNet|auglab), DATASET_ID and — for
# AugLab methods — AUGLAB_PARAMS_GPU_JSON / AUGLAB_VAL_PARAMS_GPU_JSON.
#
# WHY NNUNET_RESULTS_BASE AND LOG_DIR ARE DERIVED **HERE**, not in each wrapper (the other
# datasets do it in the wrapper): on tamia all bulk paths move to $SCRATCH, and that override
# has to be applied AFTER env.sh (which exports nnUNet_results/PREDICTIONS_ROOT/RESULTS_DIR
# unconditionally — the documented gotcha in CLAUDE.md). A wrapper that expanded
# "${PREDICTIONS_ROOT}/..." at its own top would bake in the pre-override /project path and
# silently write checkpoints to the wrong filesystem. Deriving both after the override makes
# the wrappers order-independent. A wrapper may still set either var explicitly to win.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
[ -f "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh" ] && [ -d /scratch/p/paulh ] && \
    source "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh"
cd "${PROJECT_ROOT}"

DATASET_ID_DEFAULT="080"                # ADC wrappers override to 081
NNUNET_NUM_EPOCHS_DEFAULT="2000"        # picai-prostate epoch policy (CLAUDE.md EPOCH POLICY)
# iters-per-epoch: leave nnU-Net's built-in default (250) — do not set the default var.

export NNUNET_RESULTS_BASE="${NNUNET_RESULTS_BASE:-${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY:-nnUNet}}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/_logs/nnunet_picai-prostate_${TRAINING_CONTRAST}_${METHOD}}"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/train_common.sh" "$@"
