#!/usr/bin/env bash
# Launch the full method suite on picai-prostate ADC (Dataset081), 3 folds (0 1 2).
# Thin launcher — lists the 6 ADC per-method wrappers in order and hands them to the shared
# runner (00_01_train/run_all_train_common.sh), which caps folds at 0 1 2.
#
# This is the ONE-SBATCH-PER-FOLD path (vulcan / killarney / romane). On tamia, where GPUs
# are allocated only by whole node, use 04_21_tamia_pack_launch_all.sh instead — same
# wrappers, packed onto whole nodes.
#
# NOTE ON METHOD COUNT: 6 wrappers produce the 7 headline arms, because the last one is the
# DualVal trainer — one training run emits BOTH auglabAug_v26_6_2_train050_val000 and
# ..._val100 (see 04_06's header). Do NOT add a separate val100 training run.
#
# Usage:
#   bash 04_17_run_all_adc.sh                       # launch all 6
#   bash 04_17_run_all_adc.sh --start-from srcsm    # resume the suite from a method
source "$(dirname "$0")/../00_utils/env_adc.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

METHOD_SCRIPTS=(
    "${HERE}/04_11_train_adc_baseline.sh"
    "${HERE}/04_12_train_adc_auglab_default.sh"
    "${HERE}/04_13_train_adc_synthseg_noEM.sh"
    "${HERE}/04_14_train_adc_synthseg_EM.sh"
    "${HERE}/04_15_train_adc_srcsm.sh"
    "${HERE}/04_16_train_adc_auglabAug_v26_6_2_train050_val000.sh"
)

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/run_all_train_common.sh" "$@"
