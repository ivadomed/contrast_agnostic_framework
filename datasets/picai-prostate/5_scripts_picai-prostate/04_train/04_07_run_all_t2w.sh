#!/usr/bin/env bash
# Launch the full method suite on picai-prostate T2W (Dataset080), 3 folds (0 1 2).
# Thin launcher — lists the 6 T2W per-method wrappers in order and hands them to the shared
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
#   bash 04_07_run_all_t2w.sh                       # launch all 6
#   bash 04_07_run_all_t2w.sh --start-from srcsm    # resume the suite from a method
source "$(dirname "$0")/../00_utils/env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

METHOD_SCRIPTS=(
    "${HERE}/04_01_train_t2w_baseline.sh"
    "${HERE}/04_02_train_t2w_auglab_default.sh"
    "${HERE}/04_03_train_t2w_synthseg_noEM.sh"
    "${HERE}/04_04_train_t2w_synthseg_EM.sh"
    "${HERE}/04_05_train_t2w_srcsm.sh"
    "${HERE}/04_06_train_t2w_auglabAug_v26_6_2_train050_val000.sh"
)

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/run_all_train_common.sh" "$@"
