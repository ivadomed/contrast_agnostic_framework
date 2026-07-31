#!/usr/bin/env bash
# Backfill checkpoint_best predict+eval for the T2w 6-method suite (on-harmony's
# predict step never passed -chk, so its existing untagged results are already
# checkpoint_final -- this backfills the missing checkpoint_best side; see
# 06_evaluate/configs/on-harmony_T2w_*_ckpt.yaml). 06_01_evaluate_testset.sh combines
# predict+eval in one step and reads CHECKPOINT directly (predates CKPT_TAG), so
# this launcher translates CKPT_TAG -> CHECKPOINT itself before sourcing the shared
# driver, rather than the driver guessing dataset-specific filename conventions.
# NOTE: on tamia, run each RUN_ID's folds SEQUENTIALLY on ONE GPU (export
# CUDA_VISIBLE_DEVICES=0 first) if invoking this outside the normal per-fold
# run_job dispatch -- a manually node-packed concurrent invocation hit a CUDA
# visibility race here before (silently dropped contrasts, see git history).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env_t2w.sh"
cd "${PROJECT_ROOT}"

EVAL_SCRIPT="${HERE}/06_01_evaluate_testset.sh"
EVAL_ARGS=(
    "on-harmony_T2w_baseline_20260624_191152"
    "on-harmony_T2w_synthseg_noEM_20260624_191418"
    "on-harmony_T2w_synthseg_EM_20260624_191329"
    "on-harmony_T2w_auglab_default_20260625_143115"
    "on-harmony_T2w_srcsm_20260709_122145"
    "on-harmony_T2w_auglabAug_v26_6_2_train050_val000_20260710_040443"
    "on-harmony_T2w_auglabAug_v26_6_2_train050_val100_20260727_075205"
)
export CKPT_TAG=best
export CHECKPOINT="checkpoint_${CKPT_TAG}.pth"
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/run_all_evaluate_checkpoint_sweep.sh"
