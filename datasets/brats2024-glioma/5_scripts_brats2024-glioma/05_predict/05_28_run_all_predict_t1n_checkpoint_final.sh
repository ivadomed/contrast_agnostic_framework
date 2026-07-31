#!/usr/bin/env bash
# Re-predict the t1n 6-method suite's EXISTING checkpoint_best RUN_IDs at
# checkpoint_final.pth, for the checkpoint_best vs checkpoint_final comparison
# (see 06_evaluate/configs/brats_t1n_01_results_*_ckpt.yaml). srcsm has no checkpoint_final.pth on disk (never trained with one kept) -- left out here; its final-checkpoint row is a documented blank in the comparison configs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

METHOD_SCRIPTS=(
    "${HERE}/05_01_predict_t1n_baseline.sh"
    "${HERE}/05_07_predict_synthseg_noEM.sh"
    "${HERE}/05_08_predict_synthseg_EM.sh"
    "${HERE}/05_05_predict_auglab_default.sh"
    "${HERE}/05_24_predict_t1n_auglabAug_v26_6_2_train050_val100.sh"
)
METHOD_RUN_IDS=(
    "brats2024-glioma_t1n_baseline_20260622_044535"
    "brats2024-glioma_t1n_synthseg_noEM_20260622_044535"
    "brats2024-glioma_t1n_synthseg_EM_20260622_044535"
    "brats2024-glioma_t1n_auglab_default_20260622_044535"
    "brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val100_20260725_113540"
)
export CHECKPOINT=checkpoint_final.pth
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/run_all_predict_checkpoint_sweep.sh"
