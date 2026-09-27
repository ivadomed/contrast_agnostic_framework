#!/usr/bin/env bash
# Run inference for all 6 T1n experiments in parallel (all folds, all contrasts).
# Each experiment's folds are dispatched as independent Slurm jobs via run_job.
#
# Usage:
#   bash 05_16_predict_all_t1n.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

pids=()
bash "${HERE}/05_01_predict_t1n_baseline.sh"                  brats2024-glioma_t1n_baseline_20260622_044535 all &
pids+=($!)
bash "${HERE}/05_14_predict_t1n_v26_6_2_train050_val100.sh"   brats2024-glioma_t1n_v26_6_2_train050_val100_20260622_044535 all &
pids+=($!)
bash "${HERE}/05_05_predict_auglab_default.sh"                 brats2024-glioma_t1n_auglab_default_20260622_044535 all &
pids+=($!)
bash "${HERE}/05_08_predict_synthseg_EM.sh"                   brats2024-glioma_t1n_synthseg_EM_20260622_044535 all &
pids+=($!)
bash "${HERE}/05_07_predict_synthseg_noEM.sh"                 brats2024-glioma_t1n_synthseg_noEM_20260622_044535 all &
pids+=($!)
bash "${HERE}/05_15_predict_t1n_auglabAug_v26_6_2_train025_val100.sh" \
    brats2024-glioma_t1n_auglabAug_v26_6_2_train025_val100_20260622_044535 all &
pids+=($!)

wait "${pids[@]}"
echo "All T1n predict jobs done"
