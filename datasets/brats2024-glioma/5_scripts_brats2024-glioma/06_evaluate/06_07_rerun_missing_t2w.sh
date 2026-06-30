#!/usr/bin/env bash
# One-shot: re-run only the T2W eval folds that are missing summaries.
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
export TRAINING_CONTRAST="t2w"
HERE="$(cd "$(dirname "$0")" && pwd)"

pids=()
bash "${HERE}/06_01_evaluate_run.sh" brats2024-glioma_t2w_synthseg_EM_20260620_125354 0 &
pids+=($!)
bash "${HERE}/06_01_evaluate_run.sh" brats2024-glioma_t2w_synthseg_EM_20260620_125354 3 &
pids+=($!)
bash "${HERE}/06_01_evaluate_run.sh" brats2024-glioma_t2w_synthseg_noEM_20260620_125442 all &
pids+=($!)
bash "${HERE}/06_01_evaluate_run.sh" brats2024-glioma_t2w_baseline_20260620_125115 3 &
pids+=($!)

any_failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || any_failed=1
done
[ "$any_failed" = "1" ] && echo "Some evals failed — check *_eval.log files" && exit 1
echo "Done"
