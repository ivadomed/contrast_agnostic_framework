#!/usr/bin/env bash
# Evaluate ALL 6 open-ms-trained models on MSLesSeg -- all folds, all 3 contrasts.
# Thin entry point over 06_01_evaluate_run.sh (RUN_ID/CATEGORY pairs match
# configs/mslesseg_01_results.yaml so 06_10's aggregate picks these straight up).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================================="
echo " MSLesSeg <- open-ms FLAIR models | evaluate 6-method suite"
echo "=================================================================="

declare -a RUNS_CATS=(
    "open-ms_flair_baseline_20260706_061243 nnUNet"
    "open-ms_flair_auglab_default_20260706_061243 auglab"
    "open-ms_flair_synthseg_noEM_train100_val000_20260706_061243 auglab"
    "open-ms_flair_synthseg_EM_train100_val000_20260706_061243 auglab"
    "open-ms_flair_v26_6_2_train050_val100_20260706_061243 nnUNet"
    "open-ms_flair_auglabAug_v26_6_2_train025_val100_20260706_061243 auglab"
)

for entry in "${RUNS_CATS[@]}"; do
    read -r run_id category <<< "$entry"
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${category}/${run_id}"
    bash "${HERE}/06_01_evaluate_run.sh" "$run_id" "$category" all
done

echo ""
echo "[$(date '+%H:%M:%S')] All 6 methods evaluated."
