#!/usr/bin/env bash
# Evaluate ALL 6 open-ms T1w-TRAINED models on MS3SEG -- all folds, all 3 contrasts.
set -euo pipefail
export OPENMS_TRAINING_CONTRAST="t1w"
export OPENMS_DATASET_ID="71"
export OPENMS_DS_NAME="Dataset071_OpenMS_T1W"
HERE="$(cd "$(dirname "$0")" && pwd)"
declare -a RUNS_CATS=(
    "open-ms_t1w_baseline_20260708_083441 nnUNet"
    "open-ms_t1w_auglab_default_20260708_083511 auglab"
    "open-ms_t1w_synthseg_noEM_train100_val000_20260708_083541 auglab"
    "open-ms_t1w_synthseg_EM_train100_val000_20260708_083611 auglab"
    "open-ms_t1w_v26_6_2_train050_val100_20260708_083641 nnUNet"
    "open-ms_t1w_auglabAug_v26_6_2_train025_val100_20260708_083711 auglab"
)
for entry in "${RUNS_CATS[@]}"; do
    read -r run_id category <<< "$entry"
    echo ">>> [$(date '+%H:%M:%S')] ${category}/${run_id}"
    bash "${HERE}/06_01_evaluate_run.sh" "$run_id" "$category" all
done
echo "[$(date '+%H:%M:%S')] All 6 T1w-trained methods evaluated."
