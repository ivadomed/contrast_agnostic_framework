#!/usr/bin/env bash
# Evaluate ALL 6 open-ms FLAIR-TRAINED models on MS3SEG -- all folds, all 3 contrasts.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
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
    echo ">>> [$(date '+%H:%M:%S')] ${category}/${run_id}"
    bash "${HERE}/06_01_evaluate_run.sh" "$run_id" "$category" all
done
echo "[$(date '+%H:%M:%S')] All 6 FLAIR-trained methods evaluated."
