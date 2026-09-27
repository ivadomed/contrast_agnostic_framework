#!/usr/bin/env bash
# Launch all 6 CHAOS T1in experiments (2500 epochs each, 4 folds each) in parallel.
# This replaces the earlier inconsistent T1in runs with a clean, uniform set.
#
# Usage: bash 04_39_launch_all_t1in.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[$(date '+%H:%M:%S')] === Launching all 6 CHAOS T1in experiments (2500 epochs, 4 folds each) ==="
echo ""

bash "${HERE}/04_33_train_t1in_baseline.sh" &
bash "${HERE}/04_34_train_t1in_v26_6_2_train050_val100.sh" &
bash "${HERE}/04_35_train_t1in_auglab_default.sh" &
bash "${HERE}/04_36_train_t1in_synthseg_EM.sh" &
bash "${HERE}/04_37_train_t1in_synthseg_noEM.sh" &
bash "${HERE}/04_38_train_t1in_auglabAug_v26_6_2_train025_val100.sh" &
wait

echo ""
echo "[$(date '+%H:%M:%S')] === All 6 T1in experiments submitted ==="
