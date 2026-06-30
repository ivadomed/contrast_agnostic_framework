#!/usr/bin/env bash
# Launch all 6 BraTS 2024 Glioma T1n experiments (2500 epochs each, 4 folds each) in parallel.
# Mirrors the T2w set (04_13–04_18) for T1n (Dataset051).
#
# Usage: bash 04_25_launch_all_t1n.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[$(date '+%H:%M:%S')] === Launching all 6 BraTS T1n experiments (2500 epochs, 4 folds each) ==="
echo ""

bash "${HERE}/04_19_train_t1n_baseline.sh" &
bash "${HERE}/04_20_train_t1n_v26_6_2_train050_val100.sh" &
bash "${HERE}/04_21_train_t1n_auglab_default.sh" &
bash "${HERE}/04_22_train_t1n_synthseg_EM.sh" &
bash "${HERE}/04_23_train_t1n_synthseg_noEM.sh" &
bash "${HERE}/04_24_train_t1n_auglabAug_v26_6_2_train025_val100.sh" &
wait

echo ""
echo "[$(date '+%H:%M:%S')] === All 6 T1n experiments submitted ==="
