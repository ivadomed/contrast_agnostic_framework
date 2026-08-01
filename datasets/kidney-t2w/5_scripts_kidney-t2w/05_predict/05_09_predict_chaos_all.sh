#!/usr/bin/env bash
# Predict on KIDNEY-T2W with ALL 7 chaos T1in-trained models (folds 0-2, single "t2" item).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================================="
echo " KIDNEY-T2W <- CHAOS T1in models | predict 7 methods x folds 0-2"
echo "=================================================================="

bash "${HERE}/05_02_predict_chaos_baseline.sh"
bash "${HERE}/05_03_predict_chaos_synthseg_EM.sh"
bash "${HERE}/05_04_predict_chaos_auglab_default.sh"
bash "${HERE}/05_05_predict_chaos_synthseg_noEM.sh"
bash "${HERE}/05_06_predict_chaos_srcsm.sh"
bash "${HERE}/05_07_predict_chaos_auglabAug_v26_6_2_train050_val000.sh"
bash "${HERE}/05_08_predict_chaos_auglabAug_v26_6_2_train050_val100.sh"

echo ""
echo "[$(date '+%H:%M:%S')] All T1in predictions done."
