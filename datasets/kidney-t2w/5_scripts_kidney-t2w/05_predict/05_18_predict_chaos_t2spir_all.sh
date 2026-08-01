#!/usr/bin/env bash
# Predict on KIDNEY-T2W with ALL 7 chaos T2spir-trained models (folds 0-2, single "t2" item).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================================="
echo " KIDNEY-T2W <- CHAOS T2spir models | predict 7 methods x folds 0-2"
echo "=================================================================="

bash "${HERE}/05_11_predict_chaos_t2spir_baseline.sh"
bash "${HERE}/05_12_predict_chaos_t2spir_synthseg_EM.sh"
bash "${HERE}/05_13_predict_chaos_t2spir_auglab_default.sh"
bash "${HERE}/05_14_predict_chaos_t2spir_synthseg_noEM.sh"
bash "${HERE}/05_15_predict_chaos_t2spir_srcsm.sh"
bash "${HERE}/05_16_predict_chaos_t2spir_auglabAug_v26_6_2_train050_val000.sh"
bash "${HERE}/05_17_predict_chaos_t2spir_auglabAug_v26_6_2_train050_val100.sh"

echo ""
echo "[$(date '+%H:%M:%S')] All T2spir predictions done."
