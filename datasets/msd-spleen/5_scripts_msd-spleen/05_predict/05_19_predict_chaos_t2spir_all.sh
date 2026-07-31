#!/usr/bin/env bash
# Run ALL chaos T2spir-trained models over MSD-SPLEEN CT, folds 0 1 2 each (project
# fold policy — see CLAUDE.md). Methods run SEQUENTIALLY. MSD-SPLEEN has no models of
# its own — every model here is chaos-trained (cross-dataset). See 05_01_predict_common.sh.
#
#   bash 05_19_predict_chaos_t2spir_all.sh
set -euo pipefail
HERE="$(dirname "$0")"

echo "=================================================================="
echo " MSD-SPLEEN <- CHAOS T2spir models | all 8 runs x folds 0-2 x CT"
echo "=================================================================="

for wrapper in \
    05_11_predict_chaos_t2spir_baseline.sh \
    05_12_predict_chaos_t2spir_v26_6_2.sh \
    05_13_predict_chaos_t2spir_synthseg_EM.sh \
    05_14_predict_chaos_t2spir_auglab_default.sh \
    05_15_predict_chaos_t2spir_synthseg_noEM.sh \
    05_16_predict_chaos_t2spir_srcsm.sh \
    05_17_predict_chaos_t2spir_auglabAug_v26_6_2_train050_val000.sh \
    05_18_predict_chaos_t2spir_auglabAug_v26_6_2_train050_val100.sh ; do
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${wrapper} (all folds)"
    bash "${HERE}/${wrapper}" "" all ct
done

echo ""
echo "[$(date '+%H:%M:%S')] ALL T2spir methods done."
