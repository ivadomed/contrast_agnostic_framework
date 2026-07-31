#!/usr/bin/env bash
# Run ALL chaos T1in-trained models over CIRRMRI-LIVER T1w+T2w, folds 0 1 2 each
# (project fold policy — see CLAUDE.md). Methods run SEQUENTIALLY. CIRRMRI-LIVER
# has no models of its own — every model here is chaos-trained (cross-dataset).
#
#   bash 05_10_predict_chaos_all.sh
set -euo pipefail
HERE="$(dirname "$0")"

echo "=================================================================="
echo " CIRRMRI-LIVER <- CHAOS T1in models | all 8 runs x folds 0-2 x {t1,t2}"
echo "=================================================================="

for wrapper in \
    05_02_predict_chaos_baseline.sh \
    05_03_predict_chaos_v26_6_2.sh \
    05_04_predict_chaos_synthseg_EM.sh \
    05_05_predict_chaos_auglab_default.sh \
    05_06_predict_chaos_synthseg_noEM.sh \
    05_07_predict_chaos_srcsm.sh \
    05_08_predict_chaos_auglabAug_v26_6_2_train050_val000.sh \
    05_09_predict_chaos_auglabAug_v26_6_2_train050_val100.sh ; do
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${wrapper} (all folds)"
    bash "${HERE}/${wrapper}" "" all t1 t2
done

echo ""
echo "[$(date '+%H:%M:%S')] ALL T1in methods done."
