#!/usr/bin/env bash
# Run ALL brats2024-glioma T1n-TRAINED models over BraTS-SSA 2024 (all 95 cases),
# all folds each, across all four test contrasts (T1n/T1c/T2w/T2f).
#   bash 05_14_predict_brats_t1n_all.sh
set -euo pipefail
HERE="$(dirname "$0")"
echo "=================================================================="
echo " BraTS-SSA 2024 <- brats2024-glioma T1n-TRAINED models | 6 methods x 3 folds x {t1n,t1c,t2w,t2f}"
echo "=================================================================="
for wrapper in \
    05_02_predict_brats_t1n_baseline.sh \
    05_03_predict_brats_t1n_auglab_default.sh \
    05_04_predict_brats_t1n_synthseg_noEM.sh \
    05_05_predict_brats_t1n_synthseg_EM.sh \
    05_06_predict_brats_t1n_auglabAug_v26_6_2_train050_val000.sh \
    05_07_predict_brats_t1n_auglabAug_v26_6_2_train050_val100.sh \
    ; do
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${wrapper} (all folds, all 4 contrasts, one job per fold)"
    bash "${HERE}/${wrapper}" "" all
done
echo ""
echo "[$(date '+%H:%M:%S')] ALL T1n-trained methods done."
