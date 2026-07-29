#!/usr/bin/env bash
# Run ALL open-ms T1w-TRAINED models over MS3SEG (100 patients), all folds.
set -euo pipefail
HERE="$(dirname "$0")"
for wrapper in \
    05_09_predict_openms_t1w_baseline.sh 05_10_predict_openms_t1w_auglab_default.sh \
    05_11_predict_openms_t1w_synthseg_noEM.sh 05_12_predict_openms_t1w_synthseg_EM.sh \
    05_13_predict_openms_t1w_v26_6_2.sh 05_14_predict_openms_t1w_auglabAug_v26_6_2.sh; do
    echo ">>> [$(date '+%H:%M:%S')] ${wrapper}"
    bash "${HERE}/${wrapper}" "" all
done
echo "[$(date '+%H:%M:%S')] ALL T1w-trained methods done."
