#!/usr/bin/env bash
# Run ALL open-ms FLAIR-TRAINED models over MS3SEG (100 patients), all folds.
set -euo pipefail
HERE="$(dirname "$0")"
for wrapper in \
    05_02_predict_openms_baseline.sh 05_03_predict_openms_auglab_default.sh \
    05_04_predict_openms_synthseg_noEM.sh 05_05_predict_openms_synthseg_EM.sh \
    05_06_predict_openms_v26_6_2.sh 05_07_predict_openms_auglabAug_v26_6_2.sh; do
    echo ">>> [$(date '+%H:%M:%S')] ${wrapper}"
    bash "${HERE}/${wrapper}" "" all
done
echo "[$(date '+%H:%M:%S')] ALL FLAIR-trained methods done."
