#!/usr/bin/env bash
# Run ALL open-ms-trained FLAIR models over MSLesSeg (all 115 scans), all folds each,
# across all three test contrasts (FLAIR/T1w/T2w). MSLesSeg has no models of its own —
# every model here is open-ms-trained. See 05_01_predict_common.sh.
#   bash 05_08_predict_openms_all.sh
set -euo pipefail
HERE="$(dirname "$0")"
echo "=================================================================="
echo " MSLesSeg <- open-ms FLAIR models | 6 methods x 3 folds x {flair,t1w,t2w}"
echo "=================================================================="
for wrapper in \
    05_02_predict_openms_baseline.sh \
    05_03_predict_openms_auglab_default.sh \
    05_04_predict_openms_synthseg_noEM.sh \
    05_05_predict_openms_synthseg_EM.sh \
    05_06_predict_openms_v26_6_2.sh \
    05_07_predict_openms_auglabAug_v26_6_2.sh \
    ; do
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${wrapper} (all folds, all 3 contrasts, one job per fold)"
    bash "${HERE}/${wrapper}" "" all
done
echo ""
echo "[$(date '+%H:%M:%S')] ALL methods done."
