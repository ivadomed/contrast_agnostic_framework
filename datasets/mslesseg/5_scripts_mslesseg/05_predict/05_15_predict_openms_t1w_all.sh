#!/usr/bin/env bash
# Run ALL open-ms T1w-TRAINED models over MSLesSeg (all 115 scans), all folds each,
# across all three test contrasts (FLAIR/T1w/T2w). See 05_08_predict_openms_all.sh
# for the FLAIR-trained analog.
#   bash 05_15_predict_openms_t1w_all.sh
set -euo pipefail
HERE="$(dirname "$0")"
echo "=================================================================="
echo " MSLesSeg <- open-ms T1w-TRAINED models | 6 methods x 3 folds x {flair,t1w,t2w}"
echo "=================================================================="
for wrapper in \
    05_09_predict_openms_t1w_baseline.sh \
    05_10_predict_openms_t1w_auglab_default.sh \
    05_11_predict_openms_t1w_synthseg_noEM.sh \
    05_12_predict_openms_t1w_synthseg_EM.sh \
    05_13_predict_openms_t1w_v26_6_2.sh \
    05_14_predict_openms_t1w_auglabAug_v26_6_2.sh \
    ; do
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${wrapper} (all folds, all 3 contrasts, one job per fold)"
    bash "${HERE}/${wrapper}" "" all
done
echo ""
echo "[$(date '+%H:%M:%S')] ALL methods done."
