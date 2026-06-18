#!/usr/bin/env bash
# Run ALL chaos-trained models over AMOS CT+MRI, all 4 folds each.
# Methods run SEQUENTIALLY (each fans 4 folds across the 4 GPUs and waits).
# AMOS has no models of its own — every model here is chaos-trained (cross-dataset).
# See 05_01_predict_chaos_common.sh for the cross-dataset inference rationale.
#
#   bash 05_07_predict_chaos_all.sh
#   bash 05_07_predict_chaos_all.sh "" all ct    # CT only (skip MRI)
set -euo pipefail
HERE="$(dirname "$0")"

echo "=================================================================="
echo " AMOS ← CHAOS models | all 5 methods × 4 folds × CT+MRI"
echo "=================================================================="

for wrapper in \
    05_02_predict_chaos_baseline.sh \
    05_03_predict_chaos_v26_6_2.sh \
    05_04_predict_chaos_synthseg_EM.sh \
    05_05_predict_chaos_auglab_default.sh \
    05_06_predict_chaos_synthseg_noEM.sh ; do
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${wrapper}"
    bash "${HERE}/${wrapper}" "${1:-}" "${2:-all}" ${3:+$3}
done

echo ""
echo "[$(date '+%H:%M:%S')] ALL methods done."
