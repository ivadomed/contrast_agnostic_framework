#!/usr/bin/env bash
# Run ALL chaos-trained models over SLIVER07 CT, all 4 folds each.
# Methods run SEQUENTIALLY (each method fans its 4 folds across the 4 GPUs/slots and
# waits) so we never oversubscribe the 4 GPUs. SLIVER07 has no models of its own —
# every model here is chaos-trained (cross-dataset). See 05_01_predict_common.sh.
#
#   bash 05_07_predict_chaos_all.sh
set -euo pipefail
HERE="$(dirname "$0")"

echo "=================================================================="
echo " SLIVER07 ← CHAOS models | all 5 methods × 4 folds × CT"
echo "=================================================================="

for wrapper in \
    05_02_predict_chaos_baseline.sh \
    05_03_predict_chaos_v26_6_2.sh \
    05_04_predict_chaos_synthseg_EM.sh \
    05_05_predict_chaos_auglab_default.sh \
    05_06_predict_chaos_synthseg_noEM.sh \
    05_08_predict_chaos_auglabAug_v26_6_2.sh ; do
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${wrapper} (all folds)"
    bash "${HERE}/${wrapper}" "" all ct
done

echo ""
echo "[$(date '+%H:%M:%S')] ALL methods done."
