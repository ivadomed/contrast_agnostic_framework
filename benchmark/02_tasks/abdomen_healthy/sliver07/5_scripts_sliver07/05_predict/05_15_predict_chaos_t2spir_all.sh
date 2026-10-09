#!/usr/bin/env bash
# Run ALL chaos T2spir-trained models over SLIVER07 CT, all 4 folds each.
# Methods run SEQUENTIALLY (each method fans its 4 folds across the 4 GPUs/slots and
# waits) so we never oversubscribe the 4 GPUs. SLIVER07 has no models of its own --
# every model here is chaos-trained (cross-dataset). See 05_01_predict_common.sh.
#
#   bash 05_15_predict_chaos_t2spir_all.sh
set -euo pipefail
HERE="$(dirname "$0")"

echo "=================================================================="
echo " SLIVER07 <- CHAOS T2spir models | all 6 methods x 4 folds x CT"
echo "=================================================================="

for wrapper in \
    05_09_predict_chaos_t2spir_baseline.sh \
    05_10_predict_chaos_t2spir_v26_6_2.sh \
    05_11_predict_chaos_t2spir_synthseg_EM.sh \
    05_12_predict_chaos_t2spir_synthseg_noEM.sh \
    05_13_predict_chaos_t2spir_auglab_default.sh \
    05_14_predict_chaos_t2spir_auglabAug_v26_6_2.sh ; do
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${wrapper} (all folds)"
    bash "${HERE}/${wrapper}" "" all ct
done

echo ""
echo "[$(date '+%H:%M:%S')] ALL T2spir methods done."
