#!/usr/bin/env bash
# Run ALL chaos-trained t1in models over TRUSTED (CT + US), all 4 folds each.
# Methods run SEQUENTIALLY (each fans its 4 folds across the 4 GPUs/slots and waits).
# TRUSTED has no models of its own — every model here is chaos-trained. See 05_01_predict_common.sh.
#   bash 05_07_predict_chaos_all.sh
set -euo pipefail
HERE="$(dirname "$0")"
echo "=================================================================="
echo " TRUSTED ← CHAOS t1in models | 6 methods × 4 folds × {ct,us}"
echo "=================================================================="
for wrapper in \
    05_02_predict_chaos_baseline.sh \
    05_03_predict_chaos_v26_6_2.sh \
    05_04_predict_chaos_synthseg_EM.sh \
    05_05_predict_chaos_auglab_default.sh \
    05_06_predict_chaos_synthseg_noEM.sh \
    05_08_predict_chaos_auglabAug_v26_6_2.sh \
    ; do
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${wrapper} (all folds; ct + us as SEPARATE jobs)"
    # ct and us submitted as independent per-modality passes (separate Slurm jobs,
    # 45 min each) so neither waits on the other and the big US volumes never share
    # a job with CT. Both passes run concurrently; wait before the next method.
    bash "${HERE}/${wrapper}" "" all ct &
    bash "${HERE}/${wrapper}" "" all us &
    wait
done
echo ""
echo "[$(date '+%H:%M:%S')] ALL methods done."
