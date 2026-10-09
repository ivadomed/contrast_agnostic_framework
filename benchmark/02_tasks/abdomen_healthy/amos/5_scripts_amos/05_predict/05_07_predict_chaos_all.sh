#!/usr/bin/env bash
# Run ALL chaos-trained models over AMOS CT+MRI, all 4 folds each.
# Methods run in PARALLEL: every method×fold is its own independent run_job
# (Slurm sbatch), so all 6×4 = 24 fold-jobs are submitted up front and the
# scheduler runs as many concurrently as GPUs allow. On a shared cluster there
# is no reason to gate one method on another — each fold-job is independent.
# AMOS has no models of its own — every model here is chaos-trained (cross-dataset).
# See 05_01_predict_chaos_common.sh for the cross-dataset inference rationale.
#
#   bash 05_07_predict_chaos_all.sh
#   bash 05_07_predict_chaos_all.sh "" all ct    # CT only (skip MRI)
set -uo pipefail   # NOT -e: one method must not abort the others
HERE="$(dirname "$0")"

echo "=================================================================="
echo " AMOS ← CHAOS models | all 6 methods × 4 folds × CT+MRI (PARALLEL)"
echo "=================================================================="

pids=()
for wrapper in \
    05_02_predict_chaos_baseline.sh \
    05_03_predict_chaos_v26_6_2.sh \
    05_04_predict_chaos_synthseg_EM.sh \
    05_05_predict_chaos_auglab_default.sh \
    05_06_predict_chaos_synthseg_noEM.sh \
    05_08_predict_chaos_auglabAug_v26_6_2.sh ; do
    echo ">>> [$(date '+%H:%M:%S')] launching ${wrapper}"
    bash "${HERE}/${wrapper}" "${1:-}" "${2:-all}" ${3:+$3} &
    pids+=($!)
    sleep 5   # gentle spacing of sbatch bursts (avoid hammering the scheduler)
done

echo "[$(date '+%H:%M:%S')] all methods submitted — waiting for completion..."
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done

echo ""
echo "[$(date '+%H:%M:%S')] ALL methods done (rc=${rc})."
exit "$rc"
