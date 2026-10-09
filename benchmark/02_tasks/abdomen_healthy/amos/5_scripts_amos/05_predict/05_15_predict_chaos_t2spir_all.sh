#!/usr/bin/env bash
# Run ALL chaos T2spir-trained models over AMOS CT+MRI, all 4 folds each.
# Methods run in PARALLEL: every method×fold is its own independent run_job (Slurm
# sbatch), so all 6×4 = 24 fold-jobs are submitted up front and the scheduler runs
# as many concurrently as GPUs allow — no reason to gate one method on another.
# AMOS has no models of its own — every model here is chaos-trained (cross-dataset).
# See 05_01_predict_chaos_common.sh for the cross-dataset inference rationale.
#
#   bash 05_15_predict_chaos_t2spir_all.sh
set -uo pipefail   # NOT -e: one method must not abort the others
HERE="$(dirname "$0")"

echo "=================================================================="
echo " AMOS <- CHAOS T2spir models | all 6 methods x 4 folds x CT+MRI (PARALLEL)"
echo "=================================================================="

pids=()
for wrapper in \
    05_09_predict_chaos_t2spir_baseline.sh \
    05_10_predict_chaos_t2spir_v26_6_2.sh \
    05_11_predict_chaos_t2spir_synthseg_EM.sh \
    05_12_predict_chaos_t2spir_synthseg_noEM.sh \
    05_13_predict_chaos_t2spir_auglab_default.sh \
    05_14_predict_chaos_t2spir_auglabAug_v26_6_2.sh ; do
    echo ">>> [$(date '+%H:%M:%S')] launching ${wrapper}"
    bash "${HERE}/${wrapper}" "${1:-}" "${2:-all}" ${3:+$3} &
    pids+=($!)
    sleep 5   # gentle spacing of sbatch bursts
done

echo "[$(date '+%H:%M:%S')] all T2spir methods submitted — waiting for completion..."
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[$(date '+%H:%M:%S')] ALL T2spir methods done (rc=${rc})."
exit "$rc"
