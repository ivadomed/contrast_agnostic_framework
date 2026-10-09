#!/usr/bin/env bash
# KILLARNEY SIZING PROBE for pansegdata (run ON Killarney, BEFORE the real launch; a probe's numbers do not transfer between datasets/clusters).
# Killarney allocates PER GPU (one fold = one single-GPU sbatch job through run_job), so unlike TamIA there is no whole-node packing: the probe is fold 0 of the
# 6 headline methods x both contrasts = 12 independent single-GPU jobs, PROBE_EPOCHS epochs each, written to a THROWAWAY results base
# ($SCRATCH/pansegdata/_probe_killarney) so it cannot pollute real checkpoints. Read the numbers with 04_27_probe_report.sh.
#   bash 04_26_killarney_sizing_probe.sh                      # L40S, PROBE_EPOCHS=20
#   PROBE_GPU_TYPE=h100 bash 04_26_killarney_sizing_probe.sh   # same probe on H100 (run both to choose the GPU class)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRATCH="${SCRATCH:-/scratch/${USER}}"
GPU="${PROBE_GPU_TYPE:-l40s}"
# cluster-specific run_job overrides (set BEFORE env.sh: common_env freezes the run_job defaults). Account/gres come from run_job_slurm.sh.
export RUN_JOB_ACCOUNT="${RUN_JOB_ACCOUNT:-aip-jcohen}" RUN_JOB_GPU_TYPE="${GPU}"
export RUN_JOB_TIME_DEFAULT="${PROBE_TIME:-01:30:00}"
source "${HERE}/../00_utils/env.sh"
PROBE="${SCRATCH}/pansegdata/_probe_killarney_${GPU}${PROBE_TAG:-}"
export RESULTS_DIR="${PROBE}/results"      # wrapper LOG_DIR = ${RESULTS_DIR}/_logs/...: keep every probe's job logs separate (a shared path lost the first failure traces)
export PREDICTIONS_ROOT="${PROBE}/01_predictions"          # throwaway base; env.sh derives nnUNet_results from it
mkdir -p "${PREDICTIONS_ROOT}"
export TRAIN_FOLDS="0" NNUNET_NUM_EPOCHS="${PROBE_EPOCHS:-20}"
echo "[probe] gpu=${GPU} epochs=${NNUNET_NUM_EPOCHS} cpus/gpu=${RUN_JOB_CPUS_PER_GPU} mem/gpu=${RUN_JOB_MEM_PER_GPU} results base=${PREDICTIONS_ROOT}"
n=0
for C in t1wce t2w; do
    if [ "${C}" = t1wce ]; then WRAPPERS=( "${HERE}"/04_0[1-6]_train_t1wce_*.sh ); else WRAPPERS=( "${HERE}"/04_{08,09,10,11,12,13}_train_t2w_*.sh ); fi
    [ "${#WRAPPERS[@]}" = 6 ] || { echo "ERROR: expected 6 ${C} headline wrappers, got ${#WRAPPERS[@]}" >&2; exit 1; }
    for w in "${WRAPPERS[@]}"; do
        M="$(grep -m1 -E '^METHOD="' "$w" | sed -E 's/^METHOD="([^"]*)".*/\1/')"
        echo "[probe] submit ${C}/${M}"
        ( bash "$w" "pansegdata_${C}_${M}_PROBE" ) < /dev/null
        n=$((n+1)); sleep 2    # space out rapid sbatch calls (cluster etiquette)
    done
done
echo "[probe] submitted ${n} fold-0 jobs (expect 12). Watch: squeue -u \$USER ; report: bash ${HERE}/04_27_probe_report.sh ${PROBE}"
