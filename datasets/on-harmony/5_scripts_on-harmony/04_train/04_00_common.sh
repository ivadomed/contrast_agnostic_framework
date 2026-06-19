#!/usr/bin/env bash
# Shared training template — sourced by 04_0{1..4}_train_*.sh, NOT invoked directly.
#
# Expected env vars (set by the calling script):
#   METHOD       baseline | v26_6 | synthseg_a | synthseg_b
#   TRAINER      nnUNetTrainerOnHarmony{Baseline|V26_6|SynthSegA|SynthSegB}
#   DA_WORKERS   number of batchgenerators DA workers (16 for baseline, 0 for synth)
#   LOG_DIR      e.g. /tmp/nnunet_baseline
#   RUN_ID       optional; passed as $1 for resume
#
# Optional:
#   GPUS_PER_FOLD  1 (default) → 4 folds in parallel, 1 GPU each.
#                  2           → DDP: each fold uses 2 GPUs (global batch 2 split
#                                1+1, SyncBN-free since the net uses InstanceNorm,
#                                so it is mathematically identical to 1-GPU batch 2).
#                                Folds run 2-at-a-time across the 4 GPUs (2 rounds).
#                                Per-fold epoch ≈ 1.9× faster; total CV wall-clock
#                                is ~unchanged (the 4 GPUs were already saturated).
#
# Each fold is launched through run_job() (scripts/job_runner/run_job.sh, sourced
# transitively via 00_utils/env.sh) instead of calling a resource manager directly
# — this script runs unchanged on whichever machine it's on. Unlike the
# brats2024-glioma/chaos common.sh, this one always blocks until all folds finish
# (no fire-and-exit mode), so every launch_fold call below passes --wait to
# run_job: on the set_slot backend that's a no-op (set_slot already blocks for
# the job's duration); on the slurm backend it's required (`sbatch --wait`), since
# a plain `sbatch` would return in ~1s and `wait "${PIDS[@]}"` would falsely
# report completion immediately.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RESULTS_BASE="${nnUNet_results}/runs"
TRAINER_DIR="Dataset031_OnHarmonyT1w31/${TRAINER}__nnUNetPlans__3d_fullres"
GPUS_PER_FOLD="${GPUS_PER_FOLD:-1}"

RUN_ID="${1:-${METHOD}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "[$(date '+%H:%M:%S')] ${METHOD} — RUN_ID=${RUN_ID}  (GPUS_PER_FOLD=${GPUS_PER_FOLD})"

# Launch one fold.  $1=fold  $2=slot spec (set_slot backend only, e.g. "0" or "0-1")  $3=CUDA_VISIBLE_DEVICES
launch_fold() {
    local FOLD="$1" SLOT="$2" GPUS="$3"
    local NGPU; NGPU="$(awk -F',' '{print NF}' <<<"$GPUS")"

    local CKPT="${RESULTS_BASE}/${RUN_ID}/${TRAINER_DIR}/fold_${FOLD}/checkpoint_latest.pth"
    local CONTINUE_FLAG=""
    if [ -f "$CKPT" ]; then
        CONTINUE_FLAG="--c"
        echo "  Fold ${FOLD}: resuming (slot ${SLOT}, GPUs ${GPUS})"
    else
        echo "  Fold ${FOLD}: fresh start (slot ${SLOT}, GPUs ${GPUS})"
    fi

    run_job --name "fold${FOLD}_${RUN_ID}" --gpus "${NGPU}" --slot "${SLOT}" \
        --log "${LOG_DIR}/fold${FOLD}.log" --wait -- \
        bash -c "
        export nnUNet_raw='${nnUNet_raw}'
        export nnUNet_preprocessed='${nnUNet_preprocessed}'
        export nnUNet_results='${RESULTS_BASE}/${RUN_ID}'
        export SPLITS_DIR='${SPLITS_DIR}'
        export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
        export PYTHONPATH='${PROJECT_ROOT}/datasets/on-harmony/5_scripts_on-harmony:\${PYTHONPATH:-}'
        export RUN_ID='${RUN_ID}'
        export nnUNet_n_proc_DA=${DA_WORKERS}
        export CUDA_VISIBLE_DEVICES='${GPUS}'
        export nnUNet_wandb_enabled=1
        export nnUNet_wandb_project='mri_synthesis_seg'
        export nnUNet_wandb_run_name='${RUN_ID}_fold${FOLD}'
        export TF_USE_LEGACY_KERAS=1
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
        export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
        export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
        export OMP_WAIT_POLICY=passive
        cd '${PROJECT_ROOT}'
        .venv/bin/nnUNetv2_train 031 3d_fullres ${FOLD} ${CONTINUE_FLAG} \
            -tr ${TRAINER} -p nnUNetPlans -num_gpus ${NGPU}
    "
}

if [ "${GPUS_PER_FOLD}" = "1" ]; then
    # 4 folds in parallel, one GPU each (original behaviour).
    declare -A PIDS
    for FOLD in 0 1 2 3; do
        launch_fold "${FOLD}" "${FOLD}" "${FOLD}" &
        PIDS[$FOLD]=$!
    done
    wait "${PIDS[@]}"
elif [ "${GPUS_PER_FOLD}" = "2" ]; then
    # DDP, 2 GPUs/fold, 2 folds at a time → 2 rounds.
    #   round 0: fold0 on GPUs 0,1 (slot 0-1) | fold1 on GPUs 2,3 (slot 2-3)
    #   round 1: fold2 on GPUs 0,1            | fold3 on GPUs 2,3
    for ROUND in 0 1; do
        FA=$((ROUND * 2)); FB=$((ROUND * 2 + 1))
        launch_fold "${FA}" "0-1" "0,1" &  PA=$!
        launch_fold "${FB}" "2-3" "2,3" &  PB=$!
        wait $PA $PB
    done
else
    echo "ERROR: GPUS_PER_FOLD must be 1 or 2 (got '${GPUS_PER_FOLD}'); batch size 2 cannot split across >2 GPUs." >&2
    exit 1
fi

echo "[$(date '+%H:%M:%S')] All ${METHOD} folds complete — ${RESULTS_BASE}/${RUN_ID}/"
