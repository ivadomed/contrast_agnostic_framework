#!/usr/bin/env bash
# Shared nnU-Net training driver for all datasets. NOT invoked directly and NOT
# sourced directly by method wrappers — it is sourced by each dataset's
# 04_train/04_00_common.sh shim, which first sources that dataset's 00_utils/env.sh
# and sets the few dataset-specific defaults below, then sources this file.
#
# Contract — the 04_00_common.sh shim must have already:
#   * sourced 00_utils/env.sh   (→ nnUNet_raw, nnUNet_preprocessed, SPLITS_DIR,
#                                  PROJECT_ROOT, SCRIPTS_DIR, DATASET_NAME,
#                                  TRAINING_CONTRAST, WANDB_PROJECT, run_job)
#   * cd "${PROJECT_ROOT}"
#   * set:   DATASET_ID_DEFAULT          e.g. 050 (brats) / 060 (chaos)
#            NNUNET_NUM_EPOCHS_DEFAULT   e.g. 500 (brats) / 250 (chaos)
#   * optionally set: NNUNET_ITERS_PER_EPOCH_DEFAULT  (chaos: 150; unset → the export
#            is omitted entirely, so the trainer keeps nnU-Net's built-in default)
#
# Method wrapper sets (as before): METHOD, TRAINER, DA_WORKERS, LOG_DIR, and may set
#   NNUNET_RESULTS_BASE, GPUS_PER_FOLD, NNUNET_NUM_EPOCHS, nnUNet_compile, RUN_ID ($1),
#   SINGLE_FOLD/SINGLE_SLOT/SINGLE_GPU, FOLD_SLOT_GPU, RESUME_WANDB_IDS, LAUNCH_WAIT.
#
# Each fold is launched through run_job() (scripts/job_runner/run_job.sh, sourced
# transitively via 00_utils/env.sh) instead of calling a resource manager directly —
# this runs unchanged on whichever machine it's on. On the original workstation
# (backend=set_slot) a fold is a detached systemd slice; on Vulcan (backend=slurm) a
# fold is its own independent sbatch job. By default launch_fold does NOT block
# (fire-and-exit): the fold keeps training after this returns. Set LAUNCH_WAIT=1 to
# block until all folds finish (only for non-Bash-tool / scripted use).
#
# ── RESUME NOTES (learned the hard way — don't repeat) ───────────────────────
#  1. LAUNCH so the script FIRES the folds and then EXITS — do NOT hold it open under
#     `timeout` (set_slot backend: `timeout N bash …` SIGTERMs the whole process group,
#     which can reach the set_slot/systemd-run call and kill the fold as it launches).
#     On slurm this isn't a concern — sbatch returns almost instantly and the job is
#     independent of this shell from then on.
#  2. GPU PINNING is automatic (CUDA_VISIBLE_DEVICES=<fold> below, used by the set_slot
#     backend since it doesn't isolate GPUs on that box). The slurm backend ignores this
#     and lets Slurm bind GPUs via --gres. Do NOT hand-roll launches — go through run_job.
#  3. EXTEND past the original length by raising NNUNET_NUM_EPOCHS. It is the PolyLR
#     horizon: resuming a finished run with the same value trains 0 epochs, and a stale
#     horizon makes the LR go negative→complex and crash. (fast.py sets num_epochs BEFORE
#     configure_optimizers so the scheduler gets the real value.)
#  4. WANDB resumes the same run automatically via each fold's wandb/ dir. If deleted,
#     pass RESUME_WANDB_IDS=<id per fold>; the patched logger (nnunet_logger.py:
#     nnUNet_wandb_run_id + allow_val_change) resumes by id from the cloud. On slurm,
#     runs log offline (WANDB_MODE=offline, no internet on compute nodes) — sync with
#     scripts/job_runner/wandb_sync.sh from the login node.
#  5. (set_slot backend only) After many launch/kill cycles, set_slot may briefly return
#     exit 144 (slices not released). Wait ~10s after pkill before relaunching.

# NNUNET_RESULTS_BASE overrides the env.sh default (method-specific output dirs).
RESULTS_BASE="${NNUNET_RESULTS_BASE:-${nnUNet_results}}"
DATASET_ID="${DATASET_ID:-${DATASET_ID_DEFAULT}}"
# 0* tolerates an unpadded id (e.g. 60 matching Dataset060_…) as well as a padded one.
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
TRAINER_DIR="${_DS_NAME}/${TRAINER}__nnUNetPlans__3d_fullres"
GPUS_PER_FOLD="${GPUS_PER_FOLD:-1}"

RUN_ID="${1:-${DATASET_NAME}_${TRAINING_CONTRAST}_${METHOD}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "[$(date '+%H:%M:%S')] ${METHOD} — RUN_ID=${RUN_ID}  (GPUS_PER_FOLD=${GPUS_PER_FOLD})"

launch_fold() {
    local FOLD="$1" SLOT="$2" GPUS="$3"
    local NGPU; NGPU="$(awk -F',' '{print NF}' <<<"$GPUS")"
    # On Slurm each fold is its own isolated job; the allocated GPU(s) are always
    # device 0..N-1 within the job's cgroup namespace — the physical CUDA index
    # (used by set_slot, where all 4 GPUs are visible) is meaningless here.
    local _CUDA_DEV="${GPUS}"
    if [ "${RUN_JOB_BACKEND:-}" = "slurm" ]; then
        _CUDA_DEV="$(seq -s, 0 $((NGPU-1)))"
    fi

    local FOLD_DIR="${RESULTS_BASE}/${RUN_ID}/${TRAINER_DIR}/fold_${FOLD}"
    local CKPT_LATEST="${FOLD_DIR}/checkpoint_latest.pth"
    local CKPT_FINAL="${FOLD_DIR}/checkpoint_final.pth"
    local CONTINUE_FLAG=""
    # Resume from checkpoint_latest (mid-run) or checkpoint_final (completed run that needs more epochs)
    if [ -f "$CKPT_LATEST" ]; then
        CONTINUE_FLAG="--c"
        echo "  Fold ${FOLD}: resuming from checkpoint_latest (slot ${SLOT}, GPUs ${GPUS})"
    elif [ -f "$CKPT_FINAL" ]; then
        cp "${CKPT_FINAL}" "${CKPT_LATEST}"
        CONTINUE_FLAG="--c"
        echo "  Fold ${FOLD}: resuming from checkpoint_final (slot ${SLOT}, GPUs ${GPUS})"
    else
        echo "  Fold ${FOLD}: fresh start (slot ${SLOT}, GPUs ${GPUS})"
    fi

    # Optional: resume a specific WandB run by id (one id per fold, space-separated in
    # RESUME_WANDB_IDS, e.g. "id0 id1 id2 id3"). Only needed when the local wandb/ dir
    # was lost — normally the logger auto-resumes via wandb/latest-run. Empty → no override.
    local WANDB_ID=""
    if [ -n "${RESUME_WANDB_IDS:-}" ]; then
        read -ra _RWIDS <<< "${RESUME_WANDB_IDS}"
        WANDB_ID="${_RWIDS[$FOLD]:-}"
    fi

    # Iterations-per-epoch is dataset-optional: only export it when a default is set, so
    # datasets that don't tune it keep nnU-Net's built-in default (no behaviour change).
    local _iters_export=""
    if [ -n "${NNUNET_ITERS_PER_EPOCH_DEFAULT:-}" ]; then
        _iters_export="export NNUNET_ITERS_PER_EPOCH='${NNUNET_ITERS_PER_EPOCH:-${NNUNET_ITERS_PER_EPOCH_DEFAULT}}'"
    fi

    local wait_args=()
    [ "${LAUNCH_WAIT:-0}" = "1" ] && wait_args=(--wait)

    run_job --name "fold${FOLD}_${RUN_ID}" --gpus "${NGPU}" --slot "${SLOT}" \
        --log "${LOG_DIR}/fold${FOLD}.log" "${wait_args[@]}" -- \
        bash -c "
        export nnUNet_raw='${nnUNet_raw}'
        export nnUNet_preprocessed='${nnUNet_preprocessed}'
        export nnUNet_results='${RESULTS_BASE}/${RUN_ID}'
        export SPLITS_DIR='${SPLITS_DIR}'
        export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
        export PYTHONPATH='${SCRIPTS_DIR}:\${PYTHONPATH:-}'
        export RUN_ID='${RUN_ID}'
        export nnUNet_n_proc_DA=${DA_WORKERS}
        export AUGLAB_PARAMS_GPU_JSON='${AUGLAB_PARAMS_GPU_JSON:-}'
        export AUGLAB_VAL_PARAMS_GPU_JSON='${AUGLAB_VAL_PARAMS_GPU_JSON:-}'
        export CUDA_VISIBLE_DEVICES='${_CUDA_DEV}'
        export nnUNet_wandb_enabled='${nnUNet_wandb_enabled:-1}'
        export nnUNet_wandb_project='${nnUNet_wandb_project:-${WANDB_PROJECT:-mri_synthesis_seg_${DATASET_NAME}}}'
        export nnUNet_wandb_run_name='${RUN_ID}_fold${FOLD}'
        export nnUNet_wandb_run_id='${WANDB_ID}'
        export TF_USE_LEGACY_KERAS=1
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
        export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
        export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
        export OMP_WAIT_POLICY=passive
        export NNUNET_NUM_EPOCHS='${NNUNET_NUM_EPOCHS:-${NNUNET_NUM_EPOCHS_DEFAULT}}'
        ${_iters_export}
        export nnUNet_compile='${nnUNet_compile:-0}'
        cd '${PROJECT_ROOT}'
        .venv/bin/nnUNetv2_train ${DATASET_ID} 3d_fullres ${FOLD} ${CONTINUE_FLAG} \
            -tr ${TRAINER} -p nnUNetPlans -num_gpus ${NGPU}
    "
}

# Stagger between fold launches: on the set_slot backend, simultaneous
# set_slot→sudo ml_job→systemd-run calls race during slice setup and die silently right
# after the nnU-Net banner; a few seconds between launches lets each slice come up
# cleanly. Harmless on the slurm backend, where each fold is an independent submission.
LAUNCH_STAGGER_S="${LAUNCH_STAGGER_S:-15}"

if [ -n "${FOLD_SLOT_GPU:-}" ]; then
    # Explicit per-fold placement: space-separated "FOLD,SLOT,GPU" tuples, where GPU is
    # the physical CUDA_VISIBLE_DEVICES index (set_slot does NOT isolate GPUs on this box
    # — all are visible by physical index). Enables packing >1 fold per GPU, e.g. 4 folds
    # on 2 GPUs: FOLD_SLOT_GPU="0,0,0 1,1,0 2,2,1 3,3,1". (slurm: SLOT ignored, only the
    # fold/GPU grouping matters.)
    echo "[$(date '+%H:%M:%S')] custom placement (FOLD,SLOT,GPU): ${FOLD_SLOT_GPU}"
    declare -A PIDS; _i=0
    for _tuple in ${FOLD_SLOT_GPU}; do
        IFS=',' read -r _F _S _G <<< "${_tuple}"
        launch_fold "${_F}" "${_S}" "${_G}" &
        PIDS[${_i}]=$!; _i=$((_i + 1))
        sleep "${LAUNCH_STAGGER_S}"
    done
    if [ "${LAUNCH_WAIT:-0}" = "1" ]; then
        wait "${PIDS[@]}"
        echo "[$(date '+%H:%M:%S')] All ${METHOD} folds complete — ${RESULTS_BASE}/${RUN_ID}/"
    else
        echo "[$(date '+%H:%M:%S')] ${_i} folds launched (detached) — ${RESULTS_BASE}/${RUN_ID}/"
        echo "  monitor: tail -f ${LOG_DIR}/fold0.log   |   logs: ${LOG_DIR}/fold*.log"
    fi
elif [ "${GPUS_PER_FOLD}" = "1" ]; then
    declare -A PIDS
    # SINGLE_FOLD override: resume ONE fold on a chosen slot. SINGLE_GPU is the SLOT-LOCAL
    # CUDA index: under per-slot GPU isolation the slot exposes its physical GPU as index
    # 0, so SINGLE_GPU defaults to 0 (NOT the slot number). (set_slot backend only — slurm
    # ignores SLOT/SINGLE_GPU and lets Slurm pick the physical GPU.)
    if [ -n "${SINGLE_FOLD:-}" ]; then
        launch_fold "${SINGLE_FOLD}" "${SINGLE_SLOT:-${SINGLE_FOLD}}" "${SINGLE_GPU:-0}" &
        PIDS[0]=$!
    else
        for FOLD in 0 1 2 3; do
            launch_fold "${FOLD}" "${FOLD}" "${FOLD}" &
            PIDS[$FOLD]=$!
            if [ "${FOLD}" -lt 3 ]; then sleep "${LAUNCH_STAGGER_S}"; fi
        done
    fi
    if [ "${LAUNCH_WAIT:-0}" = "1" ]; then
        wait "${PIDS[@]}"
        echo "[$(date '+%H:%M:%S')] All ${METHOD} folds complete — ${RESULTS_BASE}/${RUN_ID}/"
    else
        echo "[$(date '+%H:%M:%S')] folds launched (detached) — ${RESULTS_BASE}/${RUN_ID}/"
        echo "  monitor: tail -f ${LOG_DIR}/fold0.log"
    fi
elif [ "${GPUS_PER_FOLD}" = "2" ]; then
    # DDP: only 2 folds fit at a time, so rounds are inherently sequential → must wait for
    # round N's folds to finish before round N+1 reuses the same GPUs. LAUNCH_WAIT=1 here
    # (regardless of the caller's global setting) forces run_job to block: on set_slot a
    # no-op (it already blocks for the job's duration); on slurm required (`sbatch --wait`),
    # since a plain sbatch would return in ~1s and round 2 would race round 1's GPUs.
    for ROUND in 0 1; do
        FA=$((ROUND * 2)); FB=$((ROUND * 2 + 1))
        LAUNCH_WAIT=1 launch_fold "${FA}" "0-1" "0,1" &  PA=$!
        sleep "${LAUNCH_STAGGER_S}"
        LAUNCH_WAIT=1 launch_fold "${FB}" "2-3" "2,3" &  PB=$!
        wait $PA $PB
    done
    echo "[$(date '+%H:%M:%S')] All ${METHOD} folds complete — ${RESULTS_BASE}/${RUN_ID}/"
else
    echo "ERROR: GPUS_PER_FOLD must be 1 or 2 (got '${GPUS_PER_FOLD}')" >&2
    exit 1
fi
