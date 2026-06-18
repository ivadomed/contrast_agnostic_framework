#!/usr/bin/env bash
# Shared training template — sourced by 04_0X_train_<method>.sh, NOT invoked directly.
# Ported from brats2024-glioma/04_00_common.sh (only dataset paths/ids differ).
#
# Expected env vars (set by the calling script):
#   METHOD       baseline | ...
#   TRAINER      nnUNetTrainerCHAOSBaseline | ...
#   DA_WORKERS   batchgenerators DA workers (64 for baseline; 0 for on-GPU synth)
#   LOG_DIR      e.g. /tmp/nnunet_chaos_baseline
#   RUN_ID       optional; passed as $1 for resume
#
# Optional:
#   GPUS_PER_FOLD     1 (default) → 4 folds in parallel, 1 GPU each. 2 → DDP.
#   NNUNET_NUM_EPOCHS / NNUNET_ITERS_PER_EPOCH   training horizon (read by fast.py).
#   RESUME_WANDB_IDS  space-separated cloud run id per fold ("id0 id1 id2 id3").
#
# LAUNCH NOTE (learned the hard way — see project memory): fire-and-exit. Run
# foreground from the Bash tool; each fold is an independent set_slot job (own
# systemd slice) that survives after this script exits. Do NOT wrap in `timeout`
# or `nohup … &` — both reap/kill the folds. GPU pinning is automatic below.

set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project

PROJECT_ROOT="$(pwd)"
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
RESULTS_BASE="${NNUNET_RESULTS_BASE:-${nnUNet_results}}"
DATASET_ID="${DATASET_ID:-060}"
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
TRAINER_DIR="${_DS_NAME}/${TRAINER}__nnUNetPlans__3d_fullres"
GPUS_PER_FOLD="${GPUS_PER_FOLD:-1}"

RUN_ID="${1:-${DATASET_NAME}_${METHOD}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "[$(date '+%H:%M:%S')] ${METHOD} — RUN_ID=${RUN_ID}  (GPUS_PER_FOLD=${GPUS_PER_FOLD})"

launch_fold() {
    local FOLD="$1" SLOT="$2" GPUS="$3"
    local NGPU; NGPU="$(awk -F',' '{print NF}' <<<"$GPUS")"

    local FOLD_DIR="${RESULTS_BASE}/${RUN_ID}/${TRAINER_DIR}/fold_${FOLD}"
    local CKPT_LATEST="${FOLD_DIR}/checkpoint_latest.pth"
    local CKPT_FINAL="${FOLD_DIR}/checkpoint_final.pth"
    local CONTINUE_FLAG=""
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

    local WANDB_ID=""
    if [ -n "${RESUME_WANDB_IDS:-}" ]; then
        read -ra _RWIDS <<< "${RESUME_WANDB_IDS}"
        WANDB_ID="${_RWIDS[$FOLD]:-}"
    fi

    set_slot ${SLOT} bash -c "
        export nnUNet_raw='${nnUNet_raw}'
        export nnUNet_preprocessed='${nnUNet_preprocessed}'
        export nnUNet_results='${RESULTS_BASE}/${RUN_ID}'
        export SPLITS_DIR='${SPLITS_DIR}'
        export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
        export PYTHONPATH='${PROJECT_ROOT}/datasets/chaos/5_scripts_chaos:\${PYTHONPATH:-}'
        export RUN_ID='${RUN_ID}'
        export nnUNet_n_proc_DA=${DA_WORKERS}
        export AUGLAB_PARAMS_GPU_JSON='${AUGLAB_PARAMS_GPU_JSON:-}'
        export AUGLAB_VAL_PARAMS_GPU_JSON='${AUGLAB_VAL_PARAMS_GPU_JSON:-}'
        export CUDA_VISIBLE_DEVICES='${GPUS}'
        export nnUNet_wandb_enabled='${nnUNet_wandb_enabled:-1}'
        export nnUNet_wandb_project='${nnUNet_wandb_project:-${WANDB_PROJECT:-mri_synthesis_seg_${DATASET_NAME}}}'
        export nnUNet_wandb_run_name='${RUN_ID}_fold${FOLD}'
        export nnUNet_wandb_run_id='${WANDB_ID}'
        export TF_USE_LEGACY_KERAS=1
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
        export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
        export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
        export OMP_WAIT_POLICY=passive
        export NNUNET_NUM_EPOCHS='${NNUNET_NUM_EPOCHS:-250}'
        export NNUNET_ITERS_PER_EPOCH='${NNUNET_ITERS_PER_EPOCH:-150}'
        export nnUNet_compile='${nnUNet_compile:-0}'
        cd '${PROJECT_ROOT}'
        .venv/bin/nnUNetv2_train ${DATASET_ID} 3d_fullres ${FOLD} ${CONTINUE_FLAG} \
            -tr ${TRAINER} -p nnUNetPlans -num_gpus ${NGPU}
    " > "${LOG_DIR}/fold${FOLD}.log" 2>&1
}

LAUNCH_STAGGER_S="${LAUNCH_STAGGER_S:-15}"

if [ -n "${FOLD_SLOT_GPU:-}" ]; then
    # Explicit per-fold placement: space-separated "FOLD,SLOT,GPU" tuples, where GPU is
    # the physical CUDA_VISIBLE_DEVICES index (set_slot does NOT isolate GPUs — all are
    # visible by physical index). Enables packing >1 fold per GPU, e.g. 4 folds on 2
    # GPUs (2 per GPU): FOLD_SLOT_GPU="0,0,0 1,1,0 2,2,1 3,3,1".
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
    for ROUND in 0 1; do
        FA=$((ROUND * 2)); FB=$((ROUND * 2 + 1))
        launch_fold "${FA}" "0-1" "0,1" &  PA=$!
        sleep "${LAUNCH_STAGGER_S}"
        launch_fold "${FB}" "2-3" "2,3" &  PB=$!
        wait $PA $PB
    done
    echo "[$(date '+%H:%M:%S')] All ${METHOD} folds complete — ${RESULTS_BASE}/${RUN_ID}/"
else
    echo "ERROR: GPUS_PER_FOLD must be 1 or 2 (got '${GPUS_PER_FOLD}')" >&2
    exit 1
fi
