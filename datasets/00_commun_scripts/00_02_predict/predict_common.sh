#!/usr/bin/env bash
# Shared nnU-Net prediction driver for all datasets. NOT invoked directly and NOT
# sourced directly by method wrappers — it is sourced by each dataset's 05_predict
# common shim (05_predict_common.sh / 05_01_predict_common.sh / 05_01_predict_chaos_common.sh),
# which first sources that dataset's 00_utils/env.sh and sets the config vars below.
#
# Two modes:
#   PREDICT_MODE=own    — predict with this dataset's OWN trained models
#                         (brats2024-glioma, chaos). Model + inputs live in this dataset.
#   PREDICT_MODE=cross  — predict with ANOTHER dataset's models (amos, sliver07 use
#                         chaos checkpoints). Model dir + nnUNet_raw/preprocessed come
#                         from the CHAOS_* env vars (set by env.sh); inputs are this
#                         dataset's flat imagesTs_<item>/; outputs are segregated under
#                         this dataset's PREDICTIONS_ROOT/<chaos_model_type>/<contrast>/.
#
# Contract — the shim must have already sourced env.sh, cd'd to PROJECT_ROOT, and set:
#   PREDICT_MODE          own | cross
#   PREDICT_JOB_PREFIX    Slurm job-name prefix, e.g. brats_predict / amos_predict
#   PREDICT_LOG_PREFIX    /tmp log-file prefix, e.g. predict / amos_predict
#   PREDICT_ITEMS_DEFAULT default modality/contrast list, e.g. "t1n t1c t2w t2f"
#   PREDICT_FOLD_DEFAULT  0 (own) | all (cross)
#   PREDICT_DATASET_ID_DEFAULT   (own only; cross derives the id from CHAOS_DATASET_ID)
#   PREDICT_TIME          run_job --time value ("" → omit the flag, e.g. chaos)
#   PREDICT_MEM           run_job --mem value ("" → omit, use run_job per-GPU default;
#                         set for huge-volume datasets, e.g. TRUSTED US needs ~200G)
#   PREDICT_EXTRA_FLAGS   extra nnUNetv2_predict flags ("" | "-npp 12 -nps 6")
#   PREDICT_INPUT_SUFFIX  (optional, default "") suffix on the input dir to select a
#                         perturbed test variant, e.g. "_translation_050" →
#                         imagesTs_<item>_translation_050. For robustness experiments.
#   PREDICT_OUTPUT_SUBDIR (optional, default "") namespace inserted after fold{F}/ in
#                         the output path, e.g. "exp_translation_050" →
#                         .../fold{F}/exp_translation_050/<item>/. Pairs with the suffix.
#
# From the method wrapper (env): METHOD, TRAINER, CATEGORY (default nnUNet).
# From the user (positional args, forwarded as "$@"):
#   $1 RUN_ID   (own: required; cross: optional, wrapper supplies a default RUN_ID)
#   $2 FOLD     0-3 or "all" (default PREDICT_FOLD_DEFAULT)
#   $3… items   subset of the modality/contrast list (default PREDICT_ITEMS_DEFAULT)
# Optional env: CHECKPOINT (default checkpoint_best.pth), SLOT/GPU (single-fold only).
#
# Each prediction is launched through run_job() with --gpus 1 --wait: items within a
# fold run sequentially on the same GPU, folds run in parallel (fold N → slot/GPU N).

if [ "${PREDICT_MODE}" = "cross" ]; then
    RUN_ID="${1:-${RUN_ID:?RUN_ID required (chaos training run dir name)}}"
else
    RUN_ID="${1:?RUN_ID required (training run dir name under nnUNet_results)}"
fi
FOLD="${2:-${PREDICT_FOLD_DEFAULT}}"
shift $(( $# >= 2 ? 2 : $# )) || true
ITEMS=("$@"); [ ${#ITEMS[@]} -eq 0 ] && read -ra ITEMS <<< "${PREDICT_ITEMS_DEFAULT}"

CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"
CATEGORY="${CATEGORY:-nnUNet}"

if [ "${PREDICT_MODE}" = "cross" ]; then
    DATASET_ID="${CHAOS_DATASET_ID:-60}"
    RUN_DIR="${CHAOS_PREDICTIONS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
    _OUT_BASE="${PREDICTIONS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}"
    _IN_BASE="${nnUNet_raw}/imagesTs_"                     # flat raw layout (no Dataset<id>/ dir)
    _JOB_RAW="${CHAOS_NNUNET_RAW}"
    _JOB_PREP="${CHAOS_DATASET_ROOT}/2_nnUNet_chaos/preprocessed"
    _JOB_RESULTS="${RUN_DIR}"
else
    DATASET_ID="${DATASET_ID:-${PREDICT_DATASET_ID_DEFAULT}}"
    # Models live under PREDICTIONS_ROOT/{model_type}/{training_contrast}/{category}/ —
    # derive here (after env.sh, which resets nnUNet_results to the nnUNet root).
    export nnUNet_results="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}"
    RUN_DIR="${nnUNet_results}/${RUN_ID}"
    _OUT_BASE="${nnUNet_results}"
    _DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
    _IN_BASE="${nnUNet_raw}/${_DS_NAME}/imagesTs_"
    _JOB_RAW="${nnUNet_raw}"
    _JOB_PREP="${nnUNet_preprocessed}"
    _JOB_RESULTS="${RUN_DIR}"
fi

[ -d "$RUN_DIR" ] || { echo "ERROR: run dir not found: $RUN_DIR" >&2; exit 1; }

predict_fold() {
    local F="$1" SLOT="$2" GPU="$3"
    # CUDA pinning. own: always export the device (physical index on set_slot, 0 on
    # Slurm). cross: only pin off-Slurm (on Slurm let the scheduler bind via --gres) —
    # this preserves the original per-family behaviour exactly.
    local _CUDA_DEV="${GPU}"
    [ "${PREDICT_MODE}" != "cross" ] && [ "${RUN_JOB_BACKEND:-}" = "slurm" ] && _CUDA_DEV="0"
    local _cuda_line
    if [ "${PREDICT_MODE}" = "cross" ]; then
        _cuda_line="[ -z \"\${SLURM_JOB_ID:-}\" ] && export CUDA_VISIBLE_DEVICES='${GPU}'"
    else
        _cuda_line="export CUDA_VISIBLE_DEVICES='${_CUDA_DEV}'"
    fi

    echo "[$(date '+%H:%M:%S')] predict ${METHOD} | run=${RUN_ID} | fold=${F} | ckpt=${CHECKPOINT} | slot=${SLOT} gpu=${GPU}"

    # Pre-check items on the login node, create output dirs, and build all predict
    # commands into one script — submitted as a single Slurm job so all contrasts share
    # one GPU allocation with no re-queuing overhead between them.
    local predict_cmds="" any_valid=0
    for item in "${ITEMS[@]}"; do
        # PREDICT_INPUT_SUFFIX / PREDICT_OUTPUT_SUBDIR (both optional, default ""):
        # hooks for input-perturbation experiments. The suffix selects a variant input
        # dir (e.g. imagesTs_<item>_translation_050); the subdir namespaces the output
        # (e.g. .../fold{F}/exp_translation_050/<item>/). Empty → normal behaviour.
        local INPUT_DIR="${_IN_BASE}${item}${PREDICT_INPUT_SUFFIX:-}"
        local OUTPUT_DIR="${_OUT_BASE}/${RUN_ID}/fold${F}/${PREDICT_OUTPUT_SUBDIR:+${PREDICT_OUTPUT_SUBDIR}/}${item}"
        if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]; then
            echo "  ! fold${F} skip ${item}: input dir missing/empty ($INPUT_DIR) — run 05_00_build_test_inputs.py" >&2
            continue
        fi
        mkdir -p "$OUTPUT_DIR"
        any_valid=1
        echo "  → fold${F} ${item}: $(ls "$INPUT_DIR" | wc -l) cases → $OUTPUT_DIR"
        predict_cmds+="echo 'fold${F} ${item}...'; .venv/bin/nnUNetv2_predict -i '${INPUT_DIR}' -o '${OUTPUT_DIR}' -d ${DATASET_ID} -c 3d_fullres -tr ${TRAINER} -f ${F} --disable_tta -chk ${CHECKPOINT}${PREDICT_EXTRA_FLAGS:+ ${PREDICT_EXTRA_FLAGS}}; echo 'fold${F} ${item} done'; "
    done

    if [ "$any_valid" = "0" ]; then
        echo "  ! fold${F}: no contrast predictions found — skipping" >&2
        return
    fi

    local _time_args=(); [ -n "${PREDICT_TIME:-}" ] && _time_args=(--time "${PREDICT_TIME}")
    # PREDICT_MEM (optional, like PREDICT_TIME): override the run_job per-GPU memory
    # default — needed for datasets with very large volumes (e.g. TRUSTED 3D US,
    # ~620 M voxels, OOMs at the 110 G default). Empty → omit, use the run_job default.
    local _mem_args=(); [ -n "${PREDICT_MEM:-}" ] && _mem_args=(--mem "${PREDICT_MEM}")
    run_job --name "${PREDICT_JOB_PREFIX}_${METHOD}_fold${F}" \
        --gpus 1 --slot "${SLOT}" "${_time_args[@]}" "${_mem_args[@]}" \
        --log "/tmp/${PREDICT_LOG_PREFIX}_${METHOD}_${RUN_ID}_fold${F}.log" --wait -- \
        bash -c "
        export nnUNet_raw='${_JOB_RAW}'
        export nnUNet_preprocessed='${_JOB_PREP}'
        export nnUNet_results='${_JOB_RESULTS}'
        export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
        export PYTHONPATH='${PYTHONPATH}'
        ${_cuda_line}
        export TF_USE_LEGACY_KERAS=1
        cd '${PROJECT_ROOT}'
        ${predict_cmds}
        "
    echo "[$(date '+%H:%M:%S')] fold${F} done → ${_OUT_BASE}/${RUN_ID}/fold${F}/"
}

if [ "$FOLD" = "all" ]; then
    echo "[$(date '+%H:%M:%S')] predict ${METHOD} | run=${RUN_ID} | ALL FOLDS (parallel, fold→slot) | ckpt=${CHECKPOINT}"
    echo "  items: ${ITEMS[*]}"
    for F in 0 1 2 3; do predict_fold "$F" "$F" "$F" & done
    wait
    echo "[$(date '+%H:%M:%S')] all folds done → ${_OUT_BASE}/${RUN_ID}/"
else
    echo "  items: ${ITEMS[*]}"
    predict_fold "${FOLD}" "${SLOT:-0}" "${GPU:-0}"
fi
