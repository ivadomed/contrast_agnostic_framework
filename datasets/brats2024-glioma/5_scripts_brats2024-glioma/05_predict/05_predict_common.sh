#!/usr/bin/env bash
# Shared predict template — sourced by 05_0X_predict_<method>.sh, NOT run directly.
#
# Runs SINGLE-FOLD inference for one experiment across one or more contrasts of the
# held-out BraTS test set, and saves the predictions for separate downstream analysis.
#
# Set by the calling wrapper:
#   METHOD      label, e.g. v26_6_2
#   TRAINER     nnUNet trainer class, e.g. nnUNetTrainerBraTS2024GliomaV26_6_2
# Provided as args by the wrapper (passed through from the user):
#   RUN_ID      $1  required — the training run dir under $nnUNet_results
#   FOLD        $2  optional — single fold to predict with (default 0)
#   CONTRASTS   $3… optional — space-separated subset of: t1n t1c t2w t2f (default all)
#
# Optional env overrides:
#   CHECKPOINT  checkpoint_best.pth (default) | checkpoint_final.pth
#   SLOT GPU    set_slot slot / CUDA device (default 0)
#
# Test-input dirs (imagesTs_<contrast>/) must exist — build once with:
#   python 05_00_build_test_inputs.py

set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"

RUN_ID="${1:?RUN_ID required (training run dir name under nnUNet_results)}"
FOLD="${2:-0}"
shift $(( $# >= 2 ? 2 : $# ))
CONTRASTS=("$@"); [ ${#CONTRASTS[@]} -eq 0 ] && CONTRASTS=(t1n t1c t2w t2f)

DATASET_ID="${DATASET_ID:-051}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"
SLOT="${SLOT:-0}"; GPU="${GPU:-0}"
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset${DATASET_ID}_" | head -1)"
RUN_DIR="${nnUNet_results}/${RUN_ID}"

[ -d "$RUN_DIR" ] || { echo "ERROR: run dir not found: $RUN_DIR" >&2; exit 1; }

echo "[$(date '+%H:%M:%S')] predict ${METHOD} | run=${RUN_ID} | fold=${FOLD} | ckpt=${CHECKPOINT}"
echo "  contrasts: ${CONTRASTS[*]}"

for contrast in "${CONTRASTS[@]}"; do
    INPUT_DIR="${nnUNet_raw}/${_DS_NAME}/imagesTs_${contrast}"
    OUTPUT_DIR="${RUN_DIR}/predictions/fold${FOLD}/${contrast}"
    if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]; then
        echo "  ! skip ${contrast}: input dir missing/empty ($INPUT_DIR) — run 05_00_build_test_inputs.py" >&2
        continue
    fi
    mkdir -p "$OUTPUT_DIR"
    echo "  → ${contrast}: $(ls "$INPUT_DIR" | wc -l) cases → $OUTPUT_DIR"
    set_slot ${SLOT} bash -c "
        export nnUNet_raw='${nnUNet_raw}'
        export nnUNet_preprocessed='${nnUNet_preprocessed}'
        export nnUNet_results='${RUN_DIR}'
        export NNUNET_PROJECT_ROOT='$(pwd)'
        export PYTHONPATH='$(pwd)/datasets/brats2024-glioma/5_scripts_brats2024-glioma:\${PYTHONPATH:-}'
        export CUDA_VISIBLE_DEVICES='${GPU}'
        export TF_USE_LEGACY_KERAS=1
        cd '$(pwd)'
        .venv/bin/nnUNetv2_predict \
            -i '${INPUT_DIR}' \
            -o '${OUTPUT_DIR}' \
            -d ${DATASET_ID} \
            -c 3d_fullres \
            -tr ${TRAINER} \
            -f ${FOLD} \
            --disable_tta \
            -chk ${CHECKPOINT}
    " 2>&1 | tee "/tmp/predict_${METHOD}_${RUN_ID}_fold${FOLD}_${contrast}.log"
    echo "  ✓ ${contrast} done"
done

echo "[$(date '+%H:%M:%S')] all contrasts done → ${RUN_DIR}/predictions/fold${FOLD}/"
