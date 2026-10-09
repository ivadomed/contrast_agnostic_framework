#!/usr/bin/env bash
# GPU inference for the co-polarity intervention (C-FLIP alone, combined with S-ADD, + sham).
# Usage: bash run_copolarity_partB_predict.sh <model_key>  (t2w_noise|t2w_real only)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh

MODEL_KEY="${1:?model_key required: t2w_noise|t2w_real}"

SCRATCH_ROOT=/scratch/${USER}/brats_intervention
INPUTS_ROOT="${SCRATCH_ROOT}/inputs"
PREDS_ROOT="${SCRATCH_ROOT}/preds"
LOGS_DIR="${SCRATCH_ROOT}/logs"
mkdir -p "${LOGS_DIR}"

SETS="corefeather_t1n corefeather_sadd_t1n sham_corefeather_t1n sham_corefeather_sadd_t1n"

case "${MODEL_KEY}" in
  t2w_noise)
    NNUNET_RESULTS="${PWD}/benchmark/02_tasks/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/auglab/brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659"
    DATASET_ID=52
    TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
    ;;
  t2w_real)
    NNUNET_RESULTS="${PWD}/benchmark/02_tasks/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/nnUNet/brats2024-glioma_t2w_v26_6_2_train050_val100_20260620_125217"
    DATASET_ID=52
    TRAINER="nnUNetTrainerBraTS2024GliomaT2wV26_6_2_train050_val100"
    ;;
  *) echo "unknown model_key ${MODEL_KEY}" >&2; exit 1 ;;
esac

predict_cmds=""
for s in ${SETS}; do
    IN_DIR="${INPUTS_ROOT}/${s}"
    OUT_DIR="${PREDS_ROOT}/${MODEL_KEY}/${s}"
    mkdir -p "${OUT_DIR}"
    predict_cmds+="echo '${MODEL_KEY} ${s}...'; .venv/bin/nnUNetv2_predict -i '${IN_DIR}' -o '${OUT_DIR}' -d ${DATASET_ID} -c 3d_fullres -tr ${TRAINER} -f 0 --disable_tta -chk checkpoint_best.pth; echo '${MODEL_KEY} ${s} done'; "
done

run_job --name "brats_copolB_${MODEL_KEY}" --gpus 1 --cpus 8 --mem 64G --time "00:30:00" --wait \
  --log "${LOGS_DIR}/copolarity_partB_predict_${MODEL_KEY}.log" -- \
  bash -c "
    export nnUNet_raw='${nnUNet_raw}'
    export nnUNet_preprocessed='${nnUNet_preprocessed}'
    export nnUNet_results='${NNUNET_RESULTS}'
    export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
    export PYTHONPATH='${PYTHONPATH:-}'
    export CUDA_VISIBLE_DEVICES=0
    cd '${PROJECT_ROOT}'
    ${predict_cmds}
  "
