#!/usr/bin/env bash
# GPU inference for the texture-transplant test (X1-X4 + symmetry check + sham). Same
# nnUNetv2_predict invocation as the earlier intervention predict scripts.
# Usage: bash run_intervention_transplant_predict.sh <model_key>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh

MODEL_KEY="${1:?model_key required: t1n_noise|t1n_real|t2w_noise|t2w_real}"

SCRATCH_ROOT=/scratch/${USER}/brats_intervention
INPUTS_ROOT="${SCRATCH_ROOT}/inputs"
PREDS_ROOT="${SCRATCH_ROOT}/preds"
LOGS_DIR="${SCRATCH_ROOT}/logs"
mkdir -p "${LOGS_DIR}"

case "${MODEL_KEY}" in
  t1n_noise)
    NNUNET_RESULTS="/scratch/${USER}/brats_ladder_models/t1n_noise"
    DATASET_ID=51
    TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
    SETS="symmetry_tswap_t2f_donor_t1n"
    TIME="00:20:00"
    ;;
  t1n_real)
    NNUNET_RESULTS="/scratch/${USER}/brats_ladder_models/t1n_real"
    DATASET_ID=51
    TRAINER="nnUNetTrainerBraTS2024GliomaV26_6_2_train050_val100"
    SETS="symmetry_tswap_t2f_donor_t1n"
    TIME="00:20:00"
    ;;
  t2w_noise)
    NNUNET_RESULTS="${PWD}/benchmark/02_tasks/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/auglab/brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659"
    DATASET_ID=52
    TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
    SETS="x1_tswap_t1n_donor_t2w x2_tswap_t2w_donor_t1n x3_tswap_t1n_donor_t1c x4_permute_t1n sham_tswap_t1n_donor_t2w sham_tswap_t2w_donor_t1n"
    TIME="00:45:00"
    ;;
  t2w_real)
    NNUNET_RESULTS="${PWD}/benchmark/02_tasks/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/nnUNet/brats2024-glioma_t2w_v26_6_2_train050_val100_20260620_125217"
    DATASET_ID=52
    TRAINER="nnUNetTrainerBraTS2024GliomaT2wV26_6_2_train050_val100"
    SETS="x1_tswap_t1n_donor_t2w x2_tswap_t2w_donor_t1n x3_tswap_t1n_donor_t1c x4_permute_t1n sham_tswap_t1n_donor_t2w sham_tswap_t2w_donor_t1n"
    TIME="00:45:00"
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

run_job --name "brats_transplantpred_${MODEL_KEY}" --gpus 1 --cpus 8 --mem 64G --time "${TIME}" --wait \
  --log "${LOGS_DIR}/transplant_predict_${MODEL_KEY}.log" -- \
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
