#!/usr/bin/env bash
# GPU inference for the visibility-step intervention (S-ADD dose-response + S-REMOVE + sham).
# Same nnUNetv2_predict invocation as the earlier intervention predict scripts.
# Usage: bash run_intervention_step_predict.sh <model_key>
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh

MODEL_KEY="${1:?model_key required: t1n_noise|t1n_real|t2w_noise|t2w_real}"

SCRATCH_ROOT=/scratch/paulh/brats_intervention
INPUTS_ROOT="${SCRATCH_ROOT}/inputs"
PREDS_ROOT="${SCRATCH_ROOT}/preds"
LOGS_DIR="${SCRATCH_ROOT}/logs"
mkdir -p "${LOGS_DIR}"

V1_SETS="v1_sadd_t1n_kp050 v1_sadd_t1n_kp100 v1_sadd_t1n_km050 v1_sadd_t1n_km100"
V1_SHAM_SETS="sham_v1_sadd_t1n_kp050 sham_v1_sadd_t1n_kp100 sham_v1_sadd_t1n_km050 sham_v1_sadd_t1n_km100"
V2_SETS="v2_sremove_t2w v2_sremove_t2f"
V2_SHAM_SETS="sham_v2_sremove_t2w sham_v2_sremove_t2f"

case "${MODEL_KEY}" in
  t1n_noise)
    NNUNET_RESULTS="/scratch/paulh/brats_ladder_models/t1n_noise"
    DATASET_ID=51
    TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
    SETS="v2_sremove_t2f"
    TIME="00:20:00"
    ;;
  t1n_real)
    NNUNET_RESULTS="/scratch/paulh/brats_ladder_models/t1n_real"
    DATASET_ID=51
    TRAINER="nnUNetTrainerBraTS2024GliomaV26_6_2_train050_val100"
    SETS="v2_sremove_t2f"
    TIME="00:20:00"
    ;;
  t2w_noise)
    NNUNET_RESULTS="/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/02_tasks/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/auglab/brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659"
    DATASET_ID=52
    TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
    SETS="${V1_SETS} ${V1_SHAM_SETS} ${V2_SETS} ${V2_SHAM_SETS}"
    TIME="01:15:00"
    ;;
  t2w_real)
    NNUNET_RESULTS="/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/02_tasks/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/nnUNet/brats2024-glioma_t2w_v26_6_2_train050_val100_20260620_125217"
    DATASET_ID=52
    TRAINER="nnUNetTrainerBraTS2024GliomaT2wV26_6_2_train050_val100"
    SETS="${V1_SETS} ${V1_SHAM_SETS} ${V2_SETS} ${V2_SHAM_SETS}"
    TIME="01:15:00"
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

run_job --name "brats_steppred_${MODEL_KEY}" --gpus 1 --cpus 8 --mem 64G --time "${TIME}" --wait \
  --log "${LOGS_DIR}/step_predict_${MODEL_KEY}.log" -- \
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
