#!/usr/bin/env bash
# Smoke run: V19 guidance aug + wandb, 3 epochs, fold 0, Dataset104 t2w
set -euo pipefail

REPO=/home/ge.polymtl.ca/pahoa/mri_synthesis_project

export NNUNET_RAW="${REPO}/data/nnUNet_raw"
export NNUNET_PREPROCESSED="${REPO}/data/nnUNet_preprocessed_t2w_gen19"
export NNUNET_RESULTS="${REPO}/data/nnUNet_results_v19_smoke"
export PYTHONPATH="${REPO}/v19_guidance_map_augmentation:${PYTHONPATH:-}"

export nnUNet_wandb_enabled=1
export nnUNet_wandb_project=mri-synthesis-v19
export nnUNet_wandb_run_name=v19-guidance-aug-smoke
export WANDB_ENTITY=p-hoareau33-centrale-lyon

mkdir -p "${NNUNET_RESULTS}"

exec "${REPO}/.venv/bin/nnUNetv2_train" 104 3d_fullres 0 \
    -tr nnUNetTrainerV19SmokeTest \
    -num_gpus 1 \
    "$@"
