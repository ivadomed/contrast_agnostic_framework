#!/usr/bin/env bash
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source datasets/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh

run_job --name brats_copolarity_partA --gpus 0 --cpus 8 --mem 32G --time 02:00:00 --wait \
  --log /scratch/paulh/brats_intervention/logs/copolarity_partA.log -- \
  .venv/bin/python datasets/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1/scripts/copolarity_partA_build.py
