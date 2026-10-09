#!/usr/bin/env bash
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh

run_job --name brats_copolarity_partB_build --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --wait \
  --log /scratch/paulh/brats_intervention/logs/copolarity_partB_build.log -- \
  .venv/bin/python benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1/scripts/copolarity_partB_build.py
