#!/usr/bin/env bash
#SBATCH --job-name=ambl_aggregate
#SBATCH --account=aip-jcohen
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/paulh/ambl_aggregate_%j.out
# One-off driver: run the ambl aggregation/significance/combined/ladder summary
# steps on Vulcan (metrics already rsynced onto the project-relative path).
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
HERE="datasets/ambl/5_scripts_ambl/06_evaluate"

echo "=== 06_10 aggregate: t1wce ==="
bash "${HERE}/06_10_aggregate_from_config.sh" configs/ambl_t1wce_01_results.yaml
echo "=== 06_10 aggregate: t2w ==="
bash "${HERE}/06_10_aggregate_from_config.sh" configs/ambl_t2w_01_results.yaml

echo "=== 06_11 significance: t1wce ==="
bash "${HERE}/06_11_significance_from_config.sh" configs/ambl_t1wce_01_results.yaml
echo "=== 06_11 significance: t2w ==="
bash "${HERE}/06_11_significance_from_config.sh" configs/ambl_t2w_01_results.yaml

echo "=== 06_12 combined_modality_summary ==="
bash "${HERE}/06_12_combined_modality_summary.sh" configs/ambl_combined_01_results.yaml

echo "=== 06_13 ladder t1wce ==="
.venv/bin/python "${HERE}/06_13_ladder_summary_t1wce.py"
echo "=== 06_14 ladder t2w ==="
.venv/bin/python "${HERE}/06_14_ladder_summary_t2w.py"

echo "=== DONE ==="
