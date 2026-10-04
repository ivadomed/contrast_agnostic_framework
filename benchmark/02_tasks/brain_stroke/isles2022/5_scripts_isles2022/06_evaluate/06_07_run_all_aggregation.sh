#!/usr/bin/env bash
# Regenerate every isles2022 table + ladder in ONE CPU job: per-contrast headline tables, significance, the combined table
# and both ladders. Run AFTER 06_06_run_all_eval.sh (and 06_05_write_configs.sh). Never runs python on the login node.
#   bash 06_07_run_all_aggregation.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_DIR}/_logs"
run_job --name isles2022_aggregation --gpus 0 --cpus 4 --mem 16G --time 01:00:00 \
    --log "${RESULTS_DIR}/_logs/aggregation_$(date +%Y%m%d_%H%M%S).log" --wait -- bash -c "
set -euo pipefail
cd '${PROJECT_ROOT}'
for C in dwi flair; do
    bash '${HERE}/06_02_aggregate_from_config.sh' configs/isles2022_\${C}_01_results.yaml
    bash '${HERE}/06_03_significance_from_config.sh' configs/isles2022_\${C}_significance_01.yaml
done
bash '${HERE}/06_04_combined_modality_summary.sh'
.venv/bin/python '${HERE}/06_10_ladder_summary_dwi.py'
.venv/bin/python '${HERE}/06_11_ladder_summary_flair.py'
"
