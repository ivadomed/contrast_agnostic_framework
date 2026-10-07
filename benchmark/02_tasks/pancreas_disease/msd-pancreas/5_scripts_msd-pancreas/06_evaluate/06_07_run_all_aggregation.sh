#!/usr/bin/env bash
# Regenerate every msd-pancreas table + ladder in ONE CPU job (after 06_06 and 06_05). Never python on the login node.   bash 06_07_run_all_aggregation.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"; cd "${PROJECT_ROOT}"; mkdir -p "${RESULTS_DIR}/_logs"
run_job --name msd-pancreas_aggregation --gpus 0 --cpus 4 --mem 16G --time 01:00:00 --log "${RESULTS_DIR}/_logs/aggregation_$(date +%Y%m%d_%H%M%S).log" --wait -- bash -c "
set -euo pipefail
cd '${PROJECT_ROOT}'
for c in '${HERE}'/configs/msd-pancreas_*_01_results.yaml; do
    case \"\$c\" in *combined*) continue;; esac
    bash '${HERE}/06_02_aggregate_from_config.sh' \"\$c\"
done
bash '${HERE}/06_04_combined_modality_summary.sh'
.venv/bin/python '${HERE}/06_10_ladder_summary_pansegdatacross_ct_t1wce.py'
.venv/bin/python '${HERE}/06_11_ladder_summary_pansegdatacross_ct_t2w.py'
"
