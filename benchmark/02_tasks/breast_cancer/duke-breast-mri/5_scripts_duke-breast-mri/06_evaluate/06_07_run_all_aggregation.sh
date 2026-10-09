#!/usr/bin/env bash
# Run every duke-breast-mri aggregation step in ONE CPU run_job (not inline on the login
# node): per-item/per-training-contrast tables (06_02 over configs/duke_*_01_results.yaml),
# the combined table (06_04), and every causal-ablation ladder (06_1X).
#   bash 06_07_run_all_aggregation.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
mkdir -p "${METRICS_ROOT}/_logs"
run_job --name duke_aggregate --gpus 0 --cpus 2 --mem 8G --time 01:00:00 \
    --log "${METRICS_ROOT}/_logs/aggregate_$(date +%Y%m%d_%H%M%S).log" --wait -- bash -c "
set -euo pipefail
cd '${PROJECT_ROOT}'
for c in '${HERE}'/configs/duke_*_01_results.yaml; do
    case \"\$c\" in *combined*) continue;; esac
    bash '${HERE}/06_02_aggregate_from_config.sh' \"\$c\"
done
bash '${HERE}/06_04_combined_modality_summary.sh'
for l in '${HERE}'/06_1[0-9]_ladder_summary_*.py; do
    echo \"[ladder] \$l\"; .venv/bin/python \"\$l\"
done
"
