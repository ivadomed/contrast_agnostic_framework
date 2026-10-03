#!/usr/bin/env bash
# Run the CHAOS T2spir causal-ablation ladder (06_35_ladder_summary_t2spir.py) as a CPU job via run_job
# (no python on the login node). Refuses, via the .py's own guard, unless every rung exists in every external
# cohort (amos/sliver07 fov_crop). Usage: bash 06_37_run_ladder_summary_t2spir.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env_t2spir.sh"
cd "${PROJECT_ROOT}"
run_job --name chaos_ladder_t2spir --gpus 0 --mem 16G --time "00:30:00" --wait \
    --log "${HERE}/../../8_results_chaos/02_metrics/chaos_model/t2spir/ablations/_ladder_run.log" -- \
    .venv/bin/python "${HERE}/06_35_ladder_summary_t2spir.py"
