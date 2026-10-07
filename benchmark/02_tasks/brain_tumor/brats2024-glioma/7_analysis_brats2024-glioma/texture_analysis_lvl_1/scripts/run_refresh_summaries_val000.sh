#!/usr/bin/env bash
# 2026-10-07: re-run every BraTS predictor-vs-fill-swap summary on the refreshed outcomes (rung 5 = val000 retrains,
# region_fill_swap_significance.csv / patient_region_deltas.csv rebuilt with the t1c arm). Pure CSV->table scripts, in
# dependency order; a failure is logged and the chain continues. Submit through run_job (CPU), e.g.
#   run_job --name refresh_summ --gpus 0 --cpus 4 --mem 16G --time 02:00:00 --log outputs/logs/refresh_summaries_val000.log -- bash scripts/run_refresh_summaries_val000.sh
set -uo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${S}/../../../../../../.." && pwd)"; PY="${REPO}/.venv/bin/python"
cd "${REPO}"
rc=0
for f in ngf_vs_fill_swap_summary containment_vs_fill_swap_summary synthesis_vs_fill_swap_summary ranking_within_train \
         step_affine_vs_fill_swap_summary internal_ramp_vs_fill_swap region_surround_vs_fill_swap_summary anisotropy_summary \
         boundary_vs_fill_swap_summary anatomy_vs_pathology_texture predictor_combinations explanation_scorecard; do
  echo "=== ${f} $(date +%T)"
  "${PY}" "${S}/${f}.py" || { echo "!!! ${f} FAILED"; rc=1; }
done
echo "refresh_summaries rc=${rc}"; exit ${rc}
