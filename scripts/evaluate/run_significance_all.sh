#!/usr/bin/env bash
# Paired significance tests for the headline cross-dataset / cross-contrast tables.
# Tiny CSV I/O (well within the Vulcan login-node exception) — safe to run directly.
#
#   bash scripts/evaluate/run_significance_all.sh
#
# Emits {output_dir}/{output_prefix}_significance.md next to each summary table,
# and reports p-values under 4-fold and 3-fold side-by-side.
set -euo pipefail

: "${PROJECT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROJECT_ROOT
PY="${PROJECT_ROOT}/.venv/bin/python"
# Canonical significance script lives in the shared dataset-pipeline layer (00_commun_scripts).
SIG="${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/significance_from_config.py"

CONFIGS=(
  "datasets/chaos/5_scripts_chaos/06_evaluate/configs/cross_dataset_t1in_01_results.yaml"
  "datasets/chaos/5_scripts_chaos/06_evaluate/configs/cross_dataset_t2spir_01_results.yaml"
  "datasets/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/configs/brats_t1n_01_results.yaml"
  "datasets/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/configs/brats_t2w_01_results.yaml"
  "datasets/on-harmony/5_scripts_on-harmony/06_evaluate/configs/on-harmony_T1w.yaml"
  "datasets/on-harmony/5_scripts_on-harmony/06_evaluate/configs/on-harmony_T2w.yaml"
)

# Derive METRICS_ROOT from a config path of the form
#   datasets/<ds>/5_scripts_<ds>/06_evaluate/configs/<name>.yaml
# -> ${PROJECT_ROOT}/datasets/<ds>/8_results_<ds>/02_metrics
# (chaos configs hard-code the full path via ${PROJECT_ROOT}; the others use ${METRICS_ROOT}.)
metrics_root_for() {
  local ds
  ds="$(printf '%s\n' "$1" | sed -E 's#^datasets/([^/]+)/.*#\1#')"
  printf '%s/datasets/%s/8_results_%s/02_metrics' "${PROJECT_ROOT}" "$ds" "$ds"
}

for cfg in "${CONFIGS[@]}"; do
  echo "================================================================"
  echo "### ${cfg}"
  echo "================================================================"
  METRICS_ROOT="$(metrics_root_for "$cfg")" \
    "$PY" "$SIG" "${PROJECT_ROOT}/${cfg}" "$@" || echo "  (skipped: ${cfg})"
  echo
done
