#!/usr/bin/env bash
# Cross-dataset COMBINED significance (the paper's headline significance): sign test +
# Stouffer + dataset-clustered test over the per-dataset result configs. Tiny CSV I/O —
# safe on the Vulcan login node (well within the exception). Companion to
# scripts/evaluate/run_significance_all.sh (which runs the per-dataset tables).
#
#   bash scripts/evaluate/run_meta_significance.sh [meta_config.yaml]
#
# Default meta-config: scripts/evaluate/meta_significance_train050.yaml
set -euo pipefail
: "${PROJECT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROJECT_ROOT
PY="${PROJECT_ROOT}/.venv/bin/python"
META="${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/meta_significance.py"
CFG="${1:-${PROJECT_ROOT}/scripts/evaluate/meta_significance_train050.yaml}"
"$PY" "$META" "$CFG"
