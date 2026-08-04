#!/usr/bin/env bash
# Cross-dataset TASK-LEVEL heatmap (the "which method wins overall" table): one row
# per method, one column per dataset (task) + overall + significance vs ref. Tiny
# CSV I/O — safe on the Vulcan login node. Companion to run_meta_significance.sh
# (which does the sign-test/Stouffer combined significance) and run_significance_all.sh
# (per-dataset tables) — this is the visual "which method wins" summary.
#
#   bash scripts/evaluate/run_meta_task_heatmap.sh [meta_config.yaml]
#
# Default meta-config: scripts/evaluate/meta_task_heatmap.yaml
# Writes: datasets/01_commun_results/meta_task_heatmap_summary.md
#         datasets/01_commun_results/meta_task_heatmap_heatmap_{dice,hd95}.png
set -euo pipefail
: "${PROJECT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROJECT_ROOT
PY="${PROJECT_ROOT}/.venv/bin/python"
SCRIPT="${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/meta_task_heatmap.py"
CFG="${1:-${PROJECT_ROOT}/scripts/evaluate/meta_task_heatmap.yaml}"
"$PY" "$SCRIPT" "$CFG"
