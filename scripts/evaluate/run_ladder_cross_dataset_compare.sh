#!/usr/bin/env bash
# Cross-dataset causal-ablation ladder comparison for the breast task
# (ispy2-trained models, tested on ispy2's own held-out contrast + on
# duke-breast-mri as an external dataset). One plot per training direction,
# grouped by TRUE held-out contrast identity (not by source dataset) --
# duke's own t1wce/DCE acquisition is only OOD relative to a model NOT
# trained on t1wce (i.e. the t2w-trained direction); for the t1wce-trained
# direction duke's t1wce test is same-contrast, cross-dataset only, so it is
# left out of this OOD-contrast comparison entirely:
#   - t1wce-trained: "t2w" (ispy2's own held-out contrast only -- duke has no
#     t2w acquisition) vs "t1w (precontrast)" (duke only)
#   - t2w-trained: "t1wce" (ispy2's own held-out contrast AND duke's t1wce --
#     both are genuinely the t1wce contrast, held out from a t2w-trained
#     model, so pooled together) vs "t1w (precontrast)" (duke only)
# See ladder_cross_dataset_plot.py's own docstring for the full rationale.
# Cheap (matplotlib only, reads already-computed ladder_series.json files)
# -- fine to run on the login node.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"
PLOTTER="$REPO_ROOT/datasets/00_commun_scripts/00_03_evaluate/ladder_cross_dataset_plot.py"
# Training dataset is ispy2 (these are ispy2_model results, just scored on
# external test data too) -- output lives under ispy2's own results tree,
# not datasets/01_commun_results/ (which is reserved for cross-TRAINING-SET
# comparisons, not one training set's own cross-dataset eval).
OUT_ROOT="$REPO_ROOT/datasets/ispy2/8_results_ispy2/02_metrics/ispy2_model"

ISPY2_T1WCE="$REPO_ROOT/datasets/ispy2/8_results_ispy2/02_metrics/ispy2_model/t1wce/ablations/ladder_series.json"
ISPY2_T2W="$REPO_ROOT/datasets/ispy2/8_results_ispy2/02_metrics/ispy2_model/t2w/ablations/ladder_series.json"
DUKE_T1WCE_TRAINED="$REPO_ROOT/datasets/duke-breast-mri/8_results_duke-breast-mri/02_metrics/ispy2_model/t1wce/ablations/ladder_series.json"
DUKE_T2W_TRAINED="$REPO_ROOT/datasets/duke-breast-mri/8_results_duke-breast-mri/02_metrics/ispy2_model/t2w/ablations/ladder_series.json"
DUKE_PRECONTRAST_T1WCE="$REPO_ROOT/datasets/duke-breast-mri/8_results_duke-breast-mri/02_metrics/ispy2_model/t1wce/ablations/precontrast/ladder_series.json"
DUKE_PRECONTRAST_T2W="$REPO_ROOT/datasets/duke-breast-mri/8_results_duke-breast-mri/02_metrics/ispy2_model/t2w/ablations/precontrast/ladder_series.json"

# DUKE_T1WCE_TRAINED (duke's t1wce test scored against the t1wce-trained
# model) is deliberately unused here -- same-contrast, cross-dataset only,
# not an OOD contrast for this direction. Kept as its own ladder
# (06_21_ladder_summary_ispy2cross_t1wce.py) elsewhere, just not pooled in.

echo "=== T1WCE-trained ==="
"$PY" "$PLOTTER" \
  "$OUT_ROOT/t1wce/ablations" t1wce \
  "ispy2 T1WCE-trained, cross-dataset" \
  "t2w=${ISPY2_T1WCE}" \
  "t1w (precontrast)=${DUKE_PRECONTRAST_T1WCE}"

echo
echo "=== T2W-trained ==="
"$PY" "$PLOTTER" \
  "$OUT_ROOT/t2w/ablations" t2w \
  "ispy2 T2W-trained, cross-dataset" \
  "t1wce=${ISPY2_T2W},${DUKE_T2W_TRAINED}" \
  "t1w (precontrast)=${DUKE_PRECONTRAST_T2W}"
