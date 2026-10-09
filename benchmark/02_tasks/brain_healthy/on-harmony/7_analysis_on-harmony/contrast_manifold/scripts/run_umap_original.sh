#!/usr/bin/env bash
# Plot UMAP of the ORIGINAL (real) data only — no synthetic data.
#
# Usage:
#   bash run_umap_original.sh [MASK_TYPE] [VERSION_RUN]
# Defaults:
#   MASK_TYPE   = regional_hist_64
#   VERSION_RUN = v19/v19_c_r1
set -euo pipefail

MASK_TYPE="${1:-regional_hist_64}"
VERSION_RUN="${2:-v19/v19_c_r1}"

# contrast_manifold root (this script lives in scripts/)
CM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# repo root = .../mri_synthesis_project
REPO_ROOT="$(cd "$CM_ROOT/../../../.." && pwd)"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"

ORIGINAL_CSV="$CM_ROOT/outputs/data/original/$MASK_TYPE/on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
OUTPUT_DIR="$CM_ROOT/outputs/plots/$VERSION_RUN/$MASK_TYPE/umap"

echo "Original CSV : $ORIGINAL_CSV"
echo "Output dir   : $OUTPUT_DIR"

run_job --gpus 0 --slot 0 --wait -- "$REPO_ROOT/.venv/bin/python" \
    "$CM_ROOT/scripts/plot_umap_original.py" \
    --original_csv "$ORIGINAL_CSV" \
    --output_dir   "$OUTPUT_DIR"
