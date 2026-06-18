#!/usr/bin/env bash
# Strict by-the-book H0 Betti curve (+ relative-persistence bridging) in PCA space.
#
# Usage:
#   bash run_betti.sh <DATA_SUBDIR> <SYNTH_PREFIX> <OUTPUT_DIR> [MASK_TYPE]
set -euo pipefail

DATA_SUBDIR="${1:?data subdir, e.g. synthetic_v26_6_guidance_lhc}"
SYNTH_PREFIX="${2:?synth prefix, e.g. synthetic_v26_6_guidance_lhc}"
OUT_REL="${3:?output dir relative to contrast_manifold root}"
MASK_TYPE="${4:-regional_hist_64}"

CM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$CM_ROOT/../../../.." && pwd)"

DATA_DIR="$CM_ROOT/outputs/data/$DATA_SUBDIR/$MASK_TYPE"
ORIGINAL_CSV="$DATA_DIR/on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
SYNTH_CSV="$DATA_DIR/${SYNTH_PREFIX}_features_normalized_combined_feat_selected.csv"
OUTPUT_DIR="$CM_ROOT/$OUT_REL"

echo "Original CSV : $ORIGINAL_CSV"
echo "Synthetic CSV: $SYNTH_CSV"
echo "Output dir   : $OUTPUT_DIR"

set_slot 0 "$REPO_ROOT/.venv/bin/python" \
    "$CM_ROOT/scripts/betti_curve.py" \
    --original_csv  "$ORIGINAL_CSV" \
    --synthetic_csv "$SYNTH_CSV" \
    --output_dir    "$OUTPUT_DIR"
