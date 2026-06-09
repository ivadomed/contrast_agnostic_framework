#!/usr/bin/env bash
# Pipeline: v26_14 (mixed flat+affine), v26_15 (double remap)
# v26_14 is already partially generated; will skip complete subjects.
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project

PY=".venv/bin/python"
ANALYSIS="analysis/contrast_manifold/scripts"
DATA_ROOT="analysis/contrast_manifold/outputs/data"
ORIG_CSV="$DATA_ROOT/original/regional_hist_64/on_harmony_features.csv"
FEAT_CFG="analysis/contrast_manifold/config/feature_selection.yaml"

mkdir -p /tmp/v261415
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v261415/pipeline.log; }

# v26_15 generation (new, ~2× cost of v26_6 due to double remap)
log "=== Generating v26_15 ==="
set_slot 0 $PY scripts/generate_synthetic_guidance.py --generator v26_15 --lhc --n-variants 10 --rank 0 --world-size 4 > /tmp/v261415/gen_v26_15_r0.log 2>&1 & P0=$!
set_slot 1 $PY scripts/generate_synthetic_guidance.py --generator v26_15 --lhc --n-variants 10 --rank 1 --world-size 4 > /tmp/v261415/gen_v26_15_r1.log 2>&1 & P1=$!
set_slot 2 $PY scripts/generate_synthetic_guidance.py --generator v26_15 --lhc --n-variants 10 --rank 2 --world-size 4 > /tmp/v261415/gen_v26_15_r2.log 2>&1 & P2=$!
set_slot 3 $PY scripts/generate_synthetic_guidance.py --generator v26_15 --lhc --n-variants 10 --rank 3 --world-size 4 > /tmp/v261415/gen_v26_15_r3.log 2>&1 & P3=$!
wait $P0 $P1 $P2 $P3
log "v26_15: $(find data/ON-Harmony/derivatives/synthetic_v26_15_guidance_lhc -name '*.nii.gz' 2>/dev/null | wc -l)/1650"

# Extract (v26_14 and v26_15 in parallel — v26_14 may have fewer files)
log "=== Feature extraction ==="
mkdir -p "$DATA_ROOT/synthetic_v26_14_guidance_lhc/regional_hist_64" \
          "$DATA_ROOT/synthetic_v26_15_guidance_lhc/regional_hist_64"

set_slot 0 $PY $ANALYSIS/extract_features_regional_hist.py --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v26_14_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v26_14_guidance_lhc/regional_hist_64/synthetic_v26_14_guidance_lhc_features.csv" \
  --n-workers 28 > /tmp/v261415/extract_v26_14.log 2>&1 & P0=$!

set_slot 2 $PY $ANALYSIS/extract_features_regional_hist.py --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v26_15_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v26_15_guidance_lhc/regional_hist_64/synthetic_v26_15_guidance_lhc_features.csv" \
  --n-workers 28 > /tmp/v261415/extract_v26_15.log 2>&1 & P2=$!

wait $P0 $P2
log "Extraction done."

# Normalize
log "=== Normalize ==="
for ver in v26_14 v26_15; do
  $PY $ANALYSIS/normalize_combined.py \
    --original_csv "$ORIG_CSV" \
    --synthetic_csv "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features.csv" \
    --output_original  "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
    --output_synthetic "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features_normalized_combined.csv" \
    --feature_config "$FEAT_CFG" > /tmp/v261415/norm_${ver}.log 2>&1
done
log "Normalize done."

# Analysis
log "=== Analysis ==="
set_slot 0 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only v26_14_guidance_lhc_r1 > /tmp/v261415/analysis_v26_14.log 2>&1 & P0=$!
set_slot 1 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only v26_15_guidance_lhc_r1 > /tmp/v261415/analysis_v26_15.log 2>&1 & P1=$!
wait $P0 $P1
log "=== ALL DONE ==="
