#!/usr/bin/env bash
# Pipeline: v26_11 (large-K), v26_12 (stratified mu), v26_13 (large-K + stratified)
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project

PY=".venv/bin/python"
ANALYSIS="analysis/contrast_manifold/scripts"
DATA_ROOT="analysis/contrast_manifold/outputs/data"
ORIG_CSV="$DATA_ROOT/original/regional_hist_64/on_harmony_features.csv"
FEAT_CFG="analysis/contrast_manifold/config/feature_selection.yaml"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v261113/pipeline.log; }

# Phase 1: Generate (sequential per version, parallel 4 GPUs each)
for ver in v26_11 v26_12 v26_13; do
  log "=== Generating $ver ==="
  set_slot 0 $PY scripts/generate_synthetic_guidance.py --generator $ver --lhc --n-variants 10 --rank 0 --world-size 4 > /tmp/v261113/gen_${ver}_r0.log 2>&1 & P0=$!
  set_slot 1 $PY scripts/generate_synthetic_guidance.py --generator $ver --lhc --n-variants 10 --rank 1 --world-size 4 > /tmp/v261113/gen_${ver}_r1.log 2>&1 & P1=$!
  set_slot 2 $PY scripts/generate_synthetic_guidance.py --generator $ver --lhc --n-variants 10 --rank 2 --world-size 4 > /tmp/v261113/gen_${ver}_r2.log 2>&1 & P2=$!
  set_slot 3 $PY scripts/generate_synthetic_guidance.py --generator $ver --lhc --n-variants 10 --rank 3 --world-size 4 > /tmp/v261113/gen_${ver}_r3.log 2>&1 & P3=$!
  wait $P0 $P1 $P2 $P3
  log "$ver: $(find data/ON-Harmony/derivatives/synthetic_${ver}_guidance_lhc -name '*.nii.gz' 2>/dev/null | wc -l)/1650"
done

# Phase 2: Extract (3 parallel)
log "=== Feature extraction ==="
for ver in v26_11 v26_12 v26_13; do
  mkdir -p "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64"
done
set_slot 0 $PY $ANALYSIS/extract_features_regional_hist.py --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v26_11_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v26_11_guidance_lhc/regional_hist_64/synthetic_v26_11_guidance_lhc_features.csv" \
  --n-workers 14 > /tmp/v261113/extract_v26_11.log 2>&1 & P0=$!
set_slot 1 $PY $ANALYSIS/extract_features_regional_hist.py --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v26_12_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v26_12_guidance_lhc/regional_hist_64/synthetic_v26_12_guidance_lhc_features.csv" \
  --n-workers 14 > /tmp/v261113/extract_v26_12.log 2>&1 & P1=$!
set_slot 2 $PY $ANALYSIS/extract_features_regional_hist.py --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v26_13_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v26_13_guidance_lhc/regional_hist_64/synthetic_v26_13_guidance_lhc_features.csv" \
  --n-workers 14 > /tmp/v261113/extract_v26_13.log 2>&1 & P2=$!
wait $P0 $P1 $P2
log "Extraction done."

# Phase 3: Normalize
log "=== Normalize ==="
for ver in v26_11 v26_12 v26_13; do
  $PY $ANALYSIS/normalize_combined.py \
    --original_csv "$ORIG_CSV" \
    --synthetic_csv "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features.csv" \
    --output_original  "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
    --output_synthetic "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features_normalized_combined.csv" \
    --feature_config "$FEAT_CFG" > /tmp/v261113/norm_${ver}.log 2>&1
done
log "Normalize done."

# Phase 4: Analysis (3 parallel)
log "=== Analysis ==="
set_slot 0 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only v26_11_guidance_lhc_r1 > /tmp/v261113/analysis_v26_11.log 2>&1 & P0=$!
set_slot 1 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only v26_12_guidance_lhc_r1 > /tmp/v261113/analysis_v26_12.log 2>&1 & P1=$!
set_slot 2 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only v26_13_guidance_lhc_r1 > /tmp/v261113/analysis_v26_13.log 2>&1 & P2=$!
wait $P0 $P1 $P2
log "Analysis done."

log "=== ALL DONE ==="
