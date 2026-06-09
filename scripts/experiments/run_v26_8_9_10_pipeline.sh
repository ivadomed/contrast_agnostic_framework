#!/usr/bin/env bash
# Pipeline for v26_8 (global inversion), v26_9 (gamma), v26_10 (fractal noise)
# generate → extract regional_hist_64 → normalize → analyze
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project

PY=".venv/bin/python"
ANALYSIS="analysis/contrast_manifold/scripts"
DATA_ROOT="analysis/contrast_manifold/outputs/data"
ORIG_CSV="$DATA_ROOT/original/regional_hist_64/on_harmony_features.csv"
FEAT_CFG="analysis/contrast_manifold/config/feature_selection.yaml"

mkdir -p /tmp/v268910
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v268910/pipeline.log; }

# ── PHASE 1: Generate (parallel 4 GPUs per version, sequential per version) ────
for ver in v26_8 v26_9 v26_10; do
  log "=== PHASE 1: $ver generation ==="
  set_slot 0 $PY scripts/generate_synthetic_guidance.py --generator $ver --lhc --n-variants 10 --rank 0 --world-size 4 > /tmp/v268910/gen_${ver}_r0.log 2>&1 & P0=$!
  set_slot 1 $PY scripts/generate_synthetic_guidance.py --generator $ver --lhc --n-variants 10 --rank 1 --world-size 4 > /tmp/v268910/gen_${ver}_r1.log 2>&1 & P1=$!
  set_slot 2 $PY scripts/generate_synthetic_guidance.py --generator $ver --lhc --n-variants 10 --rank 2 --world-size 4 > /tmp/v268910/gen_${ver}_r2.log 2>&1 & P2=$!
  set_slot 3 $PY scripts/generate_synthetic_guidance.py --generator $ver --lhc --n-variants 10 --rank 3 --world-size 4 > /tmp/v268910/gen_${ver}_r3.log 2>&1 & P3=$!
  wait $P0 $P1 $P2 $P3
  log "$ver done: $(find data/ON-Harmony/derivatives/synthetic_${ver}_guidance_lhc -name '*.nii.gz' 2>/dev/null | wc -l)/1650"
done

# ── PHASE 2: Feature extraction (3 versions in parallel) ────────────────────────
log "=== PHASE 2: Feature extraction ==="
mkdir -p \
  "$DATA_ROOT/synthetic_v26_8_guidance_lhc/regional_hist_64" \
  "$DATA_ROOT/synthetic_v26_9_guidance_lhc/regional_hist_64" \
  "$DATA_ROOT/synthetic_v26_10_guidance_lhc/regional_hist_64"

set_slot 0 $PY $ANALYSIS/extract_features_regional_hist.py \
  --mode synthetic --synth-root data/ON-Harmony/derivatives/synthetic_v26_8_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v26_8_guidance_lhc/regional_hist_64/synthetic_v26_8_guidance_lhc_features.csv" \
  --n-workers 14 > /tmp/v268910/extract_v26_8.log 2>&1 & P0=$!

set_slot 1 $PY $ANALYSIS/extract_features_regional_hist.py \
  --mode synthetic --synth-root data/ON-Harmony/derivatives/synthetic_v26_9_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v26_9_guidance_lhc/regional_hist_64/synthetic_v26_9_guidance_lhc_features.csv" \
  --n-workers 14 > /tmp/v268910/extract_v26_9.log 2>&1 & P1=$!

set_slot 2 $PY $ANALYSIS/extract_features_regional_hist.py \
  --mode synthetic --synth-root data/ON-Harmony/derivatives/synthetic_v26_10_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v26_10_guidance_lhc/regional_hist_64/synthetic_v26_10_guidance_lhc_features.csv" \
  --n-workers 14 > /tmp/v268910/extract_v26_10.log 2>&1 & P2=$!

wait $P0 $P1 $P2
log "Phase 2 done."

# ── PHASE 3: Normalize ──────────────────────────────────────────────────────────
log "=== PHASE 3: Normalize ==="
for ver in v26_8 v26_9 v26_10; do
  set_slot 0 $PY $ANALYSIS/normalize_combined.py \
    --original_csv "$ORIG_CSV" \
    --synthetic_csv "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features.csv" \
    --output_original  "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
    --output_synthetic "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features_normalized_combined.csv" \
    --feature_config "$FEAT_CFG" > /tmp/v268910/norm_${ver}.log 2>&1
done
log "Phase 3 done."

# ── PHASE 4: Analysis (3 parallel) ─────────────────────────────────────────────
log "=== PHASE 4: Analysis ==="
set_slot 0 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only v26_8_guidance_lhc_r1  > /tmp/v268910/analysis_v26_8.log  2>&1 & P0=$!
set_slot 1 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only v26_9_guidance_lhc_r1  > /tmp/v268910/analysis_v26_9.log  2>&1 & P1=$!
set_slot 2 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only v26_10_guidance_lhc_r1 > /tmp/v268910/analysis_v26_10.log 2>&1 & P2=$!
wait $P0 $P1 $P2
log "Phase 4 done."

log "=== ALL DONE ==="
