#!/usr/bin/env bash
# Full pipeline for v26_5, v26_6, v26_7:
#   generate (10 variants/subject, 4 GPUs) → extract regional_hist_64 → normalize → analyze
# v26_5 and v26_6 are fast (~1s/vol); v26_7 is slower (~6s/vol due to flat assignments).
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project

PY=".venv/bin/python"
ANALYSIS="analysis/contrast_manifold/scripts"
DATA_ROOT="analysis/contrast_manifold/outputs/data"
ORIG_CSV="$DATA_ROOT/original/regional_hist_64/on_harmony_features.csv"
FEAT_CFG="analysis/contrast_manifold/config/feature_selection.yaml"

mkdir -p /tmp/v26567
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v26567/pipeline.log; }

# ── PHASE 1: Generate (sequential per version, parallel across 4 GPU slots) ─────
log "=== PHASE 1a: v26_5 generation ==="
set_slot 0 $PY scripts/generate_synthetic_guidance.py --generator v26_5 --lhc --n-variants 10 --rank 0 --world-size 4 > /tmp/v26567/gen_v26_5_r0.log 2>&1 & P0=$!
set_slot 1 $PY scripts/generate_synthetic_guidance.py --generator v26_5 --lhc --n-variants 10 --rank 1 --world-size 4 > /tmp/v26567/gen_v26_5_r1.log 2>&1 & P1=$!
set_slot 2 $PY scripts/generate_synthetic_guidance.py --generator v26_5 --lhc --n-variants 10 --rank 2 --world-size 4 > /tmp/v26567/gen_v26_5_r2.log 2>&1 & P2=$!
set_slot 3 $PY scripts/generate_synthetic_guidance.py --generator v26_5 --lhc --n-variants 10 --rank 3 --world-size 4 > /tmp/v26567/gen_v26_5_r3.log 2>&1 & P3=$!
wait $P0 $P1 $P2 $P3
log "v26_5 done: $(find data/ON-Harmony/derivatives/synthetic_v26_5_guidance_lhc -name '*.nii.gz' 2>/dev/null | wc -l)/1650"

log "=== PHASE 1b: v26_6 generation ==="
set_slot 0 $PY scripts/generate_synthetic_guidance.py --generator v26_6 --lhc --n-variants 10 --rank 0 --world-size 4 > /tmp/v26567/gen_v26_6_r0.log 2>&1 & P0=$!
set_slot 1 $PY scripts/generate_synthetic_guidance.py --generator v26_6 --lhc --n-variants 10 --rank 1 --world-size 4 > /tmp/v26567/gen_v26_6_r1.log 2>&1 & P1=$!
set_slot 2 $PY scripts/generate_synthetic_guidance.py --generator v26_6 --lhc --n-variants 10 --rank 2 --world-size 4 > /tmp/v26567/gen_v26_6_r2.log 2>&1 & P2=$!
set_slot 3 $PY scripts/generate_synthetic_guidance.py --generator v26_6 --lhc --n-variants 10 --rank 3 --world-size 4 > /tmp/v26567/gen_v26_6_r3.log 2>&1 & P3=$!
wait $P0 $P1 $P2 $P3
log "v26_6 done: $(find data/ON-Harmony/derivatives/synthetic_v26_6_guidance_lhc -name '*.nii.gz' 2>/dev/null | wc -l)/1650"

log "=== PHASE 1c: v26_7 generation ==="
set_slot 0 $PY scripts/generate_synthetic_guidance.py --generator v26_7 --lhc --n-variants 10 --rank 0 --world-size 4 > /tmp/v26567/gen_v26_7_r0.log 2>&1 & P0=$!
set_slot 1 $PY scripts/generate_synthetic_guidance.py --generator v26_7 --lhc --n-variants 10 --rank 1 --world-size 4 > /tmp/v26567/gen_v26_7_r1.log 2>&1 & P1=$!
set_slot 2 $PY scripts/generate_synthetic_guidance.py --generator v26_7 --lhc --n-variants 10 --rank 2 --world-size 4 > /tmp/v26567/gen_v26_7_r2.log 2>&1 & P2=$!
set_slot 3 $PY scripts/generate_synthetic_guidance.py --generator v26_7 --lhc --n-variants 10 --rank 3 --world-size 4 > /tmp/v26567/gen_v26_7_r3.log 2>&1 & P3=$!
wait $P0 $P1 $P2 $P3
log "v26_7 done: $(find data/ON-Harmony/derivatives/synthetic_v26_7_guidance_lhc -name '*.nii.gz' 2>/dev/null | wc -l)/1650"

# ── PHASE 2: Feature extraction (3 versions in parallel, regional_hist_64) ──────
log "=== PHASE 2: Feature extraction ==="
mkdir -p \
  "$DATA_ROOT/synthetic_v26_5_guidance_lhc/regional_hist_64" \
  "$DATA_ROOT/synthetic_v26_6_guidance_lhc/regional_hist_64" \
  "$DATA_ROOT/synthetic_v26_7_guidance_lhc/regional_hist_64"

set_slot 0 $PY $ANALYSIS/extract_features_regional_hist.py \
  --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v26_5_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v26_5_guidance_lhc/regional_hist_64/synthetic_v26_5_guidance_lhc_features.csv" \
  --n-workers 14 > /tmp/v26567/extract_v26_5.log 2>&1 & P0=$!

set_slot 1 $PY $ANALYSIS/extract_features_regional_hist.py \
  --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v26_6_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v26_6_guidance_lhc/regional_hist_64/synthetic_v26_6_guidance_lhc_features.csv" \
  --n-workers 14 > /tmp/v26567/extract_v26_6.log 2>&1 & P1=$!

set_slot 2 $PY $ANALYSIS/extract_features_regional_hist.py \
  --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v26_7_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v26_7_guidance_lhc/regional_hist_64/synthetic_v26_7_guidance_lhc_features.csv" \
  --n-workers 14 > /tmp/v26567/extract_v26_7.log 2>&1 & P2=$!

wait $P0 $P1 $P2
log "Phase 2 done."

# ── PHASE 3: Normalize (3 in parallel) ───────────────────────────────────────────
log "=== PHASE 3: Normalize ==="

set_slot 0 $PY $ANALYSIS/normalize_combined.py \
  --original_csv  "$ORIG_CSV" \
  --synthetic_csv "$DATA_ROOT/synthetic_v26_5_guidance_lhc/regional_hist_64/synthetic_v26_5_guidance_lhc_features.csv" \
  --output_original  "$DATA_ROOT/synthetic_v26_5_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
  --output_synthetic "$DATA_ROOT/synthetic_v26_5_guidance_lhc/regional_hist_64/synthetic_v26_5_guidance_lhc_features_normalized_combined.csv" \
  --feature_config "$FEAT_CFG" > /tmp/v26567/norm_v26_5.log 2>&1 & P0=$!

set_slot 1 $PY $ANALYSIS/normalize_combined.py \
  --original_csv  "$ORIG_CSV" \
  --synthetic_csv "$DATA_ROOT/synthetic_v26_6_guidance_lhc/regional_hist_64/synthetic_v26_6_guidance_lhc_features.csv" \
  --output_original  "$DATA_ROOT/synthetic_v26_6_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
  --output_synthetic "$DATA_ROOT/synthetic_v26_6_guidance_lhc/regional_hist_64/synthetic_v26_6_guidance_lhc_features_normalized_combined.csv" \
  --feature_config "$FEAT_CFG" > /tmp/v26567/norm_v26_6.log 2>&1 & P1=$!

set_slot 2 $PY $ANALYSIS/normalize_combined.py \
  --original_csv  "$ORIG_CSV" \
  --synthetic_csv "$DATA_ROOT/synthetic_v26_7_guidance_lhc/regional_hist_64/synthetic_v26_7_guidance_lhc_features.csv" \
  --output_original  "$DATA_ROOT/synthetic_v26_7_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
  --output_synthetic "$DATA_ROOT/synthetic_v26_7_guidance_lhc/regional_hist_64/synthetic_v26_7_guidance_lhc_features_normalized_combined.csv" \
  --feature_config "$FEAT_CFG" > /tmp/v26567/norm_v26_7.log 2>&1 & P2=$!

wait $P0 $P1 $P2
log "Phase 3 done."

# ── PHASE 4: Analysis (3 in parallel) ────────────────────────────────────────────
log "=== PHASE 4: Analysis ==="

set_slot 0 $PY $ANALYSIS/run_all_analysis.py \
  --mask-type regional_hist_64 --only v26_5_guidance_lhc_r1 \
  > /tmp/v26567/analysis_v26_5.log 2>&1 & P0=$!

set_slot 1 $PY $ANALYSIS/run_all_analysis.py \
  --mask-type regional_hist_64 --only v26_6_guidance_lhc_r1 \
  > /tmp/v26567/analysis_v26_6.log 2>&1 & P1=$!

set_slot 2 $PY $ANALYSIS/run_all_analysis.py \
  --mask-type regional_hist_64 --only v26_7_guidance_lhc_r1 \
  > /tmp/v26567/analysis_v26_7.log 2>&1 & P2=$!

wait $P0 $P1 $P2
log "Phase 4 done."

log "=== ALL DONE ==="
