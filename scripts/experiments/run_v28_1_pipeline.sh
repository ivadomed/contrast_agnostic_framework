#!/usr/bin/env bash
# Full pipeline for v28_1: generate → extract regional_hist_64 + hog3d_512 → normalize → analyze
# v28_1 = V26_6 signed-alpha + Rician noise + aggressive resolution (zoom 0.20-1.0)
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project

PY=".venv/bin/python"
ANALYSIS="analysis/contrast_manifold/scripts"
DATA_ROOT="analysis/contrast_manifold/outputs/data"
FEAT_CFG="analysis/contrast_manifold/config/feature_selection.yaml"
SYNTH_ROOT="data/ON-Harmony/derivatives/synthetic_v28_1_guidance_lhc"
DATA_DIR="$DATA_ROOT/synthetic_v28_1_guidance_lhc"

mkdir -p /tmp/v28_1
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v28_1/pipeline.log; }

# ── Phase 1: Generate (set_slot 0-3 → 4 GPUs, parallelise over subjects) ──────
log "=== Phase 1: Generate v28_1 (4 GPUs, 1650 files) ==="
set_slot 0 $PY scripts/generate_synthetic_guidance.py --generator v28_1 --lhc --n-variants 10 --rank 0 --world-size 4 > /tmp/v28_1/gen_r0.log 2>&1 & P0=$!
set_slot 1 $PY scripts/generate_synthetic_guidance.py --generator v28_1 --lhc --n-variants 10 --rank 1 --world-size 4 > /tmp/v28_1/gen_r1.log 2>&1 & P1=$!
set_slot 2 $PY scripts/generate_synthetic_guidance.py --generator v28_1 --lhc --n-variants 10 --rank 2 --world-size 4 > /tmp/v28_1/gen_r2.log 2>&1 & P2=$!
set_slot 3 $PY scripts/generate_synthetic_guidance.py --generator v28_1 --lhc --n-variants 10 --rank 3 --world-size 4 > /tmp/v28_1/gen_r3.log 2>&1 & P3=$!
wait $P0 $P1 $P2 $P3
log "Generated: $(find $SYNTH_ROOT -name '*.nii.gz' 2>/dev/null | wc -l)/1650"

# ── Phase 2: Extract features (regional_hist_64 + hog3d_512 in parallel) ──────
log "=== Phase 2: Feature extraction ==="
mkdir -p "$DATA_DIR/regional_hist_64" "$DATA_DIR/hog3d_512"

# regional_hist_64 on slots 0-1 (CPU-only, 112 workers)
set_slot 0-1 $PY $ANALYSIS/extract_features_regional_hist.py \
  --mode synthetic --synth-root "$SYNTH_ROOT" \
  --output-csv "$DATA_DIR/regional_hist_64/synthetic_v28_1_guidance_lhc_features.csv" \
  --n-workers 112 > /tmp/v28_1/extract_rhist.log 2>&1 & P0=$!

# hog3d_512 on slots 2-3 (CPU-only, 112 workers)
set_slot 2-3 $PY $ANALYSIS/extract_features_hog3d.py \
  --mode synthetic --synth-root "$SYNTH_ROOT" \
  --output-csv "$DATA_DIR/hog3d_512/synthetic_v28_1_guidance_lhc_features.csv" \
  --n-workers 112 > /tmp/v28_1/extract_hog3d.log 2>&1 & P1=$!

wait $P0 $P1
log "Extraction done: rhist=$(wc -l < $DATA_DIR/regional_hist_64/synthetic_v28_1_guidance_lhc_features.csv) hog3d=$(wc -l < $DATA_DIR/hog3d_512/synthetic_v28_1_guidance_lhc_features.csv)"

# ── Phase 3: Normalize (sequential, fast) ─────────────────────────────────────
log "=== Phase 3: Normalize ==="
for ft in regional_hist_64 hog3d_512; do
  orig_csv="$DATA_ROOT/original/${ft}/on_harmony_features.csv"
  $PY $ANALYSIS/normalize_combined.py \
    --original_csv  "$orig_csv" \
    --synthetic_csv "$DATA_DIR/${ft}/synthetic_v28_1_guidance_lhc_features.csv" \
    --output_original  "$DATA_DIR/${ft}/on_harmony_features_normalized_combined_downsampled100.csv" \
    --output_synthetic "$DATA_DIR/${ft}/synthetic_v28_1_guidance_lhc_features_normalized_combined.csv" \
    --feature_config "$FEAT_CFG" >> /tmp/v28_1/pipeline.log 2>&1
done
log "Normalize done."

# ── Phase 4: Analysis (both feature types in parallel) ────────────────────────
log "=== Phase 4: Analysis ==="
set_slot 0 $PY $ANALYSIS/run_all_analysis.py \
  --mask-type regional_hist_64 --only v28_1_guidance_lhc_r1 \
  > /tmp/v28_1/analysis_rhist.log 2>&1 & P0=$!
set_slot 1 $PY $ANALYSIS/run_all_analysis.py \
  --mask-type hog3d_512 --only v28_1_guidance_lhc_r1 \
  > /tmp/v28_1/analysis_hog3d.log 2>&1 & P1=$!
wait $P0 $P1
log "=== ALL DONE ==="
