#!/usr/bin/env bash
# Re-extract ALL hog3d_512 features with the native-resolution extractor,
# then normalize + analyze for v26_6.
#
# Uses set_slot 0-3 to access all 256 CPU workers in a single command —
# no rank splitting or CSV merging required.
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project

PY=".venv/bin/python"
ANALYSIS="analysis/contrast_manifold/scripts"
DATA_ROOT="analysis/contrast_manifold/outputs/data"
FEAT_CFG="analysis/contrast_manifold/config/feature_selection.yaml"
ORIG_DIR="$DATA_ROOT/original/hog3d_512"
SYNTH_DIR="$DATA_ROOT/synthetic_v26_6_guidance_lhc/hog3d_512"

mkdir -p /tmp/hog3d_native
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/hog3d_native/pipeline.log; }

# ── Phase 1: Re-extract originals (7803 scans, 224 workers ≈ 30s) ─────────────
log "=== Phase 1: original hog3d (native res, set_slot 0-3, 224 workers) ==="
# set_slot 0-3 gives access to all 4 slots (256 CPU workers, 4 GPUs) in one command.
set_slot 0-3 $PY $ANALYSIS/extract_features_hog3d.py --mode original \
  --output-csv "$ORIG_DIR/on_harmony_features.csv" \
  --n-workers 224 > /tmp/hog3d_native/orig.log 2>&1
log "Phase 1 done: $(wc -l < $ORIG_DIR/on_harmony_features.csv) rows"

# ── Phase 2: Re-extract v26_6 synthetic (1650 files, 224 workers ≈ 10s) ──────
log "=== Phase 2: v26_6 hog3d (native res) ==="
rm -f "$SYNTH_DIR/synthetic_v26_6_guidance_lhc_features.csv"
set_slot 0-3 $PY $ANALYSIS/extract_features_hog3d.py --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v26_6_guidance_lhc \
  --output-csv "$SYNTH_DIR/synthetic_v26_6_guidance_lhc_features.csv" \
  --n-workers 224 > /tmp/hog3d_native/synth_v26_6.log 2>&1
log "Phase 2 done: $(wc -l < $SYNTH_DIR/synthetic_v26_6_guidance_lhc_features.csv) rows"

# ── Phase 3: Normalize ────────────────────────────────────────────────────────
log "=== Phase 3: Normalize ==="
$PY $ANALYSIS/normalize_combined.py \
  --original_csv  "$ORIG_DIR/on_harmony_features.csv" \
  --synthetic_csv "$SYNTH_DIR/synthetic_v26_6_guidance_lhc_features.csv" \
  --output_original  "$SYNTH_DIR/on_harmony_features_normalized_combined_downsampled100.csv" \
  --output_synthetic "$SYNTH_DIR/synthetic_v26_6_guidance_lhc_features_normalized_combined.csv" \
  --feature_config   "$FEAT_CFG" >> /tmp/hog3d_native/pipeline.log 2>&1
log "Phase 3 done."

# ── Phase 4: Analysis ─────────────────────────────────────────────────────────
log "=== Phase 4: Analysis ==="
# Analysis is single-threaded CPU — one slot is enough.
set_slot 0 $PY $ANALYSIS/run_all_analysis.py \
  --mask-type hog3d_512 --only v26_6_guidance_lhc_r1 \
  > /tmp/hog3d_native/analysis.log 2>&1
log "=== ALL DONE ==="
