#!/usr/bin/env bash
# Pipeline for v28_2, v28_3, v28_4 — sequential generation, parallel extraction + analysis.
# v28_2: true resolution diversity (--resolution-diversity)
# v28_3: susceptibility signal dropout (target generator)
# v28_4: v28_2 + v28_3 combined
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project

PY=".venv/bin/python"
ANALYSIS="analysis/contrast_manifold/scripts"
DATA_ROOT="analysis/contrast_manifold/outputs/data"
FEAT_CFG="analysis/contrast_manifold/config/feature_selection.yaml"

mkdir -p /tmp/v28234
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v28234/pipeline.log; }

# ── Phase 1: Generate all three (4 GPUs each, sequential per version) ──────────
for ver in v28_2 v28_3 v28_4; do
  log "=== Generate $ver ==="
  extra=""
  if [ "$ver" = "v28_2" ] || [ "$ver" = "v28_4" ]; then
    extra="--resolution-diversity"
  fi
  set_slot 0 $PY scripts/generate_synthetic_guidance.py --generator $ver --lhc --n-variants 10 --rank 0 --world-size 4 $extra > /tmp/v28234/gen_${ver}_r0.log 2>&1 & P0=$!
  set_slot 1 $PY scripts/generate_synthetic_guidance.py --generator $ver --lhc --n-variants 10 --rank 1 --world-size 4 $extra > /tmp/v28234/gen_${ver}_r1.log 2>&1 & P1=$!
  set_slot 2 $PY scripts/generate_synthetic_guidance.py --generator $ver --lhc --n-variants 10 --rank 2 --world-size 4 $extra > /tmp/v28234/gen_${ver}_r2.log 2>&1 & P2=$!
  set_slot 3 $PY scripts/generate_synthetic_guidance.py --generator $ver --lhc --n-variants 10 --rank 3 --world-size 4 $extra > /tmp/v28234/gen_${ver}_r3.log 2>&1 & P3=$!
  wait $P0 $P1 $P2 $P3
  log "$ver: $(find data/ON-Harmony/derivatives/synthetic_${ver}_guidance_lhc -name '*.nii.gz' 2>/dev/null | wc -l)/1650"
done

# ── Phase 2: Extract features (3 versions × 2 feature types — use all 4 slots) ─
log "=== Phase 2: Feature extraction (regional_hist_64 + hog3d_512 for all 3) ==="
for ver in v28_2 v28_3 v28_4; do
  mkdir -p "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64" \
            "$DATA_ROOT/synthetic_${ver}_guidance_lhc/hog3d_512"
done

# 3 versions × 2 feature types = 6 jobs; batch on 4 slots (first 4, then 2)
set_slot 0 $PY $ANALYSIS/extract_features_regional_hist.py --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v28_2_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v28_2_guidance_lhc/regional_hist_64/synthetic_v28_2_guidance_lhc_features.csv" \
  --n-workers 56 > /tmp/v28234/extract_v28_2_rh.log 2>&1 & P0=$!

set_slot 1 $PY $ANALYSIS/extract_features_regional_hist.py --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v28_3_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v28_3_guidance_lhc/regional_hist_64/synthetic_v28_3_guidance_lhc_features.csv" \
  --n-workers 56 > /tmp/v28234/extract_v28_3_rh.log 2>&1 & P1=$!

set_slot 2 $PY $ANALYSIS/extract_features_regional_hist.py --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v28_4_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v28_4_guidance_lhc/regional_hist_64/synthetic_v28_4_guidance_lhc_features.csv" \
  --n-workers 56 > /tmp/v28234/extract_v28_4_rh.log 2>&1 & P2=$!

# hog3d for v28_2 on slot 3 while regional_hist runs on 0-2
set_slot 3 $PY $ANALYSIS/extract_features_hog3d.py --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v28_2_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v28_2_guidance_lhc/hog3d_512/synthetic_v28_2_guidance_lhc_features.csv" \
  --n-workers 56 > /tmp/v28234/extract_v28_2_hog.log 2>&1 & P3=$!

wait $P0 $P1 $P2 $P3

# hog3d for v28_3 and v28_4
set_slot 0 $PY $ANALYSIS/extract_features_hog3d.py --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v28_3_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v28_3_guidance_lhc/hog3d_512/synthetic_v28_3_guidance_lhc_features.csv" \
  --n-workers 112 > /tmp/v28234/extract_v28_3_hog.log 2>&1 & P0=$!

set_slot 2 $PY $ANALYSIS/extract_features_hog3d.py --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v28_4_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v28_4_guidance_lhc/hog3d_512/synthetic_v28_4_guidance_lhc_features.csv" \
  --n-workers 112 > /tmp/v28234/extract_v28_4_hog.log 2>&1 & P2=$!

wait $P0 $P2
log "Phase 2 done."

# ── Phase 3: Normalize (6 jobs sequentially — fast) ──────────────────────────────
log "=== Phase 3: Normalize ==="
for ver in v28_2 v28_3 v28_4; do
  for ft in regional_hist_64 hog3d_512; do
    orig_csv="$DATA_ROOT/original/${ft}/on_harmony_features.csv"
    $PY $ANALYSIS/normalize_combined.py \
      --original_csv  "$orig_csv" \
      --synthetic_csv "$DATA_ROOT/synthetic_${ver}_guidance_lhc/${ft}/synthetic_${ver}_guidance_lhc_features.csv" \
      --output_original  "$DATA_ROOT/synthetic_${ver}_guidance_lhc/${ft}/on_harmony_features_normalized_combined_downsampled100.csv" \
      --output_synthetic "$DATA_ROOT/synthetic_${ver}_guidance_lhc/${ft}/synthetic_${ver}_guidance_lhc_features_normalized_combined.csv" \
      --feature_config "$FEAT_CFG" >> /tmp/v28234/pipeline.log 2>&1
  done
done
log "Phase 3 done."

# ── Phase 4: Analysis (6 jobs, 4 in parallel then 2) ─────────────────────────────
log "=== Phase 4: Analysis ==="
set_slot 0 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only v28_2_guidance_lhc_r1 > /tmp/v28234/analysis_v28_2_rh.log 2>&1 & P0=$!
set_slot 1 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only v28_3_guidance_lhc_r1 > /tmp/v28234/analysis_v28_3_rh.log 2>&1 & P1=$!
set_slot 2 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only v28_4_guidance_lhc_r1 > /tmp/v28234/analysis_v28_4_rh.log 2>&1 & P2=$!
set_slot 3 $PY $ANALYSIS/run_all_analysis.py --mask-type hog3d_512 --only v28_2_guidance_lhc_r1     > /tmp/v28234/analysis_v28_2_hog.log 2>&1 & P3=$!
wait $P0 $P1 $P2 $P3

set_slot 0 $PY $ANALYSIS/run_all_analysis.py --mask-type hog3d_512 --only v28_3_guidance_lhc_r1 > /tmp/v28234/analysis_v28_3_hog.log 2>&1 & P0=$!
set_slot 1 $PY $ANALYSIS/run_all_analysis.py --mask-type hog3d_512 --only v28_4_guidance_lhc_r1 > /tmp/v28234/analysis_v28_4_hog.log 2>&1 & P1=$!
wait $P0 $P1
log "=== ALL DONE ==="
