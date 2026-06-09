#!/usr/bin/env bash
# Full pipeline for v27a, v27a_bis, synthseg_modeA, synthseg_modeB_em.
# Each phase uses all 4 GPUs in parallel; phases run sequentially (dependency order).
# Logs: /tmp/v27pipeline/<phase>_rN.log
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project

# Expose venv nvidia CUDA libs so TensorFlow (SynthSeg) can use the GPUs.
# PyTorch ships them inside site-packages/nvidia/*/lib; TF needs them in LD_LIBRARY_PATH.
NVIDIA_LIBS=$(find .venv/lib/python3.12/site-packages/nvidia -name "*.so*" -path "*/lib/*" 2>/dev/null \
              | xargs -I{} dirname {} | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"

PY=".venv/bin/python"
ANALYSIS="analysis/contrast_manifold/scripts"
DATA_ROOT="analysis/contrast_manifold/outputs/data"
ORIG_CSV="$DATA_ROOT/original/regional_hist_64/on_harmony_features.csv"
FEAT_CFG="analysis/contrast_manifold/config/feature_selection.yaml"

mkdir -p /tmp/v27pipeline
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v27pipeline/pipeline.log; }

# ── PHASE 1: SynthSeg segs (needed for v27a, v27a_bis, synthseg_modeA) ────────
log "=== PHASE 1: SynthSeg predict (4 GPUs) ==="
set_slot 0 $PY scripts/run_synthseg_predict.py --rank 0 --world-size 4 > /tmp/v27pipeline/ss_predict_r0.log 2>&1 & P0=$!
set_slot 1 $PY scripts/run_synthseg_predict.py --rank 1 --world-size 4 > /tmp/v27pipeline/ss_predict_r1.log 2>&1 & P1=$!
set_slot 2 $PY scripts/run_synthseg_predict.py --rank 2 --world-size 4 > /tmp/v27pipeline/ss_predict_r2.log 2>&1 & P2=$!
set_slot 3 $PY scripts/run_synthseg_predict.py --rank 3 --world-size 4 > /tmp/v27pipeline/ss_predict_r3.log 2>&1 & P3=$!
wait $P0 $P1 $P2 $P3
log "Phase 1 done. Segs: $(find data/ON-Harmony/derivatives/synthseg_segs -name '*.nii.gz' 2>/dev/null | wc -l) files"

# ── PHASE 2: Our Mode A – v27a (label-conditioned quantile chunking) ──────────
log "=== PHASE 2: v27a generation (4 GPUs) ==="
set_slot 0 $PY scripts/generate_synthetic_guidance.py --generator v27a --lhc --n-variants 10 --seg-root data/ON-Harmony/derivatives/synthseg_segs --rank 0 --world-size 4 > /tmp/v27pipeline/v27a_r0.log 2>&1 & P0=$!
set_slot 1 $PY scripts/generate_synthetic_guidance.py --generator v27a --lhc --n-variants 10 --seg-root data/ON-Harmony/derivatives/synthseg_segs --rank 1 --world-size 4 > /tmp/v27pipeline/v27a_r1.log 2>&1 & P1=$!
set_slot 2 $PY scripts/generate_synthetic_guidance.py --generator v27a --lhc --n-variants 10 --seg-root data/ON-Harmony/derivatives/synthseg_segs --rank 2 --world-size 4 > /tmp/v27pipeline/v27a_r2.log 2>&1 & P2=$!
set_slot 3 $PY scripts/generate_synthetic_guidance.py --generator v27a --lhc --n-variants 10 --seg-root data/ON-Harmony/derivatives/synthseg_segs --rank 3 --world-size 4 > /tmp/v27pipeline/v27a_r3.log 2>&1 & P3=$!
wait $P0 $P1 $P2 $P3
log "Phase 2 done. v27a: $(find data/ON-Harmony/derivatives/synthetic_v27a_guidance_lhc -name '*.nii.gz' 2>/dev/null | wc -l)/1650 files"

# ── PHASE 3: Our Mode A bis – v27a_bis (global EM + per-label refinement) ─────
log "=== PHASE 3: v27a_bis generation (4 GPUs) ==="
set_slot 0 $PY scripts/generate_synthetic_guidance.py --generator v27a_bis --lhc --n-variants 10 --seg-root data/ON-Harmony/derivatives/synthseg_segs --rank 0 --world-size 4 > /tmp/v27pipeline/v27a_bis_r0.log 2>&1 & P0=$!
set_slot 1 $PY scripts/generate_synthetic_guidance.py --generator v27a_bis --lhc --n-variants 10 --seg-root data/ON-Harmony/derivatives/synthseg_segs --rank 1 --world-size 4 > /tmp/v27pipeline/v27a_bis_r1.log 2>&1 & P1=$!
set_slot 2 $PY scripts/generate_synthetic_guidance.py --generator v27a_bis --lhc --n-variants 10 --seg-root data/ON-Harmony/derivatives/synthseg_segs --rank 2 --world-size 4 > /tmp/v27pipeline/v27a_bis_r2.log 2>&1 & P2=$!
set_slot 3 $PY scripts/generate_synthetic_guidance.py --generator v27a_bis --lhc --n-variants 10 --seg-root data/ON-Harmony/derivatives/synthseg_segs --rank 3 --world-size 4 > /tmp/v27pipeline/v27a_bis_r3.log 2>&1 & P3=$!
wait $P0 $P1 $P2 $P3
log "Phase 3 done. v27a_bis: $(find data/ON-Harmony/derivatives/synthetic_v27a_bis_guidance_lhc -name '*.nii.gz' 2>/dev/null | wc -l)/1650 files"

# ── PHASE 4: SynthSeg Mode A (SynthSeg segs → BrainGenerator, uniform GMM) ───
log "=== PHASE 4: SynthSeg Mode A generation (4 GPUs) ==="
set_slot 0 $PY scripts/generate_synthseg_comparison.py --mode modeA --n-variants 10 --rank 0 --world-size 4 > /tmp/v27pipeline/ss_modeA_r0.log 2>&1 & P0=$!
set_slot 1 $PY scripts/generate_synthseg_comparison.py --mode modeA --n-variants 10 --rank 1 --world-size 4 > /tmp/v27pipeline/ss_modeA_r1.log 2>&1 & P1=$!
set_slot 2 $PY scripts/generate_synthseg_comparison.py --mode modeA --n-variants 10 --rank 2 --world-size 4 > /tmp/v27pipeline/ss_modeA_r2.log 2>&1 & P2=$!
set_slot 3 $PY scripts/generate_synthseg_comparison.py --mode modeA --n-variants 10 --rank 3 --world-size 4 > /tmp/v27pipeline/ss_modeA_r3.log 2>&1 & P3=$!
wait $P0 $P1 $P2 $P3
log "Phase 4 done. synthseg_modeA: $(find data/ON-Harmony/derivatives/synthseg_modeA -name '*.nii.gz' 2>/dev/null | wc -l)/1650 files"

# ── PHASE 5: SynthSeg Mode B (EM cluster labels → BrainGenerator) ─────────────
log "=== PHASE 5: SynthSeg Mode B (EM) generation (4 GPUs) ==="
set_slot 0 $PY scripts/generate_synthseg_comparison.py --mode modeB --n-variants 10 --rank 0 --world-size 4 > /tmp/v27pipeline/ss_modeB_r0.log 2>&1 & P0=$!
set_slot 1 $PY scripts/generate_synthseg_comparison.py --mode modeB --n-variants 10 --rank 1 --world-size 4 > /tmp/v27pipeline/ss_modeB_r1.log 2>&1 & P1=$!
set_slot 2 $PY scripts/generate_synthseg_comparison.py --mode modeB --n-variants 10 --rank 2 --world-size 4 > /tmp/v27pipeline/ss_modeB_r2.log 2>&1 & P2=$!
set_slot 3 $PY scripts/generate_synthseg_comparison.py --mode modeB --n-variants 10 --rank 3 --world-size 4 > /tmp/v27pipeline/ss_modeB_r3.log 2>&1 & P3=$!
wait $P0 $P1 $P2 $P3
log "Phase 5 done. synthseg_modeB_em: $(find data/ON-Harmony/derivatives/synthseg_modeB_em -name '*.nii.gz' 2>/dev/null | wc -l)/1650 files"

# ── PHASE 6: Feature extraction (4 versions × 1 slot each, 56 workers/slot) ──
log "=== PHASE 6: Feature extraction (4 parallel slots) ==="

mkdir -p \
  "$DATA_ROOT/synthetic_v27a_guidance_lhc/regional_hist_64" \
  "$DATA_ROOT/synthetic_v27a_bis_guidance_lhc/regional_hist_64" \
  "$DATA_ROOT/synthetic_synthseg_modeA/regional_hist_64" \
  "$DATA_ROOT/synthetic_synthseg_modeB_em/regional_hist_64"

set_slot 0 $PY $ANALYSIS/extract_features_regional_hist.py \
  --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v27a_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v27a_guidance_lhc/regional_hist_64/synthetic_v27a_guidance_lhc_features.csv" \
  --n-workers 56 \
  > /tmp/v27pipeline/extract_v27a.log 2>&1 & P0=$!

set_slot 1 $PY $ANALYSIS/extract_features_regional_hist.py \
  --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthetic_v27a_bis_guidance_lhc \
  --output-csv "$DATA_ROOT/synthetic_v27a_bis_guidance_lhc/regional_hist_64/synthetic_v27a_bis_guidance_lhc_features.csv" \
  --n-workers 56 \
  > /tmp/v27pipeline/extract_v27a_bis.log 2>&1 & P1=$!

set_slot 2 $PY $ANALYSIS/extract_features_regional_hist.py \
  --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthseg_modeA \
  --output-csv "$DATA_ROOT/synthetic_synthseg_modeA/regional_hist_64/synthetic_synthseg_modeA_features.csv" \
  --n-workers 56 \
  > /tmp/v27pipeline/extract_ss_modeA.log 2>&1 & P2=$!

set_slot 3 $PY $ANALYSIS/extract_features_regional_hist.py \
  --mode synthetic \
  --synth-root data/ON-Harmony/derivatives/synthseg_modeB_em \
  --output-csv "$DATA_ROOT/synthetic_synthseg_modeB_em/regional_hist_64/synthetic_synthseg_modeB_em_features.csv" \
  --n-workers 56 \
  > /tmp/v27pipeline/extract_ss_modeB.log 2>&1 & P3=$!

wait $P0 $P1 $P2 $P3
log "Phase 6 done."

# ── PHASE 7: Normalize + feature selection (4 parallel, CPU-only) ──────────────
log "=== PHASE 7: Normalize ==="

set_slot 0 $PY $ANALYSIS/normalize_combined.py \
  --original_csv  "$ORIG_CSV" \
  --synthetic_csv "$DATA_ROOT/synthetic_v27a_guidance_lhc/regional_hist_64/synthetic_v27a_guidance_lhc_features.csv" \
  --output_original  "$DATA_ROOT/synthetic_v27a_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
  --output_synthetic "$DATA_ROOT/synthetic_v27a_guidance_lhc/regional_hist_64/synthetic_v27a_guidance_lhc_features_normalized_combined.csv" \
  --feature_config "$FEAT_CFG" \
  > /tmp/v27pipeline/normalize_v27a.log 2>&1 & P0=$!

set_slot 1 $PY $ANALYSIS/normalize_combined.py \
  --original_csv  "$ORIG_CSV" \
  --synthetic_csv "$DATA_ROOT/synthetic_v27a_bis_guidance_lhc/regional_hist_64/synthetic_v27a_bis_guidance_lhc_features.csv" \
  --output_original  "$DATA_ROOT/synthetic_v27a_bis_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
  --output_synthetic "$DATA_ROOT/synthetic_v27a_bis_guidance_lhc/regional_hist_64/synthetic_v27a_bis_guidance_lhc_features_normalized_combined.csv" \
  --feature_config "$FEAT_CFG" \
  > /tmp/v27pipeline/normalize_v27a_bis.log 2>&1 & P1=$!

set_slot 2 $PY $ANALYSIS/normalize_combined.py \
  --original_csv  "$ORIG_CSV" \
  --synthetic_csv "$DATA_ROOT/synthetic_synthseg_modeA/regional_hist_64/synthetic_synthseg_modeA_features.csv" \
  --output_original  "$DATA_ROOT/synthetic_synthseg_modeA/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
  --output_synthetic "$DATA_ROOT/synthetic_synthseg_modeA/regional_hist_64/synthetic_synthseg_modeA_features_normalized_combined.csv" \
  --feature_config "$FEAT_CFG" \
  > /tmp/v27pipeline/normalize_ss_modeA.log 2>&1 & P2=$!

set_slot 3 $PY $ANALYSIS/normalize_combined.py \
  --original_csv  "$ORIG_CSV" \
  --synthetic_csv "$DATA_ROOT/synthetic_synthseg_modeB_em/regional_hist_64/synthetic_synthseg_modeB_em_features.csv" \
  --output_original  "$DATA_ROOT/synthetic_synthseg_modeB_em/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
  --output_synthetic "$DATA_ROOT/synthetic_synthseg_modeB_em/regional_hist_64/synthetic_synthseg_modeB_em_features_normalized_combined.csv" \
  --feature_config "$FEAT_CFG" \
  > /tmp/v27pipeline/normalize_ss_modeB.log 2>&1 & P3=$!

wait $P0 $P1 $P2 $P3
log "Phase 7 done."

# ── PHASE 8: Full manifold analysis (UMAP, PRDC, Vendi) ──────────────────────
log "=== PHASE 8: Analysis ==="

set_slot 0 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only v27a_guidance_lhc_r1     > /tmp/v27pipeline/analysis_v27a.log     2>&1 & P0=$!
set_slot 1 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only v27a_bis_guidance_lhc_r1 > /tmp/v27pipeline/analysis_v27a_bis.log 2>&1 & P1=$!
set_slot 2 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only synthseg_modeA_r1        > /tmp/v27pipeline/analysis_ss_modeA.log 2>&1 & P2=$!
set_slot 3 $PY $ANALYSIS/run_all_analysis.py --mask-type regional_hist_64 --only synthseg_modeB_em_r1     > /tmp/v27pipeline/analysis_ss_modeB.log 2>&1 & P3=$!
wait $P0 $P1 $P2 $P3
log "Phase 8 done."

log "=== ALL PHASES COMPLETE ==="
