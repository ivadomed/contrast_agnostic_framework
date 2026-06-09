#!/usr/bin/env bash
# Extract regional_hist_13_64 / histogram_256 / hog_972 / hog3d_512
# for v27a, v27a_bis, synthseg_modeA, synthseg_modeB_em, then normalize + analyze.
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project

PY=".venv/bin/python"
ANALYSIS="analysis/contrast_manifold/scripts"
DATA_ROOT="analysis/contrast_manifold/outputs/data"
ORIG_CSV_BASE="$DATA_ROOT/original"
FEAT_CFG="analysis/contrast_manifold/config/feature_selection.yaml"
mkdir -p /tmp/v27extra

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v27extra/pipeline.log; }

# ── Version / synth-root / data-dir mapping ───────────────────────────────────
declare -A SYNTH_ROOT=(
  ["v27a"]="data/ON-Harmony/derivatives/synthetic_v27a_guidance_lhc"
  ["v27a_bis"]="data/ON-Harmony/derivatives/synthetic_v27a_bis_guidance_lhc"
  ["synthseg_modeA"]="data/ON-Harmony/derivatives/synthseg_modeA"
  ["synthseg_modeB_em"]="data/ON-Harmony/derivatives/synthseg_modeB_em"
)
declare -A DATA_NAME=(
  ["v27a"]="synthetic_v27a_guidance_lhc"
  ["v27a_bis"]="synthetic_v27a_bis_guidance_lhc"
  ["synthseg_modeA"]="synthetic_synthseg_modeA"
  ["synthseg_modeB_em"]="synthetic_synthseg_modeB_em"
)
declare -A FEAT_STEM=(
  ["v27a"]="synthetic_v27a_guidance_lhc"
  ["v27a_bis"]="synthetic_v27a_bis_guidance_lhc"
  ["synthseg_modeA"]="synthetic_synthseg_modeA"
  ["synthseg_modeB_em"]="synthetic_synthseg_modeB_em"
)

# ── Extractor script / feature-type name mapping ─────────────────────────────
declare -A EXTRACTOR=(
  ["regional_hist_13_64"]="$ANALYSIS/extract_features_regional_hist_13.py"
  ["histogram_256"]="$ANALYSIS/extract_features_histogram.py"
  ["hog_972"]="$ANALYSIS/extract_features_hog.py"
  ["hog3d_512"]="$ANALYSIS/extract_features_hog3d.py"
)
FEATURE_TYPES=(regional_hist_13_64 histogram_256 hog_972 hog3d_512)
VERSIONS=(v27a v27a_bis synthseg_modeA synthseg_modeB_em)

# ── PHASE 1: Extract (4 versions × 4 feature types = 16 jobs, batched 4 at a time) ──
log "=== PHASE 1: Feature extraction ==="

slot=0
pids=()
for ver in "${VERSIONS[@]}"; do
  for ft in "${FEATURE_TYPES[@]}"; do
    out_dir="$DATA_ROOT/${DATA_NAME[$ver]}/$ft"
    mkdir -p "$out_dir"
    out_csv="$out_dir/${FEAT_STEM[$ver]}_features.csv"
    if [ -f "$out_csv" ]; then
      log "  skip (exists): $ver / $ft"
      continue
    fi
    log "  extracting $ver / $ft on slot $slot"
    set_slot $slot $PY "${EXTRACTOR[$ft]}" \
      --mode synthetic \
      --synth-root "${SYNTH_ROOT[$ver]}" \
      --output-csv "$out_csv" \
      --n-workers 14 \
      > /tmp/v27extra/extract_${ver}_${ft}.log 2>&1 &
    pids+=($!)
    slot=$(( (slot + 1) % 4 ))
    # Wait for a full batch of 4 before launching more
    if [ ${#pids[@]} -ge 4 ]; then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
done
# Wait for any remaining
for pid in "${pids[@]}"; do wait "$pid"; done
log "Phase 1 done."

# ── PHASE 2: Normalize (16 jobs, batched 4 at a time) ─────────────────────────
log "=== PHASE 2: Normalize ==="

slot=0; pids=()
for ver in "${VERSIONS[@]}"; do
  for ft in "${FEATURE_TYPES[@]}"; do
    data_dir="$DATA_ROOT/${DATA_NAME[$ver]}/$ft"
    stem="${FEAT_STEM[$ver]}"
    orig_csv="$ORIG_CSV_BASE/$ft/on_harmony_features.csv"
    synth_csv="$data_dir/${stem}_features.csv"
    out_norm_orig="$data_dir/on_harmony_features_normalized_combined_downsampled100.csv"
    out_norm_synth="$data_dir/${stem}_features_normalized_combined.csv"
    feat_sel_synth="${out_norm_synth%.csv}_feat_selected.csv"
    if [ -f "$feat_sel_synth" ]; then
      log "  skip (exists): $ver / $ft"
      continue
    fi
    set_slot $slot $PY $ANALYSIS/normalize_combined.py \
      --original_csv  "$orig_csv" \
      --synthetic_csv "$synth_csv" \
      --output_original  "$out_norm_orig" \
      --output_synthetic "$out_norm_synth" \
      --feature_config "$FEAT_CFG" \
      > /tmp/v27extra/normalize_${ver}_${ft}.log 2>&1 &
    pids+=($!)
    slot=$(( (slot + 1) % 4 ))
    if [ ${#pids[@]} -ge 4 ]; then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done
log "Phase 2 done."

# ── PHASE 3: Analysis (4 versions × 4 feature types, batched 4 at a time) ────
log "=== PHASE 3: Analysis ==="

declare -A RUN_NAME=(
  ["v27a"]="v27a_guidance_lhc_r1"
  ["v27a_bis"]="v27a_bis_guidance_lhc_r1"
  ["synthseg_modeA"]="synthseg_modeA_r1"
  ["synthseg_modeB_em"]="synthseg_modeB_em_r1"
)

slot=0; pids=()
for ver in "${VERSIONS[@]}"; do
  for ft in "${FEATURE_TYPES[@]}"; do
    set_slot $slot $PY $ANALYSIS/run_all_analysis.py \
      --mask-type "$ft" \
      --only "${RUN_NAME[$ver]}" \
      > /tmp/v27extra/analysis_${ver}_${ft}.log 2>&1 &
    pids+=($!)
    slot=$(( (slot + 1) % 4 ))
    if [ ${#pids[@]} -ge 4 ]; then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done
log "Phase 3 done."

log "=== ALL DONE ==="
