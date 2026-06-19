#!/usr/bin/env bash
# Pipeline for v26_8 (global inversion), v26_9 (gamma), v26_10 (fractal noise)
# generate → extract regional_hist_64 → normalize → analyze
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
CM_ROOT="${REPO_ROOT}/datasets/on-harmony/7_analysis_on-harmony/contrast_manifold"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"

PY="${REPO_ROOT}/.venv/bin/python"
ANALYSIS="${CM_ROOT}/scripts"
DATA_ROOT="${CM_ROOT}/outputs/data"
ORIG_CSV="$DATA_ROOT/original/regional_hist_64/on_harmony_features.csv"
FEAT_CFG="${CM_ROOT}/config/feature_selection.yaml"

mkdir -p /tmp/v268910
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v268910/pipeline.log; }

# ── PHASE 1: Generate (parallel 4 GPUs per version, sequential per version) ────
for ver in v26_8 v26_9 v26_10; do
  log "=== PHASE 1: $ver generation ==="
  PIDS=()
  for rank in 0 1 2 3; do
      run_job --name "gen_${ver}_r${rank}" --gpus 1 --slot $rank --wait \
          --log "/tmp/v268910/gen_${ver}_r${rank}.log" -- \
          "$PY" "${REPO_ROOT}/scripts/generate_synthetic_guidance.py" \
          --generator "$ver" --lhc --n-variants 10 \
          --rank $rank --world-size 4 &
      PIDS+=($!)
  done
  wait "${PIDS[@]}"
  log "$ver done: $(find "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_${ver}_guidance_lhc" -name '*.nii.gz' 2>/dev/null | wc -l)/1650"
done

# ── PHASE 2: Feature extraction (3 versions in parallel) ────────────────────────
log "=== PHASE 2: Feature extraction ==="
for ver in v26_8 v26_9 v26_10; do
  mkdir -p "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64"
done

PIDS=()
for ver in v26_8 v26_9 v26_10; do
    run_job --gpus 0 --slot 0 --wait \
        --log "/tmp/v268910/extract_${ver}.log" -- \
        "$PY" "$ANALYSIS/extract_features_regional_hist.py" \
        --mode synthetic \
        --synth-root "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_${ver}_guidance_lhc" \
        --output-csv "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features.csv" \
        --n-workers 14 &
    PIDS+=($!)
done
wait "${PIDS[@]}"
log "Phase 2 done."

# ── PHASE 3: Normalize ──────────────────────────────────────────────────────────
log "=== PHASE 3: Normalize ==="
for ver in v26_8 v26_9 v26_10; do
  run_job --gpus 0 --slot 0 --wait \
    --log "/tmp/v268910/norm_${ver}.log" -- \
    "$PY" "$ANALYSIS/normalize_combined.py" \
    --original_csv "$ORIG_CSV" \
    --synthetic_csv "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features.csv" \
    --output_original  "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
    --output_synthetic "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features_normalized_combined.csv" \
    --feature_config "$FEAT_CFG"
done
log "Phase 3 done."

# ── PHASE 4: Analysis (3 parallel) ─────────────────────────────────────────────
log "=== PHASE 4: Analysis ==="
PIDS=()
for ver in v26_8 v26_9 v26_10; do
    run_job --gpus 0 --slot 0 --wait \
        --log "/tmp/v268910/analysis_${ver}.log" -- \
        "$PY" "$ANALYSIS/run_all_analysis.py" \
        --mask-type regional_hist_64 --only "${ver}_guidance_lhc_r1" &
    PIDS+=($!)
done
wait "${PIDS[@]}"
log "Phase 4 done."

log "=== ALL DONE ==="
