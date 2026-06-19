#!/usr/bin/env bash
# Full pipeline for v26_5, v26_6, v26_7:
#   generate (10 variants/subject, 4 GPUs) → extract regional_hist_64 → normalize → analyze
# v26_5 and v26_6 are fast (~1s/vol); v26_7 is slower (~6s/vol due to flat assignments).
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

mkdir -p /tmp/v26567
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v26567/pipeline.log; }

_generate_4ranks() {
    local ver="$1" logdir="$2"
    local PIDS=()
    for rank in 0 1 2 3; do
        run_job --name "gen_${ver}_r${rank}" --gpus 1 --slot $rank --wait \
            --log "${logdir}/gen_${ver}_r${rank}.log" -- \
            "$PY" "${REPO_ROOT}/scripts/generate_synthetic_guidance.py" \
            --generator "$ver" --lhc --n-variants 10 \
            --rank $rank --world-size 4 &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"
}

# ── PHASE 1: Generate (sequential per version, parallel across 4 GPU slots) ─────
log "=== PHASE 1a: v26_5 generation ==="
_generate_4ranks v26_5 /tmp/v26567
log "v26_5 done: $(find "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_v26_5_guidance_lhc" -name '*.nii.gz' 2>/dev/null | wc -l)/1650"

log "=== PHASE 1b: v26_6 generation ==="
_generate_4ranks v26_6 /tmp/v26567
log "v26_6 done: $(find "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_v26_6_guidance_lhc" -name '*.nii.gz' 2>/dev/null | wc -l)/1650"

log "=== PHASE 1c: v26_7 generation ==="
_generate_4ranks v26_7 /tmp/v26567
log "v26_7 done: $(find "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_v26_7_guidance_lhc" -name '*.nii.gz' 2>/dev/null | wc -l)/1650"

# ── PHASE 2: Feature extraction (3 versions in parallel, regional_hist_64) ──────
log "=== PHASE 2: Feature extraction ==="
mkdir -p \
  "$DATA_ROOT/synthetic_v26_5_guidance_lhc/regional_hist_64" \
  "$DATA_ROOT/synthetic_v26_6_guidance_lhc/regional_hist_64" \
  "$DATA_ROOT/synthetic_v26_7_guidance_lhc/regional_hist_64"

PIDS=()
for ver in v26_5 v26_6 v26_7; do
    run_job --gpus 0 --slot 0 --wait \
        --log "/tmp/v26567/extract_${ver}.log" -- \
        "$PY" "$ANALYSIS/extract_features_regional_hist.py" \
        --mode synthetic \
        --synth-root "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_${ver}_guidance_lhc" \
        --output-csv "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features.csv" \
        --n-workers 14 &
    PIDS+=($!)
done
wait "${PIDS[@]}"
log "Phase 2 done."

# ── PHASE 3: Normalize (3 in parallel) ───────────────────────────────────────────
log "=== PHASE 3: Normalize ==="
PIDS=()
for ver in v26_5 v26_6 v26_7; do
    run_job --gpus 0 --slot 0 --wait \
        --log "/tmp/v26567/norm_${ver}.log" -- \
        "$PY" "$ANALYSIS/normalize_combined.py" \
        --original_csv  "$ORIG_CSV" \
        --synthetic_csv "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features.csv" \
        --output_original  "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
        --output_synthetic "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features_normalized_combined.csv" \
        --feature_config "$FEAT_CFG" &
    PIDS+=($!)
done
wait "${PIDS[@]}"
log "Phase 3 done."

# ── PHASE 4: Analysis (3 in parallel) ────────────────────────────────────────────
log "=== PHASE 4: Analysis ==="
PIDS=()
for ver in v26_5 v26_6 v26_7; do
    run_job --gpus 0 --slot 0 --wait \
        --log "/tmp/v26567/analysis_${ver}.log" -- \
        "$PY" "$ANALYSIS/run_all_analysis.py" \
        --mask-type regional_hist_64 --only "${ver}_guidance_lhc_r1" &
    PIDS+=($!)
done
wait "${PIDS[@]}"
log "Phase 4 done."

log "=== ALL DONE ==="
