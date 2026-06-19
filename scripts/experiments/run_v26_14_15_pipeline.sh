#!/usr/bin/env bash
# Pipeline: v26_14 (mixed flat+affine), v26_15 (double remap)
# v26_14 is already partially generated; will skip complete subjects.
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

mkdir -p /tmp/v261415
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v261415/pipeline.log; }

# v26_15 generation (new, ~2× cost of v26_6 due to double remap)
log "=== Generating v26_15 ==="
PIDS=()
for rank in 0 1 2 3; do
    run_job --name "gen_v26_15_r${rank}" --gpus 1 --slot $rank --wait \
        --log "/tmp/v261415/gen_v26_15_r${rank}.log" -- \
        "$PY" "${REPO_ROOT}/scripts/generate_synthetic_guidance.py" \
        --generator v26_15 --lhc --n-variants 10 \
        --rank $rank --world-size 4 &
    PIDS+=($!)
done
wait "${PIDS[@]}"
log "v26_15: $(find "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_v26_15_guidance_lhc" -name '*.nii.gz' 2>/dev/null | wc -l)/1650"

# Extract (v26_14 and v26_15 in parallel — v26_14 may have fewer files)
log "=== Feature extraction ==="
mkdir -p "$DATA_ROOT/synthetic_v26_14_guidance_lhc/regional_hist_64" \
          "$DATA_ROOT/synthetic_v26_15_guidance_lhc/regional_hist_64"

PIDS=()
run_job --gpus 0 --slot 0 --wait \
    --log "/tmp/v261415/extract_v26_14.log" -- \
    "$PY" "$ANALYSIS/extract_features_regional_hist.py" \
    --mode synthetic \
    --synth-root "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_v26_14_guidance_lhc" \
    --output-csv "$DATA_ROOT/synthetic_v26_14_guidance_lhc/regional_hist_64/synthetic_v26_14_guidance_lhc_features.csv" \
    --n-workers 28 &
PIDS+=($!)

run_job --gpus 0 --slot 0 --wait \
    --log "/tmp/v261415/extract_v26_15.log" -- \
    "$PY" "$ANALYSIS/extract_features_regional_hist.py" \
    --mode synthetic \
    --synth-root "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_v26_15_guidance_lhc" \
    --output-csv "$DATA_ROOT/synthetic_v26_15_guidance_lhc/regional_hist_64/synthetic_v26_15_guidance_lhc_features.csv" \
    --n-workers 28 &
PIDS+=($!)

wait "${PIDS[@]}"
log "Extraction done."

# Normalize (sequential)
log "=== Normalize ==="
for ver in v26_14 v26_15; do
  run_job --gpus 0 --slot 0 --wait \
    --log "/tmp/v261415/norm_${ver}.log" -- \
    "$PY" "$ANALYSIS/normalize_combined.py" \
    --original_csv "$ORIG_CSV" \
    --synthetic_csv "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features.csv" \
    --output_original  "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
    --output_synthetic "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features_normalized_combined.csv" \
    --feature_config "$FEAT_CFG"
done
log "Normalize done."

# Analysis (2 in parallel)
log "=== Analysis ==="
PIDS=()
for ver in v26_14 v26_15; do
    run_job --gpus 0 --slot 0 --wait \
        --log "/tmp/v261415/analysis_${ver}.log" -- \
        "$PY" "$ANALYSIS/run_all_analysis.py" \
        --mask-type regional_hist_64 --only "${ver}_guidance_lhc_r1" &
    PIDS+=($!)
done
wait "${PIDS[@]}"
log "=== ALL DONE ==="
