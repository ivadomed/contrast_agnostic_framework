#!/usr/bin/env bash
# Full pipeline for v28_1: generate → extract regional_hist_64 + hog3d_512 → normalize → analyze
# v28_1 = V26_6 signed-alpha + Rician noise + aggressive resolution (zoom 0.20-1.0)
#
# On Slurm, set_slot 0-1/2-3 (multi-slot) is mapped to single CPU jobs with --cpus 32.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
CM_ROOT="${REPO_ROOT}/datasets/on-harmony/7_analysis_on-harmony/contrast_manifold"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"

PY="${REPO_ROOT}/.venv/bin/python"
ANALYSIS="${CM_ROOT}/scripts"
DATA_ROOT="${CM_ROOT}/outputs/data"
FEAT_CFG="${CM_ROOT}/config/feature_selection.yaml"
SYNTH_ROOT="${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_v28_1_guidance_lhc"
DATA_DIR="$DATA_ROOT/synthetic_v28_1_guidance_lhc"

mkdir -p /tmp/v28_1
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v28_1/pipeline.log; }

# ── Phase 1: Generate (4 GPUs in parallel) ───────────────────────────────────
log "=== Phase 1: Generate v28_1 (4 GPUs, 1650 files) ==="
PIDS=()
for rank in 0 1 2 3; do
    run_job --name "gen_v28_1_r${rank}" --gpus 1 --slot $rank --wait \
        --log "/tmp/v28_1/gen_r${rank}.log" -- \
        "$PY" "${REPO_ROOT}/scripts/generate_synthetic_guidance.py" \
        --generator v28_1 --lhc --n-variants 10 \
        --rank $rank --world-size 4 &
    PIDS+=($!)
done
wait "${PIDS[@]}"
log "Generated: $(find "$SYNTH_ROOT" -name '*.nii.gz' 2>/dev/null | wc -l)/1650"

# ── Phase 2: Extract features (regional_hist_64 + hog3d_512 in parallel) ──────
log "=== Phase 2: Feature extraction ==="
mkdir -p "$DATA_DIR/regional_hist_64" "$DATA_DIR/hog3d_512"

# regional_hist_64 and hog3d_512 in parallel (--cpus 32 each for adequate workers)
PIDS=()
run_job --gpus 0 --cpus 32 --slot 0 --wait \
    --log "/tmp/v28_1/extract_rhist.log" -- \
    "$PY" "$ANALYSIS/extract_features_regional_hist.py" \
    --mode synthetic --synth-root "$SYNTH_ROOT" \
    --output-csv "$DATA_DIR/regional_hist_64/synthetic_v28_1_guidance_lhc_features.csv" \
    --n-workers 28 &
PIDS+=($!)

run_job --gpus 0 --cpus 32 --slot 0 --wait \
    --log "/tmp/v28_1/extract_hog3d.log" -- \
    "$PY" "$ANALYSIS/extract_features_hog3d.py" \
    --mode synthetic --synth-root "$SYNTH_ROOT" \
    --output-csv "$DATA_DIR/hog3d_512/synthetic_v28_1_guidance_lhc_features.csv" \
    --n-workers 28 &
PIDS+=($!)

wait "${PIDS[@]}"
log "Extraction done: rhist=$(wc -l < $DATA_DIR/regional_hist_64/synthetic_v28_1_guidance_lhc_features.csv) hog3d=$(wc -l < $DATA_DIR/hog3d_512/synthetic_v28_1_guidance_lhc_features.csv)"

# ── Phase 3: Normalize (sequential, fast) ─────────────────────────────────────
log "=== Phase 3: Normalize ==="
for ft in regional_hist_64 hog3d_512; do
  run_job --gpus 0 --slot 0 --wait \
    --log "/tmp/v28_1/norm_${ft}.log" -- \
    "$PY" "$ANALYSIS/normalize_combined.py" \
    --original_csv  "$DATA_ROOT/original/${ft}/on_harmony_features.csv" \
    --synthetic_csv "$DATA_DIR/${ft}/synthetic_v28_1_guidance_lhc_features.csv" \
    --output_original  "$DATA_DIR/${ft}/on_harmony_features_normalized_combined_downsampled100.csv" \
    --output_synthetic "$DATA_DIR/${ft}/synthetic_v28_1_guidance_lhc_features_normalized_combined.csv" \
    --feature_config "$FEAT_CFG"
done
log "Normalize done."

# ── Phase 4: Analysis (both feature types in parallel) ────────────────────────
log "=== Phase 4: Analysis ==="
PIDS=()
run_job --gpus 0 --slot 0 --wait \
    --log "/tmp/v28_1/analysis_rhist.log" -- \
    "$PY" "$ANALYSIS/run_all_analysis.py" \
    --mask-type regional_hist_64 --only v28_1_guidance_lhc_r1 &
PIDS+=($!)
run_job --gpus 0 --slot 0 --wait \
    --log "/tmp/v28_1/analysis_hog3d.log" -- \
    "$PY" "$ANALYSIS/run_all_analysis.py" \
    --mask-type hog3d_512 --only v28_1_guidance_lhc_r1 &
PIDS+=($!)
wait "${PIDS[@]}"
log "=== ALL DONE ==="
