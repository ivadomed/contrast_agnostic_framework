#!/usr/bin/env bash
# Pipeline for v28_2, v28_3, v28_4 — sequential generation, parallel extraction + analysis.
# v28_2: true resolution diversity (--resolution-diversity)
# v28_3: susceptibility signal dropout (target generator)
# v28_4: v28_2 + v28_3 combined
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
CM_ROOT="${REPO_ROOT}/datasets/on-harmony/7_analysis_on-harmony/contrast_manifold"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"

PY="${REPO_ROOT}/.venv/bin/python"
ANALYSIS="${CM_ROOT}/scripts"
DATA_ROOT="${CM_ROOT}/outputs/data"
FEAT_CFG="${CM_ROOT}/config/feature_selection.yaml"

mkdir -p /tmp/v28234
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v28234/pipeline.log; }

# ── Phase 1: Generate all three (4 GPUs each, sequential per version) ──────────
for ver in v28_2 v28_3 v28_4; do
  log "=== Generate $ver ==="
  extra=""
  if [ "$ver" = "v28_2" ] || [ "$ver" = "v28_4" ]; then
    extra="--resolution-diversity"
  fi
  PIDS=()
  for rank in 0 1 2 3; do
      run_job --name "gen_${ver}_r${rank}" --gpus 1 --slot $rank --wait \
          --log "/tmp/v28234/gen_${ver}_r${rank}.log" -- \
          "$PY" "${REPO_ROOT}/scripts/generate_synthetic_guidance.py" \
          --generator "$ver" --lhc --n-variants 10 \
          --rank $rank --world-size 4 $extra &
      PIDS+=($!)
  done
  wait "${PIDS[@]}"
  log "$ver: $(find "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_${ver}_guidance_lhc" -name '*.nii.gz' 2>/dev/null | wc -l)/1650"
done

# ── Phase 2: Extract features (3 versions × 2 feature types — batch 4 then 2) ─
log "=== Phase 2: Feature extraction (regional_hist_64 + hog3d_512 for all 3) ==="
for ver in v28_2 v28_3 v28_4; do
  mkdir -p "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64" \
            "$DATA_ROOT/synthetic_${ver}_guidance_lhc/hog3d_512"
done

# First batch: regional_hist for v28_2/3/4 + hog3d for v28_2 (4 jobs)
PIDS=()
for ver in v28_2 v28_3 v28_4; do
    run_job --gpus 0 --slot 0 --wait \
        --log "/tmp/v28234/extract_${ver}_rh.log" -- \
        "$PY" "$ANALYSIS/extract_features_regional_hist.py" \
        --mode synthetic \
        --synth-root "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_${ver}_guidance_lhc" \
        --output-csv "$DATA_ROOT/synthetic_${ver}_guidance_lhc/regional_hist_64/synthetic_${ver}_guidance_lhc_features.csv" \
        --n-workers 56 &
    PIDS+=($!)
done
run_job --gpus 0 --slot 0 --wait \
    --log "/tmp/v28234/extract_v28_2_hog.log" -- \
    "$PY" "$ANALYSIS/extract_features_hog3d.py" \
    --mode synthetic \
    --synth-root "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_v28_2_guidance_lhc" \
    --output-csv "$DATA_ROOT/synthetic_v28_2_guidance_lhc/hog3d_512/synthetic_v28_2_guidance_lhc_features.csv" \
    --n-workers 56 &
PIDS+=($!)
wait "${PIDS[@]}"

# Second batch: hog3d for v28_3 and v28_4
PIDS=()
run_job --gpus 0 --slot 0 --wait \
    --log "/tmp/v28234/extract_v28_3_hog.log" -- \
    "$PY" "$ANALYSIS/extract_features_hog3d.py" \
    --mode synthetic \
    --synth-root "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_v28_3_guidance_lhc" \
    --output-csv "$DATA_ROOT/synthetic_v28_3_guidance_lhc/hog3d_512/synthetic_v28_3_guidance_lhc_features.csv" \
    --n-workers 56 &
PIDS+=($!)
run_job --gpus 0 --slot 0 --wait \
    --log "/tmp/v28234/extract_v28_4_hog.log" -- \
    "$PY" "$ANALYSIS/extract_features_hog3d.py" \
    --mode synthetic \
    --synth-root "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_v28_4_guidance_lhc" \
    --output-csv "$DATA_ROOT/synthetic_v28_4_guidance_lhc/hog3d_512/synthetic_v28_4_guidance_lhc_features.csv" \
    --n-workers 56 &
PIDS+=($!)
wait "${PIDS[@]}"
log "Phase 2 done."

# ── Phase 3: Normalize (6 jobs sequentially — fast) ──────────────────────────────
log "=== Phase 3: Normalize ==="
for ver in v28_2 v28_3 v28_4; do
  for ft in regional_hist_64 hog3d_512; do
    run_job --gpus 0 --slot 0 --wait \
      --log "/tmp/v28234/norm_${ver}_${ft}.log" -- \
      "$PY" "$ANALYSIS/normalize_combined.py" \
      --original_csv  "$DATA_ROOT/original/${ft}/on_harmony_features.csv" \
      --synthetic_csv "$DATA_ROOT/synthetic_${ver}_guidance_lhc/${ft}/synthetic_${ver}_guidance_lhc_features.csv" \
      --output_original  "$DATA_ROOT/synthetic_${ver}_guidance_lhc/${ft}/on_harmony_features_normalized_combined_downsampled100.csv" \
      --output_synthetic "$DATA_ROOT/synthetic_${ver}_guidance_lhc/${ft}/synthetic_${ver}_guidance_lhc_features_normalized_combined.csv" \
      --feature_config "$FEAT_CFG"
  done
done
log "Phase 3 done."

# ── Phase 4: Analysis (6 jobs, 4 in parallel then 2) ─────────────────────────────
log "=== Phase 4: Analysis ==="
PIDS=()
for ver in v28_2 v28_3 v28_4; do
    run_job --gpus 0 --slot 0 --wait \
        --log "/tmp/v28234/analysis_${ver}_rh.log" -- \
        "$PY" "$ANALYSIS/run_all_analysis.py" \
        --mask-type regional_hist_64 --only "${ver}_guidance_lhc_r1" &
    PIDS+=($!)
done
run_job --gpus 0 --slot 0 --wait \
    --log "/tmp/v28234/analysis_v28_2_hog.log" -- \
    "$PY" "$ANALYSIS/run_all_analysis.py" \
    --mask-type hog3d_512 --only v28_2_guidance_lhc_r1 &
PIDS+=($!)
wait "${PIDS[@]}"

PIDS=()
run_job --gpus 0 --slot 0 --wait \
    --log "/tmp/v28234/analysis_v28_3_hog.log" -- \
    "$PY" "$ANALYSIS/run_all_analysis.py" \
    --mask-type hog3d_512 --only v28_3_guidance_lhc_r1 &
PIDS+=($!)
run_job --gpus 0 --slot 0 --wait \
    --log "/tmp/v28234/analysis_v28_4_hog.log" -- \
    "$PY" "$ANALYSIS/run_all_analysis.py" \
    --mask-type hog3d_512 --only v28_4_guidance_lhc_r1 &
PIDS+=($!)
wait "${PIDS[@]}"
log "=== ALL DONE ==="
