#!/usr/bin/env bash
# Re-extract ALL hog3d_512 features with the native-resolution extractor,
# then normalize + analyze for v26_6.
#
# On Slurm, the original set_slot 0-3 (all 4 workers simultaneously) is mapped
# to a single CPU job with --cpus 32 and proportionally reduced --n-workers.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CM_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CM_ROOT}/../../../.." && pwd)"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"

PY="${REPO_ROOT}/.venv/bin/python"
ANALYSIS="${CM_ROOT}/scripts"
DATA_ROOT="${CM_ROOT}/outputs/data"
FEAT_CFG="${CM_ROOT}/config/feature_selection.yaml"
ORIG_DIR="$DATA_ROOT/original/hog3d_512"
SYNTH_DIR="$DATA_ROOT/synthetic_v26_6_guidance_lhc/hog3d_512"

mkdir -p /tmp/hog3d_native
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/hog3d_native/pipeline.log; }

# ── Phase 1: Re-extract originals (--cpus 32 → 28 workers) ───────────────────
log "=== Phase 1: original hog3d (native res, cpus=32, 28 workers) ==="
run_job --gpus 0 --cpus 32 --slot 0 --wait --log "/tmp/hog3d_native/orig.log" -- \
    "$PY" "$ANALYSIS/extract_features_hog3d.py" --mode original \
    --output-csv "$ORIG_DIR/on_harmony_features.csv" \
    --n-workers 28
log "Phase 1 done: $(wc -l < $ORIG_DIR/on_harmony_features.csv) rows"

# ── Phase 2: Re-extract v26_6 synthetic (--cpus 32 → 28 workers) ─────────────
log "=== Phase 2: v26_6 hog3d (native res) ==="
rm -f "$SYNTH_DIR/synthetic_v26_6_guidance_lhc_features.csv"
run_job --gpus 0 --cpus 32 --slot 0 --wait --log "/tmp/hog3d_native/synth_v26_6.log" -- \
    "$PY" "$ANALYSIS/extract_features_hog3d.py" --mode synthetic \
    --synth-root "${REPO_ROOT}/data/ON-Harmony/derivatives/synthetic_v26_6_guidance_lhc" \
    --output-csv "$SYNTH_DIR/synthetic_v26_6_guidance_lhc_features.csv" \
    --n-workers 28
log "Phase 2 done: $(wc -l < $SYNTH_DIR/synthetic_v26_6_guidance_lhc_features.csv) rows"

# ── Phase 3: Normalize ────────────────────────────────────────────────────────
log "=== Phase 3: Normalize ==="
run_job --gpus 0 --slot 0 --wait --log "/tmp/hog3d_native/normalize.log" -- \
    "$PY" "$ANALYSIS/normalize_combined.py" \
    --original_csv  "$ORIG_DIR/on_harmony_features.csv" \
    --synthetic_csv "$SYNTH_DIR/synthetic_v26_6_guidance_lhc_features.csv" \
    --output_original  "$SYNTH_DIR/on_harmony_features_normalized_combined_downsampled100.csv" \
    --output_synthetic "$SYNTH_DIR/synthetic_v26_6_guidance_lhc_features_normalized_combined.csv" \
    --feature_config   "$FEAT_CFG"
log "Phase 3 done."

# ── Phase 4: Analysis ─────────────────────────────────────────────────────────
log "=== Phase 4: Analysis ==="
run_job --gpus 0 --slot 0 --wait --log "/tmp/hog3d_native/analysis.log" -- \
    "$PY" "$ANALYSIS/run_all_analysis.py" \
    --mask-type hog3d_512 --only v26_6_guidance_lhc_r1
log "=== ALL DONE ==="
