#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
CM_ROOT="${REPO_ROOT}/datasets/on-harmony/7_analysis_on-harmony/contrast_manifold"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"

PY="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${CM_ROOT}/scripts/extract_features_regional_hist.py"
NORM="${CM_ROOT}/scripts/normalize_combined.py"
ANALYSIS="${CM_ROOT}/scripts/run_all_analysis.py"
DATA="${CM_ROOT}/outputs/data"
FEAT_CONFIG="${CM_ROOT}/config/feature_selection.yaml"
BIDS="${REPO_ROOT}/data/ON-Harmony"
MASK_TYPE="regional_hist_64"

VER="v26_2_guidance_lhc"
RUN="v26_2_r1"

SYNTH_ROOT="$BIDS/derivatives/synthetic_$VER"
OUT_DIR="$DATA/synthetic_$VER/$MASK_TYPE"
OUT_CSV="$OUT_DIR/synthetic_${VER}_features.csv"

mkdir -p "$OUT_DIR"

if [ -f "$OUT_CSV" ]; then
    echo "Already exists"
else
    EXTRACT_PIDS=()
    echo "  $VER: launching 4 ranks × 14 workers …"
    for rank in 0 1 2 3; do
        run_job --gpus 0 --slot 0 --wait \
            --log "/tmp/reghist_${VER}_r${rank}.log" -- \
            "$PY" "$SCRIPT" \
            --mode synthetic \
            --synth-root "$SYNTH_ROOT" \
            --output-csv "$OUT_CSV" \
            --n-bins 64 \
            --n-workers 14 \
            --rank $rank --world-size 4 &
        EXTRACT_PIDS+=($!)
    done
    for pid in "${EXTRACT_PIDS[@]:-}"; do
        wait "$pid" && echo "  PID $pid done" || echo "  PID $pid failed"
    done

    "$PY" - <<PYEOF
import pandas as pd, pathlib, sys
out = pathlib.Path("$OUT_DIR")
ver = "$VER"
csvs = sorted(out.glob(f"synthetic_{ver}_features_rank*.csv"))
if csvs:
    merged = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    merged.to_csv(out / f"synthetic_{ver}_features.csv", index=False)
    for f in csvs:
        f.unlink()
PYEOF
fi

echo "  normalize_combined …"
SYNTH_CSV="$OUT_CSV"
NORM_OUT="$OUT_DIR/synthetic_${VER}_features_normalized_combined.csv"
ORIG_NORM="$OUT_DIR/on_harmony_features_normalized_combined_downsampled100.csv"

run_job --gpus 0 --slot 0 --wait --log "/tmp/reghist_norm_${VER}.log" -- \
    "$PY" "$NORM" \
    --original_csv "$DATA/original/${MASK_TYPE}/on_harmony_features.csv" \
    --synthetic_csv "$SYNTH_CSV" \
    --output_original "$ORIG_NORM" \
    --output_synthetic "$NORM_OUT" \
    --feature_config "$FEAT_CONFIG"

echo "  run_all_analysis …"
run_job --gpus 0 --slot 0 --wait --log "/tmp/reghist_analysis_${VER}.log" -- \
    "$PY" "$ANALYSIS" \
    --only "$RUN" \
    --mask-type "${MASK_TYPE}"

echo "Done"
