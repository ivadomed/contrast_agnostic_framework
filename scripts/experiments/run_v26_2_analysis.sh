#!/usr/bin/env bash
set -euo pipefail

PROJ="/home/ge.polymtl.ca/pahoa/mri_synthesis_project"
PY="$PROJ/.venv/bin/python"
SCRIPT="$PROJ/analysis/contrast_manifold/scripts/extract_features_regional_hist.py"
NORM="$PROJ/analysis/contrast_manifold/scripts/normalize_combined.py"
ANALYSIS="$PROJ/analysis/contrast_manifold/scripts/run_all_analysis.py"
DATA="$PROJ/analysis/contrast_manifold/outputs/data"
FEAT_CONFIG="$PROJ/analysis/contrast_manifold/config/feature_selection.yaml"
BIDS="$PROJ/data/ON-Harmony"
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
        set_slot 0 "$PY" "$SCRIPT" \
            --mode synthetic \
            --synth-root "$SYNTH_ROOT" \
            --output-csv "$OUT_CSV" \
            --n-bins 64 \
            --n-workers 14 \
            --rank $rank --world-size 4 \
            > "/tmp/reghist_${VER}_r${rank}.log" 2>&1 &
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

set_slot 0 "$PY" "$NORM" \
        --original_csv "$DATA/original/${MASK_TYPE}/on_harmony_features.csv" \
        --synthetic_csv "$SYNTH_CSV" \
        --output_original "$ORIG_NORM" \
        --output_synthetic "$NORM_OUT" \
        --feature_config "$FEAT_CONFIG" \
        > "/tmp/reghist_norm_${VER}.log" 2>&1


echo "  run_all_analysis …"
set_slot 0 "$PY" "$ANALYSIS" \
        --only "$RUN" \
        --mask-type "${MASK_TYPE}" \
        > "/tmp/reghist_analysis_${VER}.log" 2>&1

echo "Done"
