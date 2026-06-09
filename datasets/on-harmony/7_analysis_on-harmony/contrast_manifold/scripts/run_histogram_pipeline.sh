#!/usr/bin/env bash
# Full histogram feature pipeline: extract → normalize → analyze for all 4 versions.
#
# Step 0: Extract histogram features for original ON-Harmony (56 workers).
# Step 1: Extract for all 4 synthetic datasets in parallel (14 workers each).
# Step 2: Merge rank CSVs, normalize_combined, run_all_analysis — per version.
#
# Uses all 64 CPUs. No GPU required.

set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PY="$PROJ/.venv/bin/python"
SCRIPT="$PROJ/analysis/contrast_manifold/scripts/extract_features_histogram.py"
NORM="$PROJ/analysis/contrast_manifold/scripts/normalize_combined.py"
ANALYSIS="$PROJ/analysis/contrast_manifold/scripts/run_all_analysis.py"
DATA="$PROJ/analysis/contrast_manifold/outputs/data"
FEAT_CONFIG="$PROJ/analysis/contrast_manifold/config/feature_selection.yaml"
BIDS="$PROJ/data/ON-Harmony"

# ── 0. Original extraction (56 workers, single process) ──────────────────────
ORIG_OUT="$DATA/original/histogram_256/on_harmony_features.csv"
if [ ! -f "$ORIG_OUT" ]; then
    echo "=== [0] Extracting original histogram features ==="
    mkdir -p "$(dirname "$ORIG_OUT")"
    set_slot 0 "$PY" "$SCRIPT" \
        --mode original \
        --bids-root "$BIDS" \
        --deriv-root "$BIDS/derivatives" \
        --output-csv "$ORIG_OUT" \
        --n-workers 56 \
        > /tmp/hist_orig.log 2>&1
    echo "  Original done."
else
    echo "=== [0] Original already exists, skipping ==="
fi

# ── 1. Synthetic extraction — all 4 versions, 4 parallel × 4 ranks each ──────
declare -A SYNTH_DIRS=(
    ["v19_c"]="$BIDS/derivatives/synthetic_v19_c"
    ["v19_c_lhc"]="$BIDS/derivatives/synthetic_v19_c_lhc"
    ["v22_1_lhc"]="$BIDS/derivatives/synthetic_v22_1_lhc"
    ["v22_2_lhc"]="$BIDS/derivatives/synthetic_v22_2_lhc"
)

echo "=== [1] Launching synthetic extractions in parallel ==="
EXTRACT_PIDS=()
for VER in v19_c v19_c_lhc v22_1_lhc v22_2_lhc; do
    SYNTH_ROOT="${SYNTH_DIRS[$VER]}"
    OUT_DIR="$DATA/synthetic_${VER}/histogram_256"
    OUT_CSV="$OUT_DIR/synthetic_${VER}_features.csv"
    mkdir -p "$OUT_DIR"

    if [ -f "$OUT_CSV" ]; then
        NROWS=$( (tail -n +2 "$OUT_CSV" | wc -l) 2>/dev/null || echo 0 )
        echo "  $VER: already has $NROWS rows, skipping extraction"
        continue
    fi

    echo "  $VER: launching 4-rank extraction …"
    for rank in 0 1 2 3; do
        set_slot 0 "$PY" "$SCRIPT" \
            --mode synthetic \
            --synth-root "$SYNTH_ROOT" \
            --output-csv "$OUT_CSV" \
            --n-workers 14 \
            --rank $rank --world-size 4 \
            > "/tmp/hist_${VER}_r${rank}.log" 2>&1 &
        EXTRACT_PIDS+=($!)
    done
done

echo "  Waiting for all extractions … PIDs: ${EXTRACT_PIDS[*]:-none}"
for pid in "${EXTRACT_PIDS[@]:-}"; do
    wait "$pid" && echo "  PID $pid done" || echo "  PID $pid failed"
done
echo "=== Extraction complete ==="

# ── 2. Merge rank CSVs ────────────────────────────────────────────────────────
echo "=== [2] Merging rank CSVs ==="
for VER in v19_c v19_c_lhc v22_1_lhc v22_2_lhc; do
    OUT_DIR="$DATA/synthetic_${VER}/histogram_256"
    OUT_CSV="$OUT_DIR/synthetic_${VER}_features.csv"
    "$PY" - <<PYEOF
import pandas as pd, pathlib, sys
out = pathlib.Path("$OUT_DIR")
ver = "$VER"
csvs = sorted(out.glob(f"synthetic_{ver}_features_rank*.csv"))
if not csvs:
    single = out / f"synthetic_{ver}_features.csv"
    if single.exists():
        print(f"  {ver}: single-rank CSV exists ({len(pd.read_csv(single))} rows)")
        sys.exit(0)
    print(f"  {ver}: NO CSV FOUND — extraction may have failed"); sys.exit(1)
merged = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
merged.to_csv(out / f"synthetic_{ver}_features.csv", index=False)
print(f"  {ver}: merged {len(csvs)} ranks → {len(merged)} rows")
for f in csvs:
    f.unlink()
PYEOF
done

# ── 3. normalize_combined + run_all_analysis — sequential per version ─────────
echo "=== [3] Normalize + analyze ==="

_normalize_and_analyze() {
    local VER="$1"
    local RUN_NAME="$2"
    local MAJOR="$3"

    local SYNTH_OUT="$DATA/synthetic_${VER}/histogram_256"
    local SYNTH_CSV="$SYNTH_OUT/synthetic_${VER}_features.csv"
    local NORM_OUT="$SYNTH_OUT/synthetic_${VER}_features_normalized_combined.csv"
    local ORIG_NORM="$SYNTH_OUT/on_harmony_features_normalized_combined_downsampled100.csv"

    echo "  [$VER] normalize_combined …"
    set_slot 0 "$PY" "$NORM" \
        --original_csv "$DATA/original/histogram_256/on_harmony_features.csv" \
        --synthetic_csv "$SYNTH_CSV" \
        --output_original "$ORIG_NORM" \
        --output_synthetic "$NORM_OUT" \
        --feature_config "$FEAT_CONFIG" \
        >> "/tmp/hist_norm_${VER}.log" 2>&1

    # Populate the global original CSV used by run_all_analysis for v19_c_r1
    local GLOBAL_ORIG="$DATA/original/histogram_256/on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
    if [ ! -f "$GLOBAL_ORIG" ]; then
        cp "${ORIG_NORM%.csv}_feat_selected.csv" "$GLOBAL_ORIG" 2>/dev/null || true
    fi

    echo "  [$VER] run_all_analysis …"
    set_slot 0 "$PY" "$ANALYSIS" \
        --only "$RUN_NAME" \
        --mask-type histogram_256 \
        >> "/tmp/hist_analysis_${VER}.log" 2>&1

    echo "  [$VER] done."
}

_normalize_and_analyze v19_c     v19_c_r1     v19
_normalize_and_analyze v19_c_lhc v19_c_lhc_r1 v19
_normalize_and_analyze v22_1_lhc v22_1_lhc_r1 v22
_normalize_and_analyze v22_2_lhc v22_2_lhc_r1 v22

echo "=== Histogram pipeline complete ==="
