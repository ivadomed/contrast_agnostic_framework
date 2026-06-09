#!/usr/bin/env bash
# Regional histogram pipeline (64 bins × 7 SynthSeg macro-regions = 448 dims).
#
# Hardware layout (4 slots × 56 workers = 224 workers):
#   Step 0: Original — 4 ranks × slot 0-3, 56 workers each (224 total)
#   Step 1: Synthetic — each version on its own slot (0-3), 4 ranks × 14 workers
#   Step 2: Merge rank CSVs
#   Step 3: All 4 normalize+analyze in parallel, one per slot (0-3)

set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PY="$PROJ/.venv/bin/python"
SCRIPT="$PROJ/analysis/contrast_manifold/scripts/extract_features_regional_hist.py"
NORM="$PROJ/analysis/contrast_manifold/scripts/normalize_combined.py"
ANALYSIS="$PROJ/analysis/contrast_manifold/scripts/run_all_analysis.py"
DATA="$PROJ/analysis/contrast_manifold/outputs/data"
FEAT_CONFIG="$PROJ/analysis/contrast_manifold/config/feature_selection.yaml"
BIDS="$PROJ/data/ON-Harmony"
MASK_TYPE="regional_hist_64"

# ── 0. Original extraction — 4 ranks across 4 slots (224 workers total) ──────
ORIG_OUT="$DATA/original/${MASK_TYPE}/on_harmony_features.csv"
if [ ! -f "$ORIG_OUT" ]; then
    echo "=== [0] Extracting original regional histogram features (4 ranks × 56 workers) ==="
    mkdir -p "$(dirname "$ORIG_OUT")"
    ORIG_PIDS=()
    for rank in 0 1 2 3; do
        set_slot $rank "$PY" "$SCRIPT" \
            --mode original \
            --bids-root "$BIDS" \
            --deriv-root "$BIDS/derivatives" \
            --output-csv "$ORIG_OUT" \
            --n-bins 64 \
            --n-workers 56 \
            --rank $rank --world-size 4 \
            > "/tmp/reghist_orig_r${rank}.log" 2>&1 &
        ORIG_PIDS+=($!)
    done
    echo "  Waiting for original extraction … PIDs: ${ORIG_PIDS[*]}"
    for pid in "${ORIG_PIDS[@]}"; do
        wait "$pid" && echo "  PID $pid done" || echo "  PID $pid FAILED"
    done
    echo "  Original extraction done."
else
    NROWS=$( (tail -n +2 "$ORIG_OUT" | wc -l) 2>/dev/null || echo 0 )
    echo "=== [0] Original already exists ($NROWS rows), skipping ==="
fi

# ── 1. Synthetic extraction — 4 versions × 4 ranks, each version on own slot ─
declare -A SYNTH_DIRS=(
    ["v19_c"]="$BIDS/derivatives/synthetic_v19_c"
    ["v19_c_lhc"]="$BIDS/derivatives/synthetic_v19_c_lhc"
    ["v22_1_lhc"]="$BIDS/derivatives/synthetic_v22_1_lhc"
    ["v22_2_lhc"]="$BIDS/derivatives/synthetic_v22_2_lhc"
)
declare -A VER_SLOTS=(
    ["v19_c"]="0"
    ["v19_c_lhc"]="1"
    ["v22_1_lhc"]="2"
    ["v22_2_lhc"]="3"
)

echo "=== [1] Launching synthetic extractions in parallel (each version on its own slot) ==="
EXTRACT_PIDS=()
for VER in v19_c v19_c_lhc v22_1_lhc v22_2_lhc; do
    SYNTH_ROOT="${SYNTH_DIRS[$VER]}"
    SLOT="${VER_SLOTS[$VER]}"
    OUT_DIR="$DATA/synthetic_${VER}/${MASK_TYPE}"
    OUT_CSV="$OUT_DIR/synthetic_${VER}_features.csv"
    mkdir -p "$OUT_DIR"

    if [ -f "$OUT_CSV" ]; then
        NROWS=$( (tail -n +2 "$OUT_CSV" | wc -l) 2>/dev/null || echo 0 )
        echo "  $VER: already has $NROWS rows, skipping"
        continue
    fi

    echo "  $VER → slot $SLOT: launching 4 ranks × 14 workers …"
    for rank in 0 1 2 3; do
        set_slot $SLOT "$PY" "$SCRIPT" \
            --mode synthetic \
            --synth-root "$SYNTH_ROOT" \
            --output-csv "$OUT_CSV" \
            --n-bins 64 \
            --n-workers 14 \
            --rank $rank --world-size 4 \
            > "/tmp/reghist_${VER}_r${rank}.log" 2>&1 &
        EXTRACT_PIDS+=($!)
    done
done

echo "  Waiting for all synthetic extractions … PIDs: ${EXTRACT_PIDS[*]:-none}"
for pid in "${EXTRACT_PIDS[@]:-}"; do
    wait "$pid" && echo "  PID $pid done" || echo "  PID $pid failed"
done
echo "=== Extraction complete ==="

# ── 2. Merge rank CSVs ────────────────────────────────────────────────────────
echo "=== [2] Merging rank CSVs ==="
for VER in v19_c v19_c_lhc v22_1_lhc v22_2_lhc; do
    OUT_DIR="$DATA/synthetic_${VER}/${MASK_TYPE}"
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

# ── 3. normalize_combined + run_all_analysis — all 4 versions in parallel ────
echo "=== [3] Normalize + analyze (4 versions in parallel, one per slot) ==="

_normalize_and_analyze() {
    local VER="$1"
    local RUN_NAME="$2"
    local SLOT="$3"

    local SYNTH_OUT="$DATA/synthetic_${VER}/${MASK_TYPE}"
    local SYNTH_CSV="$SYNTH_OUT/synthetic_${VER}_features.csv"
    local NORM_OUT="$SYNTH_OUT/synthetic_${VER}_features_normalized_combined.csv"
    local ORIG_NORM="$SYNTH_OUT/on_harmony_features_normalized_combined_downsampled100.csv"

    echo "  [$VER] normalize_combined (slot $SLOT) …"
    set_slot $SLOT "$PY" "$NORM" \
        --original_csv "$DATA/original/${MASK_TYPE}/on_harmony_features.csv" \
        --synthetic_csv "$SYNTH_CSV" \
        --output_original "$ORIG_NORM" \
        --output_synthetic "$NORM_OUT" \
        --feature_config "$FEAT_CONFIG" \
        >> "/tmp/reghist_norm_${VER}.log" 2>&1

    # Populate global original CSV used by v19_c_r1 (no per-version override)
    local GLOBAL_ORIG="$DATA/original/${MASK_TYPE}/on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
    if [ ! -f "$GLOBAL_ORIG" ] && [ "$VER" = "v19_c" ]; then
        cp "${ORIG_NORM%.csv}_feat_selected.csv" "$GLOBAL_ORIG" 2>/dev/null || true
    fi

    echo "  [$VER] run_all_analysis (slot $SLOT) …"
    set_slot $SLOT "$PY" "$ANALYSIS" \
        --only "$RUN_NAME" \
        --mask-type "${MASK_TYPE}" \
        >> "/tmp/reghist_analysis_${VER}.log" 2>&1

    echo "  [$VER] done."
}

_normalize_and_analyze v19_c     v19_c_r1     0 &
_normalize_and_analyze v19_c_lhc v19_c_lhc_r1 1 &
_normalize_and_analyze v22_1_lhc v22_1_lhc_r1 2 &
_normalize_and_analyze v22_2_lhc v22_2_lhc_r1 3 &
wait

echo "=== Regional histogram pipeline complete ==="
