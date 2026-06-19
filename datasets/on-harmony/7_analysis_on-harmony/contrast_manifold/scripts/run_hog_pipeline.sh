#!/usr/bin/env bash
# HOG feature pipeline (3 planes × 324 = 972 dims per scan).
#
# Hardware layout (4 slots × 56 workers = 224 workers):
#   Step 0: Original — 4 ranks × slot 0-3, 56 workers each
#   Step 1: Synthetic — each version on its own slot, 4 ranks × 14 workers
#   Step 2: Merge rank CSVs
#   Step 3: All 4 normalize+analyze in parallel, one per slot

set -euo pipefail

CM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CM_ROOT}/../../../.." && pwd)"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"

PY="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${CM_ROOT}/scripts/extract_features_hog.py"
NORM="${CM_ROOT}/scripts/normalize_combined.py"
ANALYSIS="${CM_ROOT}/scripts/run_all_analysis.py"
DATA="${CM_ROOT}/outputs/data"
FEAT_CONFIG="${CM_ROOT}/config/feature_selection.yaml"
BIDS="${REPO_ROOT}/data/ON-Harmony"
MASK_TYPE="hog_972"

# ── 0. Original extraction — 4 ranks across 4 slots ──────────────────────────
ORIG_OUT="$DATA/original/${MASK_TYPE}/on_harmony_features.csv"
if [ ! -f "$ORIG_OUT" ]; then
    echo "=== [0] Extracting original HOG features (4 ranks × 56 workers) ==="
    mkdir -p "$(dirname "$ORIG_OUT")"
    ORIG_PIDS=()
    for rank in 0 1 2 3; do
        run_job --gpus 0 --slot $rank --wait \
            --log "/tmp/hog_orig_r${rank}.log" -- \
            "$PY" "$SCRIPT" \
            --mode original \
            --bids-root "$BIDS" \
            --deriv-root "$BIDS/derivatives" \
            --output-csv "$ORIG_OUT" \
            --n-workers 56 \
            --rank $rank --world-size 4 &
        ORIG_PIDS+=($!)
    done
    echo "  Waiting … PIDs: ${ORIG_PIDS[*]}"
    for pid in "${ORIG_PIDS[@]}"; do
        wait "$pid" && echo "  PID $pid done" || echo "  PID $pid FAILED"
    done
    # Merge rank CSVs into main
    "$PY" - <<PYEOF
import pandas as pd, pathlib
d = pathlib.Path("$(dirname "$ORIG_OUT")")
csvs = sorted(d.glob("on_harmony_features_rank*.csv"))
merged = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
merged.to_csv(d / "on_harmony_features.csv", index=False)
print(f"  Merged {len(csvs)} rank files → {len(merged)} rows")
for f in csvs: f.unlink()
PYEOF
    echo "  Original done."
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

echo "=== [1] Launching synthetic HOG extractions in parallel ==="
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
    echo "  $VER → slot $SLOT: 4 ranks × 14 workers"
    for rank in 0 1 2 3; do
        run_job --gpus 0 --slot $SLOT --wait \
            --log "/tmp/hog_${VER}_r${rank}.log" -- \
            "$PY" "$SCRIPT" \
            --mode synthetic \
            --synth-root "$SYNTH_ROOT" \
            --output-csv "$OUT_CSV" \
            --n-workers 14 \
            --rank $rank --world-size 4 &
        EXTRACT_PIDS+=($!)
    done
done

echo "  Waiting … PIDs: ${EXTRACT_PIDS[*]:-none}"
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
out = pathlib.Path("$OUT_DIR"); ver = "$VER"
csvs = sorted(out.glob(f"synthetic_{ver}_features_rank*.csv"))
if not csvs:
    single = out / f"synthetic_{ver}_features.csv"
    if single.exists():
        print(f"  {ver}: already merged ({len(pd.read_csv(single))} rows)")
        sys.exit(0)
    print(f"  {ver}: NO CSV FOUND"); sys.exit(1)
merged = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
merged.to_csv(out / f"synthetic_{ver}_features.csv", index=False)
print(f"  {ver}: merged {len(csvs)} ranks → {len(merged)} rows")
for f in csvs: f.unlink()
PYEOF
done

# ── 3. normalize_combined + run_all_analysis — all 4 in parallel ─────────────
echo "=== [3] Normalize + analyze (4 versions, one per slot) ==="

_normalize_and_analyze() {
    local VER="$1" RUN_NAME="$2" SLOT="$3"
    local SYNTH_OUT="$DATA/synthetic_${VER}/${MASK_TYPE}"
    local SYNTH_CSV="$SYNTH_OUT/synthetic_${VER}_features.csv"
    local NORM_OUT="$SYNTH_OUT/synthetic_${VER}_features_normalized_combined.csv"
    local ORIG_NORM="$SYNTH_OUT/on_harmony_features_normalized_combined_downsampled100.csv"
    local GLOBAL_ORIG="$DATA/original/${MASK_TYPE}/on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"

    run_job --gpus 0 --slot $SLOT --wait --log "/tmp/hog_norm_${VER}.log" -- \
        "$PY" "$NORM" \
        --original_csv "$DATA/original/${MASK_TYPE}/on_harmony_features.csv" \
        --synthetic_csv "$SYNTH_CSV" \
        --output_original "$ORIG_NORM" \
        --output_synthetic "$NORM_OUT" \
        --feature_config "$FEAT_CONFIG"

    if [ "$VER" = "v19_c" ] && [ ! -f "$GLOBAL_ORIG" ]; then
        cp "${ORIG_NORM%.csv}_feat_selected.csv" "$GLOBAL_ORIG" 2>/dev/null || true
    fi

    run_job --gpus 0 --slot $SLOT --wait --log "/tmp/hog_analysis_${VER}.log" -- \
        "$PY" "$ANALYSIS" \
        --only "$RUN_NAME" \
        --mask-type "${MASK_TYPE}"

    echo "  [$VER] done."
}

_normalize_and_analyze v19_c     v19_c_r1     0 &
_normalize_and_analyze v19_c_lhc v19_c_lhc_r1 1 &
_normalize_and_analyze v22_1_lhc v22_1_lhc_r1 2 &
_normalize_and_analyze v22_2_lhc v22_2_lhc_r1 3 &
wait

echo "=== HOG pipeline complete ==="
