#!/usr/bin/env bash
# Post-generation pipeline for LHC synthetic datasets.
# Runs for a given VERSION (v19_c_lhc or v22_1_lhc):
#   1. Extract CURIA embeddings (3 parallel GPU workers)
#   2. Merge rank CSVs
#   3. Run normalize_combined
#   4. Run full analysis (coverage + UMAP + PCA + clustering)
#
# Usage:
#   set_slot 3 bash scripts/run_lhc_postprocess.sh v19_c_lhc
#   set_slot 3 bash scripts/run_lhc_postprocess.sh v22_1_lhc

set -euo pipefail

VERSION="${1:?Usage: $0 <version>  (e.g. v19_c_lhc or v22_1_lhc)}"

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$PROJ/.venv/bin/python"
SCRIPT_DIR="$PROJ/analysis/contrast_manifold/scripts"
DATA="$PROJ/analysis/contrast_manifold/outputs/data"
FEAT_CONFIG="$PROJ/analysis/contrast_manifold/config/feature_selection.yaml"

SYNTH_ROOT="$PROJ/data/ON-Harmony/derivatives/synthetic_${VERSION}"
OUT_DIR="$DATA/synthetic_${VERSION}/curia_embeddings"
mkdir -p "$OUT_DIR"

echo "=== [1/4] CURIA feature extraction for ${VERSION} ==="
for rank in 0 1 2; do
    "$PYTHON" "$SCRIPT_DIR/extract_features_curia.py" \
        --mode synthetic \
        --synth-root "$SYNTH_ROOT" \
        --output-csv "$OUT_DIR/synthetic_${VERSION}_features.csv" \
        --rank $rank --world-size 3 --gpu-id $rank \
        > "/tmp/curia_${VERSION}_rank${rank}.log" 2>&1 &
done
wait
echo "  CURIA extraction done for all ranks."

echo "=== [2/4] Merging rank CSVs ==="
"$PYTHON" - <<PYEOF
import pandas as pd, pathlib, sys
out = pathlib.Path("$OUT_DIR")
csvs = sorted(out.glob("synthetic_${VERSION}_features_rank*.csv"))
if not csvs:
    print("No rank CSVs found, checking for single output ...")
    single = out / "synthetic_${VERSION}_features.csv"
    if single.exists():
        print(f"Single output found: {single}")
        sys.exit(0)
    sys.exit(1)
merged = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
merged.to_csv(out / "synthetic_${VERSION}_features.csv", index=False)
print(f"Merged {len(csvs)} rank CSVs → {len(merged)} rows")
for f in csvs:
    f.unlink()
PYEOF

echo "=== [3/4] normalize_combined ==="
"$PYTHON" "$SCRIPT_DIR/normalize_combined.py" \
    --original_csv "$DATA/original/curia_embeddings/on_harmony_features.csv" \
    --synthetic_csv "$OUT_DIR/synthetic_${VERSION}_features.csv" \
    --output_original "$OUT_DIR/on_harmony_features_normalized_combined_downsampled100.csv" \
    --output_synthetic "$OUT_DIR/synthetic_${VERSION}_features_normalized_combined.csv" \
    --feature_config "$FEAT_CONFIG"

echo "=== [4/4] run_all_analysis --only ${VERSION/_lhc/_lhc}_r1 ==="
RUN_NAME="${VERSION}_r1"
"$PYTHON" "$SCRIPT_DIR/run_all_analysis.py" \
    --only "$RUN_NAME" \
    --mask-type curia_embeddings

echo "=== Done: ${VERSION} pipeline complete ==="
