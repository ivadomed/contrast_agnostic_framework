#!/usr/bin/env bash
# PCA-dimensionality sensitivity sweep for the coverage metrics.
# k-NN metrics (PRDC) suffer distance concentration in higher dimensions; this checks
# whether the method ranking is stable across PCA cuts, so the operating point is chosen
# on principled grounds (not tuned to a result). Reuses the already-extracted synth CSV.
#
# Usage:  bash run_pca_sweep.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../../../../.." && pwd)"
source "${PROJECT_ROOT}/scripts/job_runner/run_job.sh"

PY="${PROJECT_ROOT}/.venv/bin/python"
OUT="${HERE}/../outputs"
REAL_CSV="${OUT}/real_regional_hist31.csv"
SYNTH_CSV="${OUT}/synth_regional_hist31.csv"
LOGDIR="${OUT}/logs"; mkdir -p "${LOGDIR}"

for V in 0.50 0.60 0.70 0.80 0.90; do
    tag="pca$(echo "$V" | tr -d '.')"
    run_job --name "hcov_${tag}" --gpus 0 --slot 0 --cpus 8 --mem 32G \
        --log "${LOGDIR}/compute_${tag}.log" --wait -- \
        "${PY}" "${HERE}/compute_coverage_metrics.py" \
            --real-csv "${REAL_CSV}" --synth-csv "${SYNTH_CSV}" \
            --output-dir "${OUT}/sweep/${tag}" --pca-variance "${V}" &
done
wait
echo "Sweep done → ${OUT}/sweep/"

# ── Aggregate: one row per (pca_cut × method) ────────────────────────────────
"${PY}" - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
import pandas as pd
out = Path(sys.argv[1]) / "sweep"
rows = []
for d in sorted(out.glob("pca*")):
    cfg = json.loads((d / "coverage_summary.json").read_text())["config"]
    df = pd.read_csv(d / "coverage_metrics.csv")
    for _, r in df.iterrows():
        rows.append({"pca_var": d.name, "pca_dims": cfg["pca_dims"], "method": r["method"],
                     "macro_coverage": round(r["macro_coverage"], 4), "vendi": round(r["vendi"], 2)})
agg = pd.DataFrame(rows)
agg.to_csv(out / "sweep_summary.csv", index=False)
print(agg.to_string(index=False))
PYEOF
