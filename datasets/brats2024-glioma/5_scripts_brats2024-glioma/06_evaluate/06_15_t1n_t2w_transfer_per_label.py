#!/usr/bin/env python3
"""
Per-label breakdown of the T1n<->T2w cross-contrast transfer asymmetry
(paper/compute_fillswap_per_contrast.py showed this at the whole-tumour /
pooled-OOD level; this asks whether the fill-swap step's effect -- and the
resulting transfer asymmetry -- is uniform across BraTS labels (NCR, SNFH,
ET, RC) or concentrated in specific ones).

Two things computed, both restricted to the single directly-relevant eval
contrast per direction (t2w when trained on t1n; t1n when trained on t2w),
per label:
  1. Absolute OOD Dice of the deployed real-fill (v26_6_2) rung in both
     directions -- the "does this label fail" view.
  2. The fill-swap step's own effect (rung 3 "+voronoi noise fill" -> rung 4
     "v26_6_2 real fill") per label, Holm-corrected across the 4-label family
     within each direction -- the "is the causal effect uniform" view.

Reuses load_case_means-style CSV loading and stat_tests.py's wilcoxon_p/holm
-- no new statistical machinery. Predictions for T1n's official ablation
rungs were already cleaned off disk (see FINDINGS below); this script reads
metrics (eval_all.csv) only, which still exist for both directions.

Usage:
  .venv/bin/python 06_15_t1n_t2w_transfer_per_label.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/brats2024-glioma
REPO_ROOT = DATASET_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT / "datasets/00_commun_scripts/00_00_utils"))
from stat_tests import holm, wilcoxon_p, fmt_p  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_brats2024-glioma/02_metrics/brats2024_glioma_model"
OUT_DIR = METRICS_ROOT / "ablations" / "t1n_t2w_transfer_per_label"

LABELS = ["NCR", "SNFH", "ET", "RC"]

# (direction label, real-fill metrics dir, voronoi(noise-fill) metrics dir, eval contrast)
DIRECTIONS = [
    ("T1n -> T2w (transfers well)",
     METRICS_ROOT / "t1n/ablations/nnUNet_brats2024-glioma_t1n_v26_6_2_train050_val100_20260730_200711",
     METRICS_ROOT / "t1n/ablations/auglab_brats2024-glioma_t1n_baseline_kmeans_label_remap_voronoi_20260730_200711",
     "t2w"),
    ("T2w -> T1n (fails)",
     METRICS_ROOT / "t2w/nnUNet_brats2024-glioma_t2w_v26_6_2_train050_val100_20260620_125217",
     METRICS_ROOT / "t2w/ablations/auglab_brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659",
     "t1n"),
]


def load_per_label(metrics_root: Path, contrast: str) -> dict:
    """label -> {case: mean dice over folds} for one eval contrast."""
    per = defaultdict(lambda: defaultdict(list))
    for fold_dir in sorted(metrics_root.glob("fold*")):
        csv_path = fold_dir / "eval_all.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                group = row.get("group") or row.get("contrast")
                if group != contrast:
                    continue
                try:
                    v = float(row["dice"])
                except (KeyError, ValueError):
                    continue
                if not np.isfinite(v):
                    continue
                per[row["label"]][row["case"]].append(v)
    return {lab: {c: float(np.mean(v)) for c, v in cases.items()} for lab, cases in per.items()}


lines = ["# T1n <-> T2w transfer asymmetry, per label", "",
         "Companion to the ladder tables (Tab. 3/9, main paper + supp) and "
         "`paper/compute_fillswap_per_contrast.py`. Restricted to the single directly-relevant "
         "eval contrast per direction (t2w when trained on t1n; t1n when trained on t2w). "
         "Kept out of the paper for now -- exploratory.", "",
         "## 1. Absolute OOD Dice (%), deployed real-fill rung",
         "", "| label | " + " | ".join(name for name, *_ in DIRECTIONS) + " | gap |",
         "|---|" + "---|" * (len(DIRECTIONS) + 1)]

abs_by_dir = []
for name, real_dir, voronoi_dir, contrast in DIRECTIONS:
    abs_by_dir.append(load_per_label(real_dir, contrast))

for lab in LABELS:
    vals = []
    for d in abs_by_dir:
        cases = d.get(lab, {})
        vals.append(float(np.mean(list(cases.values()))) * 100 if cases else float("nan"))
    gap = vals[0] - vals[1] if len(vals) == 2 else float("nan")
    lines.append(f"| {lab} | " + " | ".join(f"{v:.1f}" for v in vals) + f" | {gap:+.1f} |")

lines += ["", "## 2. Fill-swap step effect (+voronoi noise fill -> v26_6_2 real fill), per label",
          "", "Holm-corrected paired Wilcoxon, family = the 4 labels within each direction "
          "(same test family size/logic as `compute_fillswap_per_contrast.py`'s per-contrast test).",
          "", "| label | " + " | ".join(name for name, *_ in DIRECTIONS) + " |",
          "|---|" + "---|" * len(DIRECTIONS)]

per_dir_rows = []
for name, real_dir, voronoi_dir, contrast in DIRECTIONS:
    real = load_per_label(real_dir, contrast)
    noise = load_per_label(voronoi_dir, contrast)
    pvals, deltas = [], []
    for lab in LABELS:
        c_real = real.get(lab, {})
        c_noise = noise.get(lab, {})
        common = sorted(set(c_real) & set(c_noise))
        x = np.array([c_noise[k] for k in common])
        y = np.array([c_real[k] for k in common])
        p = wilcoxon_p(x, y) if len(x) else float("nan")
        d = float(np.mean(y) - np.mean(x)) * 100 if len(x) else float("nan")
        pvals.append(p)
        deltas.append(d)
    adj = holm(pvals)
    per_dir_rows.append(list(zip(deltas, adj)))

for i, lab in enumerate(LABELS):
    cells = []
    for row in per_dir_rows:
        d, p = row[i]
        star = "**" if (np.isfinite(p) and p < 0.05) else ""
        cells.append(f"{d:+.1f} (p={fmt_p(p)}){star}")
    lines.append(f"| {lab} | " + " | ".join(cells) + " |")

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "summary.md"
out_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n-> {out_path}")
