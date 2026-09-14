#!/usr/bin/env python3
"""
duke-breast-mri cross-dataset (ISPY2 T2W-trained) causal-ablation ladder,
scored on the PRE-CONTRAST (no Gd) test contrast. Sibling of
06_23_ladder_summary_ispy2cross_precontrast_t1wce.py -- see that file's
header for the full rationale and rung-location convention.

Usage:
  .venv/bin/python 06_24_ladder_summary_ispy2cross_precontrast_t2w.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/duke-breast-mri
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_duke-breast-mri/02_metrics/ispy2_model/t2w"
ABLATIONS_ROOT = METRICS_ROOT / "ablations" / "precontrast"

IN_DOMAIN = "precontrast"
OOD_CONTRASTS = ["precontrast"]

TS = "20260905_163655"
RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     f"precontrast/ispy2_t2w_baseline_{TS}"),
    ("+kmeans", "+ K-means intensity clustering",
     f"ablations/precontrast/ispy2_t2w_baseline_kmeans_{TS}"),
    ("+label_remap", "+ label remap",
     f"ablations/precontrast/ispy2_t2w_baseline_kmeans_label_remap_{TS}"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     f"ablations/precontrast/ispy2_t2w_baseline_kmeans_label_remap_voronoi_{TS}"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     f"ablations/precontrast/ispy2_t2w_v26_6_2_train050_val100_{TS}"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"precontrast/ispy2_t2w_auglabAug_v26_6_2_train050_val000_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="duke-breast-mri cross-dataset PRE-CONTRAST (ispy2 T2W-trained)",
              contrast_label="precontrast",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
