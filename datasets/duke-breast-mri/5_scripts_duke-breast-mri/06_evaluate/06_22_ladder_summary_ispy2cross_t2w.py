#!/usr/bin/env python3
"""
duke-breast-mri cross-dataset (ISPY2 T2W-trained) causal-ablation ladder:
scores the ispy2 T2W-trained ladder rungs on duke-breast-mri's 291-case
external test set (cross-contrast: t2w-trained model tested on duke's t1wce
imaging). Sibling of 06_21_ladder_summary_ispy2cross_t1wce.py -- see that
file's header for the full rationale (duke's single t1wce item is set as
both in_domain and its own OOD entry, since the shared engine's plots are
now OOD-only and this dataset's real generalization test is cross-dataset,
not cross-contrast).

Usage:
  .venv/bin/python 06_22_ladder_summary_ispy2cross_t2w.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/duke-breast-mri
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_duke-breast-mri/02_metrics/ispy2_model/t2w"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t1wce"   # duke's only test item, scored against the t2w-trained model
OOD_CONTRASTS = ["t1wce"]   # duke's single item, doubling as its own OOD entry -- see docstring

TS = "20260905_163655"
RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     f"ispy2_t2w_baseline_{TS}"),
    ("+kmeans", "+ K-means intensity clustering",
     f"ablations/ispy2_t2w_baseline_kmeans_{TS}"),
    ("+label_remap", "+ label remap",
     f"ablations/ispy2_t2w_baseline_kmeans_label_remap_{TS}"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     f"ablations/ispy2_t2w_baseline_kmeans_label_remap_voronoi_{TS}"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     f"ablations/ispy2_t2w_v26_6_2_train050_val100_{TS}"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"ispy2_t2w_auglabAug_v26_6_2_train050_val000_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="duke-breast-mri cross-dataset T2W (ispy2-trained)", contrast_label="t1wce",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
