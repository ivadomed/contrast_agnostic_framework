#!/usr/bin/env python3
"""
BraTS T1n causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py (retrofitted 2026-08-03;
see that module's docstring). OOD = mean over held-out contrasts (t1c, t2w,
t2f), excluding the training contrast (t1n).

Usage:
  .venv/bin/python 06_13_ladder_summary.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/brats2024-glioma
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t1n"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t1n"
OOD_CONTRASTS = ["t1c", "t2w", "t2f"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     "brats2024-glioma_t1n_baseline_20260622_044535"),
    ("+kmeans", "+ K-means intensity clustering",
     "ablations/auglab_brats2024-glioma_t1n_baseline_kmeans_20260730_200711"),
    ("+label_remap", "+ label remap",
     "ablations/auglab_brats2024-glioma_t1n_baseline_kmeans_label_remap_20260730_200711"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "ablations/auglab_brats2024-glioma_t1n_baseline_kmeans_label_remap_voronoi_20260730_200711"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "ablations/nnUNet_brats2024-glioma_t1n_v26_6_2_train050_val100_20260730_200711"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     "auglab_brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val000_20260710_040303"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     "auglab_brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val100_20260725_113540"),
]

if __name__ == "__main__":
    run_ladder(task_name="BraTS T1n", contrast_label="t1n",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
