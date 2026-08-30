#!/usr/bin/env python3
"""
BraTS T2w causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py (same pattern as the T1n
sibling, 06_13_ladder_summary.py).

OOD = mean over held-out contrasts (t1n, t1c, t2f), excluding the training
contrast (t2w).

Usage:
  .venv/bin/python 06_14_ladder_summary_t2w.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/brats2024-glioma
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t2w"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t2w"
OOD_CONTRASTS = ["t1n", "t1c", "t2f"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     "brats2024-glioma_t2w_baseline_20260620_125115"),
    ("+kmeans", "+ K-means intensity clustering",
     "ablations/brats2024-glioma_t2w_baseline_kmeans_20260805_020550"),
    ("+label_remap", "+ label remap",
     "ablations/brats2024-glioma_t2w_baseline_kmeans_label_remap_20260805_020628"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "ablations/brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "brats2024-glioma_t2w_v26_6_2_train050_val100_20260620_125217"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     "brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val000_20260725_113540"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     "brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val100_20260725_113540"),
]

if __name__ == "__main__":
    run_ladder(task_name="BraTS T2w", contrast_label="t2w",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
