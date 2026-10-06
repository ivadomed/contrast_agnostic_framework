#!/usr/bin/env python3
"""
BraTS T2f (FLAIR) causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py (same pattern as the T1n/T2w siblings,
06_13_ladder_summary.py / 06_14_ladder_summary_t2w.py).

OOD = mean over held-out contrasts (t1n, t1c, t2w), excluding the training contrast (t2f).
Layout difference vs T2w: here rung 5 (v26_6_2 alone) also lives under ablations/ (as in T1n).

Usage:
  .venv/bin/python 06_16_ladder_summary_t2f.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/brain_tumor/brats2024-glioma
sys.path.insert(0, str(DATASET_ROOT.parent.parent.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t2f"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t2f"
OOD_CONTRASTS = ["t1n", "t1c", "t2w"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     "brats2024-glioma_t2f_baseline_20260917_113412"),
    ("+kmeans", "+ K-means intensity clustering",
     "ablations/brats2024-glioma_t2f_baseline_kmeans_20260917_113412"),
    ("+label_remap", "+ label remap",
     "ablations/brats2024-glioma_t2f_baseline_kmeans_label_remap_20260917_113412"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "ablations/brats2024-glioma_t2f_baseline_kmeans_label_remap_voronoi_20260917_094019"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "ablations/brats2024-glioma_t2f_v26_6_2_train050_val000_20261005_222333"),   # val000 (real-image checkpoint selection, like rung 4) 2026-10-06; was ..._val100_20260917_113412
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     "brats2024-glioma_t2f_auglabAug_v26_6_2_train050_val000_20260917_113412"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     "brats2024-glioma_t2f_auglabAug_v26_6_2_train050_val100_20260917_113412"),
]

if __name__ == "__main__":
    run_ladder(task_name="BraTS T2f", contrast_label="t2f",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
