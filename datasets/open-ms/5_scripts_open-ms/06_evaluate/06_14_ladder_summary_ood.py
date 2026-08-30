#!/usr/bin/env python3
"""
Open-MS FLAIR causal-ablation ladder (OOD/HD95 version). The existing 06_12
script pre-dates the standardized rung set used by the other three datasets'
ladders -- it has no label-remap rung, no HD95, and no OOD/in-domain split, so
its numbers are not comparable to BraTS/CHAOS/ON-Harmony's and it was NOT the
source of this paper's tab:dissociation figures for Open-MS (those were
reconstructed by hand from per-contrast summary tables, then re-verified here).
06_12 is left in place for its own auglab-anchored/baseline-anchored breakdown,
which this script does not replace. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py.

Usage:
  .venv/bin/python 06_14_ladder_summary_ood.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/open-ms
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_open-ms/02_metrics/open_ms_model/flair"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "flair"
OOD_CONTRASTS = ["t1w", "t2w"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     "nnUNet_open-ms_flair_baseline_20260706_061243"),
    ("+kmeans", "+ K-means intensity clustering",
     "ablations/nnUNet_open-ms_flair_baseline_kmeans_train050_val000_20260711_065455"),
    ("+label_remap", "+ label remap",
     "ablations/nnUNet_open-ms_flair_baseline_kmeans_label_remap_train050_val000_20260711_062805"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "ablations/nnUNet_open-ms_flair_baseline_kmeans_label_remap_voronoi_train050_val000_20260711_062857"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "ablations/nnUNet_open-ms_flair_v26_6_2_train050_val000_20260711_062946"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     "auglab_open-ms_flair_auglabAug_v26_6_2_train050_val000_20260710_054455"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     "auglab_open-ms_flair_auglabAug_v26_6_2_train050_val100_20260716_095413"),
]

if __name__ == "__main__":
    run_ladder(task_name="Open-MS FLAIR", contrast_label="flair",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
