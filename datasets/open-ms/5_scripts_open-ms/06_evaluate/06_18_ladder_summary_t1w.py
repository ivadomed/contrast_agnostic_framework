#!/usr/bin/env python3
"""
open-ms T1w causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py (same pattern as
BraTS/CHAOS/on-harmony's ladder scripts). NOTE: the historical 06_12_ladder_summary.py
(FLAIR-only) predates this shared engine and is intentionally left as-is; this is a
fresh sibling for the T1w training modality, not a retrofit of that file.

OOD = mean over held-out contrasts (flair, t2w), excluding the training contrast (t1w).

Usage:
  .venv/bin/python 06_18_ladder_summary_t1w.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/open-ms
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_open-ms/02_metrics/open_ms_model/t1w"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t1w"
OOD_CONTRASTS = ["flair", "t2w"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     "open-ms_t1w_baseline_20260708_083441"),
    ("+kmeans", "+ K-means intensity clustering",
     "ablations/open-ms_t1w_baseline_kmeans_train050_val000_20260805_020222"),
    ("+label_remap", "+ label remap",
     "ablations/open-ms_t1w_baseline_kmeans_label_remap_train050_val000_20260805_020311"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "ablations/open-ms_t1w_baseline_kmeans_label_remap_voronoi_train050_val000_20260805_020341"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "open-ms_t1w_v26_6_2_train050_val100_20260708_083641"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     "open-ms_t1w_auglabAug_v26_6_2_train050_val000_20260710_054525"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     "open-ms_t1w_auglabAug_v26_6_2_train050_val100_20260723_194125"),
]

if __name__ == "__main__":
    run_ladder(task_name="open-ms T1w", contrast_label="t1w",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
