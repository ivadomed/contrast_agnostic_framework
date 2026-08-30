#!/usr/bin/env python3
"""
CHAOS T1in causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py (see that module's
docstring for why this used to be copy-pasted per dataset and no longer is).

Usage:
  .venv/bin/python 06_34_ladder_summary_t1in.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/chaos
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_chaos/02_metrics/chaos_model/t1in"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t1in"
OOD_CONTRASTS = ["t1out", "t2spir", "ct"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     "nnUNet_chaos_t1in_baseline_20260614_153230"),
    ("+kmeans", "+ K-means intensity clustering",
     "nnUNet_chaos_t1in_baseline_kmeans_train050_val000_20260715_081839"),
    ("+label_remap", "+ label remap",
     "nnUNet_chaos_t1in_baseline_kmeans_label_remap_train050_val000_20260715_081919"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "nnUNet_chaos_t1in_baseline_kmeans_label_remap_voronoi_train050_val000_20260715_081959"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "nnUNet_chaos_t1in_v26_6_2_train050_val000_20260715_082040"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     "auglab_chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     "auglab_chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420"),
]

if __name__ == "__main__":
    run_ladder(task_name="CHAOS T1in", contrast_label="t1in",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
