#!/usr/bin/env python3
"""
CHAOS T2spir causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py (retrofitted 2026-08-03;
see that module's docstring). OOD = mean over held-out contrasts (t1in, t1out,
ct), excluding the training contrast (t2spir).

Usage:
  .venv/bin/python 06_33_ladder_summary_t2spir.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/chaos
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_chaos/02_metrics/chaos_model/t2spir"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t2spir"
OOD_CONTRASTS = ["t1in", "t1out", "ct"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     "chaos_t2spir_baseline_20260620_111146"),
    ("+kmeans", "+ K-means intensity clustering",
     "ablations/baseline_kmeans_train050_val000"),
    ("+label_remap", "+ label remap",
     "ablations/baseline_kmeans_label_remap_train050_val000"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "ablations/baseline_kmeans_label_remap_voronoi_train050_val000"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "chaos_t2spir_v26_6_2_train050_val100_20260620_112122"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     "chaos_t2spir_auglabAug_v26_6_2_train050_val000_20260710_054202"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     "chaos_t2spir_auglabAug_v26_6_2_train050_val100_20260723_194053"),
]

if __name__ == "__main__":
    run_ladder(task_name="CHAOS T2spir", contrast_label="t2spir",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
