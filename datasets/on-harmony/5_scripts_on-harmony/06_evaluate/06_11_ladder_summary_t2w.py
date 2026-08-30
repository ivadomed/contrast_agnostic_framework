#!/usr/bin/env python3
"""
ON-Harmony T2w causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py (same pattern as the T1w
sibling, 06_10_ladder_summary.py). OOD = mean over held-out contrasts (T1w, bold,
dwi_ap, epi_ap, gre_echo1_mag), excluding the training contrast (T2w).

Usage:
  .venv/bin/python 06_11_ladder_summary_t2w.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/on-harmony
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_on-harmony/02_metrics/on_harmony_model/T2w"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "T2w"
OOD_CONTRASTS = ["T1w", "bold", "dwi_ap", "epi_ap", "gre_echo1_mag"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     "on-harmony_T2w_baseline_20260624_191152"),
    ("+kmeans", "+ K-means intensity clustering",
     "ablations/on-harmony_T2w_baseline_kmeans_20260804_195921"),
    ("+label_remap", "+ label remap",
     "ablations/on-harmony_T2w_baseline_kmeans_label_remap_20260804_195921"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "ablations/on-harmony_T2w_baseline_kmeans_label_remap_voronoi_20260804_195921"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "on-harmony_T2w_v26_6_2_train050_val100_20260625_154418"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     "on-harmony_T2w_auglabAug_v26_6_2_train050_val000_20260710_040443"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     "on-harmony_T2w_auglabAug_v26_6_2_train050_val100_20260727_075205"),
]

if __name__ == "__main__":
    run_ladder(task_name="ON-Harmony T2w", contrast_label="t2w",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
