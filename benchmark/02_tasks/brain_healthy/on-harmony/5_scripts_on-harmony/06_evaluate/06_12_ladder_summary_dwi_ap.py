#!/usr/bin/env python3
"""
ON-Harmony dwi_ap causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py (same pattern as the T1w/T2w
siblings, 06_10/06_11). RUN TWICE, over two different OOD pools — see the appearance
caveat this arm was flagged with (project memory project_onharmony_dwi_ap_arm_20260921):
dwi_ap is EPI-adjacent to bold/epi_ap (|Spearman| 0.42/0.33 vs a 0.20 within-roster
median), so a rung-4->5 (texture) delta computed only over the full pool could be
inflated or deflated by that adjacency in a way the T1w/T2w ladders don't have to
worry about. A delta that flips sign between the two pools is NOT reportable (same
standard as the Atlas-Liver-HCC exclusion) — check before citing either run's numbers.

  FULL pool  (default): OOD = (T1w, T2w, bold, epi_ap, gre_echo1_mag) — matches the
             existing T1w/T2w ladders' "all held-out contrasts" convention.
  RESTRICTED pool (--restricted): OOD = (T1w, T2w, gre_echo1_mag) only — excludes the
             two EPI-family contrasts (bold, epi_ap) that sit closest to dwi_ap.

Usage:
  .venv/bin/python 06_12_ladder_summary_dwi_ap.py               # full pool -> ablations/
  .venv/bin/python 06_12_ladder_summary_dwi_ap.py --restricted   # restricted pool -> ablations_restricted/
"""
import argparse
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/brain_healthy/on-harmony
sys.path.insert(0, str(DATASET_ROOT.parent.parent.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_on-harmony/02_metrics/on_harmony_model/dwi_ap"

IN_DOMAIN = "dwi_ap"
OOD_FULL = ["T1w", "T2w", "bold", "epi_ap", "gre_echo1_mag"]
OOD_RESTRICTED = ["T1w", "T2w", "gre_echo1_mag"]

TS = "20260921_203727"
RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     f"on-harmony_dwi_ap_baseline_{TS}"),
    ("+kmeans", "+ K-means intensity clustering",
     f"ablations/on-harmony_dwi_ap_baseline_kmeans_{TS}"),
    ("+label_remap", "+ label remap",
     f"ablations/on-harmony_dwi_ap_baseline_kmeans_label_remap_{TS}"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     f"ablations/on-harmony_dwi_ap_baseline_kmeans_label_remap_voronoi_{TS}"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     f"on-harmony_dwi_ap_v26_6_2_train050_val100_{TS}"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"on-harmony_dwi_ap_auglabAug_v26_6_2_train050_val000_{TS}"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     f"on-harmony_dwi_ap_auglabAug_v26_6_2_train050_val100_{TS}"),
]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--restricted", action="store_true",
                    help="use the EPI-adjacency-excluded OOD pool instead of the full one")
    args = p.parse_args()

    if args.restricted:
        run_ladder(task_name="ON-Harmony dwi_ap (restricted OOD pool)", contrast_label="dwi_ap_restricted",
                   metrics_root=METRICS_ROOT, ablations_root=METRICS_ROOT / "ablations_restricted",
                   in_domain=IN_DOMAIN, ood_contrasts=OOD_RESTRICTED, rungs=RUNGS)
    else:
        run_ladder(task_name="ON-Harmony dwi_ap (full OOD pool)", contrast_label="dwi_ap",
                   metrics_root=METRICS_ROOT, ablations_root=METRICS_ROOT / "ablations",
                   in_domain=IN_DOMAIN, ood_contrasts=OOD_FULL, rungs=RUNGS)
