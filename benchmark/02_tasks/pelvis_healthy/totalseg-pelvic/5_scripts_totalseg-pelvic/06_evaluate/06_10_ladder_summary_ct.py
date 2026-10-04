#!/usr/bin/env python3
"""
totalseg-pelvic CT-trained causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py (same pattern as open-ms's 06_18).
Added 2026-10-03: every rung had been trained/evaluated (2026-09-16) but no ladder script existed.
All rung run dirs are FLAT under 02_metrics/totalseg_pelvic_model/ct/ (category prefix resolved
by the engine). The ablations/ output dir also holds an older 02_ablations_summary.md (aggregate
table, not this ladder).

OOD = the held-out modality (mri); in-domain = the training modality (ct).

Usage:  bash 06_12_run_ladders.sh   (both modalities, via run_job)
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/pelvis_healthy/totalseg-pelvic
sys.path.insert(0, str(DATASET_ROOT.parent.parent.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/ct"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "ct"
OOD_CONTRASTS = ["mri"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)", "ct_baseline_20260916_072434"),
    ("+kmeans", "+ K-means intensity clustering", "ct_baseline_kmeans_20260916_072434"),
    ("+label_remap", "+ label remap", "ct_baseline_kmeans_label_remap_20260916_072434"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "ct_baseline_kmeans_label_remap_voronoi_20260916_072434"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "ct_v26_6_2_train050_val100_20260916_072434"),
    ("+AugLab (OURS)", "+ full AugLab recipe on top", "ct_ours_20260916_072434"),
]

if __name__ == "__main__":
    run_ladder(task_name="totalseg-pelvic CT", contrast_label="ct",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
