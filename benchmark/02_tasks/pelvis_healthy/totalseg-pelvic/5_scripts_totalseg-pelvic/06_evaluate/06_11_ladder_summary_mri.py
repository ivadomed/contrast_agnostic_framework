#!/usr/bin/env python3
"""
totalseg-pelvic MRI-trained causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py (same pattern as open-ms's 06_18).
Added 2026-10-03: every rung had been trained/evaluated (2026-09-16) but no ladder script existed.
All rung run dirs are FLAT under 02_metrics/totalseg_pelvic_model/mri/ (category prefix resolved
by the engine). The ablations/ output dir also holds an older 02_ablations_summary.md (aggregate
table, not this ladder).

OOD = the held-out modality (ct); in-domain = the training modality (mri).

Usage:  bash 06_12_run_ladders.sh   (both modalities, via run_job)
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/pelvis_healthy/totalseg-pelvic
sys.path.insert(0, str(DATASET_ROOT.parent.parent.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/mri"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "mri"
OOD_CONTRASTS = ["ct"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)", "mri_baseline_20260916_072453"),
    ("+kmeans", "+ K-means intensity clustering", "mri_baseline_kmeans_20260916_072453"),
    ("+label_remap", "+ label remap", "mri_baseline_kmeans_label_remap_20260916_072453"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "totalseg-pelvic_mri_baseline_kmeans_label_remap_voronoi_lblvor_20261007_171228"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "totalseg-pelvic_mri_v26_6_2_train050_val000_20261005_223058"),   # val000 (2026-10-06; was mri_v26_6_2_train050_val100_20260916_072453)
    ("+AugLab (OURS)", "+ full AugLab recipe on top", "mri_ours_20260916_072453"),
]

if __name__ == "__main__":
    run_ladder(task_name="totalseg-pelvic MRI", contrast_label="mri",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
