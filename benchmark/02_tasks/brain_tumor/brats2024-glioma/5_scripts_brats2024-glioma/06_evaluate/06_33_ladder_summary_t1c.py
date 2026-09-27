#!/usr/bin/env python3
"""
BraTS T1c (contrast-enhanced T1) causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py (same pattern as the T1n/T2w/T2f siblings).

OOD = mean over held-out contrasts (t1n, t2w, t2f), excluding the training contrast (t1c).
Layout as T1n/T2f: rung 5 (v26_6_2 alone) lives under ablations/.

Usage:
  .venv/bin/python 06_33_ladder_summary_t1c.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/brain_tumor/brats2024-glioma
sys.path.insert(0, str(DATASET_ROOT.parent.parent.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t1c"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t1c"
OOD_CONTRASTS = ["t1n", "t2w", "t2f"]
TS = "20260921_140000"   # 00_utils/t1c_runs.sh: T1C_TS

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     f"brats2024-glioma_t1c_baseline_{TS}"),
    ("+kmeans", "+ K-means intensity clustering",
     f"ablations/brats2024-glioma_t1c_baseline_kmeans_{TS}"),
    ("+label_remap", "+ label remap",
     f"ablations/brats2024-glioma_t1c_baseline_kmeans_label_remap_{TS}"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     f"ablations/brats2024-glioma_t1c_baseline_kmeans_label_remap_voronoi_{TS}"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     f"ablations/brats2024-glioma_t1c_v26_6_2_train050_val100_{TS}"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"brats2024-glioma_t1c_auglabAug_v26_6_2_train050_val000_{TS}"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     f"brats2024-glioma_t1c_auglabAug_v26_6_2_train050_val100_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="BraTS T1c", contrast_label="t1c",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
