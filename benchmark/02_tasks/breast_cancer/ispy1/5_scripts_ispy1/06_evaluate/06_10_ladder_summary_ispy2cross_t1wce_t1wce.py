#!/usr/bin/env python3
"""
ispy1 cross-dataset (ISPY2 T1WCE-trained) causal-ablation ladder on the I-SPY1
`t1wce` test item (167 MAMA-MIA-expert-masked cases, natively unilateral -- see
02_nnunet/02_01_convert_test.py). Mirrors duke-breast-mri's 06_1X ladder scripts.
SAME-contrast / cross-DATASET arm (t1wce-trained model on I-SPY1 t1wce) -- NOT held-out-contrast evidence; supplementary only, never pooled into an OOD bucket (CLAUDE.md ladder gotcha).
Layout: rungs 1 and 6 (baseline, +AugLab val000) live at <root>/t1wce/<run_id>;
rungs 2-5 at <root>/ablations/t1wce/<run_id> (06_01_evaluate_run.sh LADDER=1).

Usage:
  .venv/bin/python 06_10_ladder_summary_ispy2cross_t1wce_t1wce.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/breast_cancer/ispy1
sys.path.insert(0, str(DATASET_ROOT.parent.parent.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_ispy1/02_metrics/ispy2_model/t1wce"
ABLATIONS_ROOT = METRICS_ROOT / "ablations" / "t1wce"

IN_DOMAIN = "t1wce"
OOD_CONTRASTS = ["t1wce"]   # single item, doubling as its own OOD entry

TS = "20260905_163655"
RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     f"t1wce/ispy2_t1wce_baseline_{TS}"),
    ("+kmeans", "+ K-means intensity clustering",
     f"ablations/t1wce/ispy2_t1wce_baseline_kmeans_{TS}"),
    ("+label_remap", "+ label remap",
     f"ablations/t1wce/ispy2_t1wce_baseline_kmeans_label_remap_{TS}"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     f"ablations/t1wce/ispy2_t1wce_baseline_kmeans_label_remap_voronoi_{TS}"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     f"ablations/t1wce/ispy2_t1wce_v26_6_2_train050_val100_{TS}"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"t1wce/ispy2_t1wce_auglabAug_v26_6_2_train050_val000_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="ispy1 cross-dataset t1wce (ispy2 T1WCE-trained)",
              contrast_label="t1wce",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
