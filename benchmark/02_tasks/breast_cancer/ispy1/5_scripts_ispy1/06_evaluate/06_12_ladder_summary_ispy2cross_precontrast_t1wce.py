#!/usr/bin/env python3
"""
ispy1 cross-dataset (ISPY2 T1WCE-trained) causal-ablation ladder on the I-SPY1
`precontrast` test item (167 MAMA-MIA-expert-masked cases, natively unilateral -- see
02_nnunet/02_01_convert_test.py). Mirrors duke-breast-mri's 06_1X ladder scripts.
Genuinely held-out contrast for this direction (t1wce-trained model -> I-SPY1 precontrast): valid OOD evidence.
Layout: rungs 1 and 6 (baseline, +AugLab val000) live at <root>/precontrast/<run_id>;
rungs 2-5 at <root>/ablations/precontrast/<run_id> (06_01_evaluate_run.sh LADDER=1).

Usage:
  .venv/bin/python 06_12_ladder_summary_ispy2cross_precontrast_t1wce.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/breast_cancer/ispy1
sys.path.insert(0, str(DATASET_ROOT.parent.parent.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_ispy1/02_metrics/ispy2_model/t1wce"
ABLATIONS_ROOT = METRICS_ROOT / "ablations" / "precontrast"

IN_DOMAIN = "precontrast"
OOD_CONTRASTS = ["precontrast"]   # single item, doubling as its own OOD entry

TS = "20260905_163655"
RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     f"precontrast/ispy2_t1wce_baseline_{TS}"),
    ("+kmeans", "+ K-means intensity clustering",
     f"ablations/precontrast/ispy2_t1wce_baseline_kmeans_{TS}"),
    ("+label_remap", "+ label remap",
     f"ablations/precontrast/ispy2_t1wce_baseline_kmeans_label_remap_{TS}"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     f"ablations/precontrast/ispy2_t1wce_baseline_kmeans_label_remap_voronoi_{TS}"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "ablations/precontrast/ispy2_t1wce_v26_6_2_train050_val000_20261005_222919"),   # val000 (2026-10-07; was ..._val100_{TS})
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"precontrast/ispy2_t1wce_auglabAug_v26_6_2_train050_val000_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="ispy1 cross-dataset precontrast (ispy2 T1WCE-trained)",
              contrast_label="precontrast",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
