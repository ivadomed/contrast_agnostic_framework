#!/usr/bin/env python3
"""
acrin6698 cross-dataset (ISPY2 T1WCE-trained) causal-ablation ladder on the ACRIN-6698 T0
`dwi_uniap` test item (whole-tumour DWI ROI, lesion-side unilateral crop -- see
02_nnunet/02_01_convert_test.py). Mirrors duke-breast-mri's 06_1X ladder scripts.
Genuinely held-out contrast for this direction (t1wce-trained model -> ACRIN-6698 T0 dwi_uniap): valid OOD evidence.
Layout: rungs 1 and 6 (baseline, +AugLab val000) live at <root>/dwi_uniap/<run_id>;
rungs 2-5 at <root>/ablations/dwi_uniap/<run_id> (06_01_evaluate_run.sh LADDER=1).

Usage:
  .venv/bin/python 06_10_ladder_summary_ispy2cross_dwi_t1wce.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/breast_cancer/acrin6698
sys.path.insert(0, str(DATASET_ROOT.parent.parent.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_acrin6698/02_metrics/ispy2_model/t1wce"
ABLATIONS_ROOT = METRICS_ROOT / "ablations" / "dwi_uniap"

IN_DOMAIN = "dwi_uniap"
OOD_CONTRASTS = ["dwi_uniap"]   # single item, doubling as its own OOD entry

TS = "20260905_163655"
RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     f"dwi_uniap/ispy2_t1wce_baseline_{TS}"),
    ("+kmeans", "+ K-means intensity clustering",
     f"ablations/dwi_uniap/ispy2_t1wce_baseline_kmeans_{TS}"),
    ("+label_remap", "+ label remap",
     f"ablations/dwi_uniap/ispy2_t1wce_baseline_kmeans_label_remap_{TS}"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     f"ablations/dwi_uniap/ispy2_t1wce_baseline_kmeans_label_remap_voronoi_lblvor_20261007_171154"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "ablations/dwi_uniap/ispy2_t1wce_v26_6_2_train050_val000_20261005_222919"),   # val000 (2026-10-07; was ..._val100_{TS})
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"dwi_uniap/ispy2_t1wce_auglabAug_v26_6_2_train050_val000_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="acrin6698 cross-dataset dwi_uniap (ispy2 T1WCE-trained)",
              contrast_label="dwi_uniap",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
