#!/usr/bin/env python3
"""
duke-breast-mri cross-dataset (ISPY2 T1WCE-trained) causal-ablation ladder:
scores the ispy2 T1WCE-trained ladder rungs on duke-breast-mri's 291-case
external test set. Modeled directly on
ambl/5_scripts_ambl/06_evaluate/06_23_ladder_summary_ispy2cross_t1wce.py (same
mechanism: cross-dataset eval results live under the TARGET dataset's own
tree, distinguished by the ispy2_model subfolder -- see that file for the
convention). Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py's run_ladder().

Unlike ambl (which has both t1wce+t2w test items), duke-breast-mri has only
ONE test item (t1wce/DCE -- no t2w acquisition in this cohort, see
00_utils/env.sh). The shared engine's plots are now OOD-only (see
ladder_ood_common.py's 2026-09-07 rework to match the paper's fig:ladder), so
an empty OOD_CONTRASTS list would leave duke's plot blank. Duke's own
generalization is inherently cross-DATASET (a different institution/
population) rather than cross-contrast, so its single "t1wce" item is set as
BOTH in_domain and its own OOD entry here -- honestly reflecting that this
one item already IS the held-out generalization test for this dataset, just
not a second in-dataset contrast the way ambl/ispy2 have. 6 rungs (rung 7 /
val100-AugLab-mirror not wired -- see ispy2's own 06_04_ladder_summary_t1wce.py
note).

Usage:
  .venv/bin/python 06_21_ladder_summary_ispy2cross_t1wce.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/duke-breast-mri
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_duke-breast-mri/02_metrics/ispy2_model/t1wce"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t1wce"
OOD_CONTRASTS = ["t1wce"]   # duke's single item, doubling as its own OOD entry -- see docstring

TS = "20260905_163655"
RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     f"ispy2_t1wce_baseline_{TS}"),
    ("+kmeans", "+ K-means intensity clustering",
     f"ablations/ispy2_t1wce_baseline_kmeans_{TS}"),
    ("+label_remap", "+ label remap",
     f"ablations/ispy2_t1wce_baseline_kmeans_label_remap_{TS}"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     f"ablations/ispy2_t1wce_baseline_kmeans_label_remap_voronoi_{TS}"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     f"ablations/ispy2_t1wce_v26_6_2_train050_val100_{TS}"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"ispy2_t1wce_auglabAug_v26_6_2_train050_val000_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="duke-breast-mri cross-dataset T1WCE (ispy2-trained)", contrast_label="t1wce",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
