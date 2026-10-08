#!/usr/bin/env python3
"""
2026-10-01: *_uniap items (L-R + skin-anchored A-P crop, 02_04_derive_ap_crop.sh) replace the
L-R-only *_uni items (archived at benchmark/03_archive/duke-breast-mri_lr_only_unicrop_eval_20261001/).
duke-breast-mri cross-dataset (ISPY2 T1WCE-trained) causal-ablation ladder on the
UNILATERAL-CROP `t1wce_uniap` test item -- the standard and only reported duke evaluation
since 2026-09-17 (bilateral ladders archived at
benchmark/03_archive/duke-breast-mri_bilateral_eval_20260917/). Sibling of
06_11/06_12/06_13_ladder_summary_ispy2cross_*.py (other train-contrast/test-item
combinations); see 02_nnunet/02_03_derive_unilateral_crop.py for the crop method.
Same-contrast/cross-DATASET arm for this direction -- NOT held-out-contrast evidence; supplementary only, never pooled into an OOD bucket (see CLAUDE.md ladder gotcha).
Layout: rungs 1 and 6 (baseline, +AugLab val000) live at <root>/t1wce_uniap/<run_id>;
rungs 2-5 at <root>/ablations/t1wce_uniap/<run_id>.

Usage:
  .venv/bin/python 06_10_ladder_summary_ispy2cross_t1wce_uniap_t1wce.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/breast_cancer/duke-breast-mri
sys.path.insert(0, str(DATASET_ROOT.parent.parent.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_duke-breast-mri/02_metrics/ispy2_model/t1wce"
ABLATIONS_ROOT = METRICS_ROOT / "ablations" / "t1wce_uniap"

IN_DOMAIN = "t1wce_uniap"
OOD_CONTRASTS = ["t1wce_uniap"]   # duke's single item, doubling as its own OOD entry

TS = "20260905_163655"
RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     f"t1wce_uniap/ispy2_t1wce_baseline_{TS}"),
    ("+kmeans", "+ K-means intensity clustering",
     f"ablations/t1wce_uniap/ispy2_t1wce_baseline_kmeans_{TS}"),
    ("+label_remap", "+ label remap",
     f"ablations/t1wce_uniap/ispy2_t1wce_baseline_kmeans_label_remap_{TS}"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     f"ablations/t1wce_uniap/ispy2_t1wce_baseline_kmeans_label_remap_voronoi_lblvor_20261007_171154"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "ablations/t1wce_uniap/ispy2_t1wce_v26_6_2_train050_val000_20261005_222919"),   # val000 (2026-10-07; was ..._val100_{TS})
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"t1wce_uniap/ispy2_t1wce_auglabAug_v26_6_2_train050_val000_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="duke-breast-mri cross-dataset t1wce_uniap unilateral (ispy2 T1WCE-trained)",
              contrast_label="t1wce_uniap",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
