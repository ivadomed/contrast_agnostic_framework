#!/usr/bin/env python3
"""
duke-breast-mri cross-dataset (ISPY2 T2W-trained) causal-ablation ladder on the
UNILATERAL-CROP `precontrast_uni` test item -- the standard and only reported duke evaluation
since 2026-09-17 (bilateral ladders archived at
datasets/03_archive/duke-breast-mri_bilateral_eval_20260917/). Sibling of
06_24_ladder_summary_ispy2cross_precontrast_t2w.py; see 02_nnunet/02_03_derive_unilateral_crop.py for the crop method.

Layout: rungs 1 and 6 (baseline, +AugLab val000) live at <root>/precontrast_uni/<run_id>;
rungs 2-5 at <root>/ablations/precontrast_uni/<run_id>.

Usage:
  .venv/bin/python 06_28_ladder_summary_ispy2cross_precontrast_uni_t2w.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/duke-breast-mri
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_duke-breast-mri/02_metrics/ispy2_model/t2w"
ABLATIONS_ROOT = METRICS_ROOT / "ablations" / "precontrast_uni"

IN_DOMAIN = "precontrast_uni"
OOD_CONTRASTS = ["precontrast_uni"]   # duke's single item, doubling as its own OOD entry

TS = "20260905_163655"
RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     f"precontrast_uni/ispy2_t2w_baseline_{TS}"),
    ("+kmeans", "+ K-means intensity clustering",
     f"ablations/precontrast_uni/ispy2_t2w_baseline_kmeans_{TS}"),
    ("+label_remap", "+ label remap",
     f"ablations/precontrast_uni/ispy2_t2w_baseline_kmeans_label_remap_{TS}"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     f"ablations/precontrast_uni/ispy2_t2w_baseline_kmeans_label_remap_voronoi_{TS}"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     f"ablations/precontrast_uni/ispy2_t2w_v26_6_2_train050_val100_{TS}"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"precontrast_uni/ispy2_t2w_auglabAug_v26_6_2_train050_val000_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="duke-breast-mri cross-dataset precontrast_uni unilateral (ispy2 T2W-trained)",
              contrast_label="precontrast_uni",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
