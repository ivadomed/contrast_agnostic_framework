#!/usr/bin/env python3
"""
duke-breast-mri cross-dataset (ISPY2 T1WCE-trained) causal-ablation ladder,
scored on the PRE-CONTRAST (no Gd) test contrast -- a third test contrast
alongside the headline t1wce item. See 06_21_ladder_summary_ispy2cross_t1wce.py
and 02_nnunet/02_02_convert_test_precontrast.py for the full rationale
(I-SPY2's t1wce training data is uniformly first-post-contrast, so
pre-contrast is a genuinely clean held-out contrast test). Same single-item-
as-own-OOD-entry convention as 06_21 (see that file's header).

Rungs 1 (baseline floor) and 6 (+AugLab val000 = the headline OURS run) are
identical to two of the 6 headline precontrast methods already evaluated in
the main precontrast pass -- they live under
8_results_duke-breast-mri/02_metrics/ispy2_model/t1wce/precontrast/ (no
"ablations/" prefix). Rungs 2-5 (the ladder-specific intermediate methods)
were evaluated separately and live under .../t1wce/ablations/precontrast/.
METRICS_ROOT is set to the t1wce contrast root so both locations resolve via
relative run_key prefixes -- see resolve_run_dir() in ladder_ood_common.py.

Usage:
  .venv/bin/python 06_23_ladder_summary_ispy2cross_precontrast_t1wce.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/duke-breast-mri
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_duke-breast-mri/02_metrics/ispy2_model/t1wce"
ABLATIONS_ROOT = METRICS_ROOT / "ablations" / "precontrast"

IN_DOMAIN = "precontrast"
OOD_CONTRASTS = ["precontrast"]   # duke's single item, doubling as its own OOD entry

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
     f"ablations/precontrast/ispy2_t1wce_v26_6_2_train050_val100_{TS}"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"precontrast/ispy2_t1wce_auglabAug_v26_6_2_train050_val000_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="duke-breast-mri cross-dataset PRE-CONTRAST (ispy2 T1WCE-trained)",
              contrast_label="precontrast",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS)
