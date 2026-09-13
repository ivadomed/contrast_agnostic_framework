#!/usr/bin/env python3
"""
ispy2 OWN-model T2w causal-ablation ladder. See 06_04_ladder_summary_t1wce.py
for the full rationale (same mechanism, other training contrast). OOD = the
held-out training contrast (t1wce), scored on ispy2's own held-out test
patients. 6 rungs (rung 7 / val100-AugLab-mirror not wired, see 06_04's note).

The OOD pool is NOT limited to I-SPY2's own held-out contrast. The external
duke-breast-mri cohort (MAMA-MIA) was predicted and evaluated on every rung of
this same ladder, so BOTH its arms are pooled in here as equally-weighted
held-out-contrast items (291 cases each, vs the 102 I-SPY2 supplies): unlike
the T1WCE-trained ladder, this model trained on t2w, so duke t1wce AND duke
pre-contrast are both genuinely held-out contrasts.

Usage:
  .venv/bin/python 06_05_ladder_summary_t2w.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/ispy2
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_ispy2/02_metrics/ispy2_model/t2w"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t2w"
OOD_CONTRASTS = ["t1wce"]

DUKE_ROOT = (DATASET_ROOT.parent / "duke-breast-mri"
             / "8_results_duke-breast-mri/02_metrics/ispy2_model/t2w")
# The bare root contributes duke's t1wce arm; the run_subdir entry contributes
# its pre-contrast arm, which lives in a parallel `precontrast/` subdir.
EXTRA_OOD_SOURCES = [DUKE_ROOT,
                     {"metrics_root": DUKE_ROOT, "run_subdir": "precontrast"}]

TS = "20260905_163655"
RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     f"ispy2_t2w_baseline_{TS}"),
    ("+kmeans", "+ K-means intensity clustering",
     f"ablations/ispy2_t2w_baseline_kmeans_{TS}"),
    ("+label_remap", "+ label remap",
     f"ablations/ispy2_t2w_baseline_kmeans_label_remap_{TS}"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     f"ablations/ispy2_t2w_baseline_kmeans_label_remap_voronoi_{TS}"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     f"ablations/ispy2_t2w_v26_6_2_train050_val100_{TS}"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"ispy2_t2w_auglabAug_v26_6_2_train050_val000_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="ispy2 T2w (own-model)", contrast_label="t2w",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS,
              extra_ood_sources=EXTRA_OOD_SOURCES)
