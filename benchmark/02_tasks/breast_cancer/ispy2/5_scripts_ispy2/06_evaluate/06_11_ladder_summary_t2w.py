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

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/breast_cancer/ispy2
sys.path.insert(0, str(DATASET_ROOT.parent.parent.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_ispy2/02_metrics/ispy2_model/t2w"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t2w"
OOD_CONTRASTS = ["t1wce"]

DUKE_ROOT = (DATASET_ROOT.parent / "duke-breast-mri"
             / "8_results_duke-breast-mri/02_metrics/ispy2_model/t2w")
# 2026-09-17: duke arms are the UNILATERAL-CROP items (t1wce_uniap / precontrast_uniap),
# the standard and only reported duke evaluation -- the bilateral ladders are
# archived at benchmark/03_archive/{duke-breast-mri,ispy2}_bilateral_eval_20260917/.
# Each item lives in a parallel `<item>/` subdir (ablations/<item>/<run_id>).
# 2026-10-01 (Paul): + ispy1 (I-SPY1/MAMA-MIA, both arms -- t1wce AND precontrast are
# held out for a t2w-trained model) and acrin6698 (ACRIN-6698 T0 DWI, `dwi`).
# Pre-extension outputs: ablations/*.bak_20261001_pre_ispy1_acrin.
ISPY1_ROOT = (DATASET_ROOT.parent / "ispy1" / "8_results_ispy1/02_metrics/ispy2_model/t2w")
ACRIN_ROOT = (DATASET_ROOT.parent / "acrin6698" / "8_results_acrin6698/02_metrics/ispy2_model/t2w")
EXTRA_OOD_SOURCES = [{"metrics_root": DUKE_ROOT, "run_subdir": "t1wce_uniap"},
                     {"metrics_root": DUKE_ROOT, "run_subdir": "precontrast_uniap"},
                     {"metrics_root": ISPY1_ROOT, "run_subdir": "t1wce"},
                     {"metrics_root": ISPY1_ROOT, "run_subdir": "precontrast"},
                     {"metrics_root": ACRIN_ROOT, "run_subdir": "dwi_uniap"}]

# Pool by TRUE held-out contrast, not by (dataset, item) column (2026-10-01, Paul):
# every case of a contrast from every cohort goes into ONE pool (all pre-contrast
# together, ...); OOD = equal weight per contrast; OOD-only report; significance is
# patient-merged (an I-SPY2 patient's _uni/_bil FOV variants are one patient).
# Engine: ladder_ood_common.run_ladder(ood_groups=...) -> run_ladder_grouped.
OOD_GROUPS = {
    "t1wce": ["t1wce", "duke-breast-mri/t1wce_uniap", "ispy1/t1wce"],
    "precontrast": ["duke-breast-mri/precontrast_uniap", "ispy1/precontrast"],
    "dwi": ["acrin6698/dwi_uniap"],
}

# 2026-10-01: duke + acrin6698 columns/sources switched to the *_uniap items (L-R + skin-anchored
# A-P crop to I-SPY2 unilateral training geometry); L-R-only versions archived under benchmark/03_archive/.
TS = "20260905_163655"
RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     f"ispy2_t2w_baseline_{TS}"),
    ("+kmeans", "+ K-means intensity clustering",
     f"ablations/ispy2_t2w_baseline_kmeans_{TS}"),
    ("+label_remap", "+ label remap",
     f"ablations/ispy2_t2w_baseline_kmeans_label_remap_{TS}"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     f"ablations/ispy2_t2w_baseline_kmeans_label_remap_voronoi_lblvor_20261007_171431"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "ablations/ispy2_t2w_v26_6_2_train050_val000_20261005_222952"),   # val000 (2026-10-06; was ..._val100_{TS})
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"ispy2_t2w_auglabAug_v26_6_2_train050_val000_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="ispy2 T2w (own-model)", contrast_label="t2w",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS,
              extra_ood_sources=EXTRA_OOD_SOURCES,
              ood_groups=OOD_GROUPS)
