#!/usr/bin/env python3
"""
ispy2 OWN-model T1WCE causal-ablation ladder. Thin wrapper over the shared
engine in 00_commun_scripts/00_03_evaluate/ladder_ood_common.py's run_ladder()
-- same mechanism as ambl/BraTS/CHAOS's own ladders (ispy2 trains TWO
modalities, needs run_ladder() not run_ladder_cross_dataset()). OOD = the
held-out training contrast (t2w), scored on ispy2's own held-out test patients.

6 rungs (baseline through +AugLab val000) -- rung 7 (+AugLab val100, the
DualVal val100-mirror prediction) is NOT wired here; it needs a separate
predict wrapper (TRAINER=...AugLabDualVal on the val100 RUN_ID) that does not
yet exist for ispy2, unlike ambl's ladder script which already has it. Add it
later if the val000-vs-val100 comparison is wanted for ispy2.

The OOD pool is NOT limited to I-SPY2's own held-out contrast. The external
duke-breast-mri cohort (MAMA-MIA) was predicted and evaluated on every rung of
this same ladder, so its PRE-CONTRAST arm is pooled in here as a second,
equally-weighted held-out-contrast item (291 cases vs the 168 I-SPY2 supplies).
Deliberately NOT pooled: duke's own t1wce arm, which is the SAME contrast this
model trained on -- cross-DATASET evidence, not cross-CONTRAST evidence, and
mixing it in would change what the rung-4->5 delta measures. It stays in
duke-breast-mri's own ladder (06_21) and in the paper's supplement.

Usage:
  .venv/bin/python 06_04_ladder_summary_t1wce.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/breast_cancer/ispy2
sys.path.insert(0, str(DATASET_ROOT.parent.parent.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_ispy2/02_metrics/ispy2_model/t1wce"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t1wce"
OOD_CONTRASTS = ["t2w"]

DUKE_ROOT = (DATASET_ROOT.parent / "duke-breast-mri"
             / "8_results_duke-breast-mri/02_metrics/ispy2_model/t1wce")
# 2026-09-17: the duke arm is the UNILATERAL-CROP item (precontrast_uniap), the
# standard and only reported duke evaluation -- bilateral ladders archived at
# benchmark/03_archive/{duke-breast-mri,ispy2}_bilateral_eval_20260917/.
# run_subdir addresses the parallel `precontrast_uniap/` subdir with this same rung list.
# 2026-10-01 (Paul): the OOD pool also includes the two breast eval cohorts added
# 2026-09-30, same rung list, same parallel-subdir layout:
#   ispy1 (I-SPY1/MAMA-MIA, 167 pts): ONLY its precontrast arm -- its t1wce arm is the
#     training contrast (cross-dataset, not cross-contrast) and stays in ispy1's own
#     ladder, exactly like duke's t1wce arm.
#   acrin6698 (ACRIN-6698 T0 DWI, 371 pts): `dwi` -- held-out contrast for this direction.
# Pre-extension outputs: ablations/*.bak_20261001_pre_ispy1_acrin.
ISPY1_ROOT = (DATASET_ROOT.parent / "ispy1" / "8_results_ispy1/02_metrics/ispy2_model/t1wce")
ACRIN_ROOT = (DATASET_ROOT.parent / "acrin6698" / "8_results_acrin6698/02_metrics/ispy2_model/t1wce")
EXTRA_OOD_SOURCES = [{"metrics_root": DUKE_ROOT, "run_subdir": "precontrast_uniap"},
                     {"metrics_root": ISPY1_ROOT, "run_subdir": "precontrast"},
                     {"metrics_root": ACRIN_ROOT, "run_subdir": "dwi_uniap"}]

# Pool by TRUE held-out contrast, not by (dataset, item) column (2026-10-01, Paul):
# every case of a contrast from every cohort goes into ONE pool (all pre-contrast
# together, ...); OOD = equal weight per contrast; OOD-only report; significance is
# patient-merged (an I-SPY2 patient's _uni/_bil FOV variants are one patient).
# Engine: ladder_ood_common.run_ladder(ood_groups=...) -> run_ladder_grouped.
OOD_GROUPS = {
    "t2w": ["t2w"],
    "precontrast": ["duke-breast-mri/precontrast_uniap", "ispy1/precontrast"],
    "dwi": ["acrin6698/dwi_uniap"],
}

# 2026-10-01: duke + acrin6698 columns/sources switched to the *_uniap items (L-R + skin-anchored
# A-P crop to I-SPY2 unilateral training geometry); L-R-only versions archived under benchmark/03_archive/.
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
     "ablations/ispy2_t1wce_v26_6_2_train050_val000_20261005_222919"),   # val000 (2026-10-06; was ..._val100_{TS})
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"ispy2_t1wce_auglabAug_v26_6_2_train050_val000_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="ispy2 T1WCE (own-model)", contrast_label="t1wce",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS,
              extra_ood_sources=EXTRA_OOD_SOURCES,
              ood_groups=OOD_GROUPS)
