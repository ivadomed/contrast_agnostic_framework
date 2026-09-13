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

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/ispy2
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_ispy2/02_metrics/ispy2_model/t1wce"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t1wce"
OOD_CONTRASTS = ["t2w"]

DUKE_ROOT = (DATASET_ROOT.parent / "duke-breast-mri"
             / "8_results_duke-breast-mri/02_metrics/ispy2_model/t1wce")
# duke stores its pre-contrast evaluation in a parallel `precontrast/` subdir
# under each run-key prefix; run_subdir addresses it with this same rung list.
EXTRA_OOD_SOURCES = [{"metrics_root": DUKE_ROOT, "run_subdir": "precontrast"}]

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
    run_ladder(task_name="ispy2 T1WCE (own-model)", contrast_label="t1wce",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS,
              extra_ood_sources=EXTRA_OOD_SOURCES)
