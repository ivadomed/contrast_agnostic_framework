#!/usr/bin/env python3
"""
atlas-liver-hcc causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py's run_ladder_cross_dataset()
(same rung mechanics as the other datasets' ladders, see that module's docstring),
EXCEPT: atlas-liver-hcc trains on a single modality (T1w) permanently, so there is
no held-out training-CONTRAST to use as OOD the way BraTS/CHAOS/ON-Harmony/open-ms
do. Instead, OOD here = pooled cross-DATASET generalization onto atlas-liver-hcc's
two independent evaluation-only test sets (lld-mmri-hcc: T2WI+DWI; liverhccseg: 4
CE-T1w phases) -- the same cross-dataset evaluators already used for the 6-method
headline suite (see each dataset's own 06_01_evaluate_run.sh).

Each ladder rung's RUN_ID must have already been predicted+evaluated by BOTH
lld-mmri-hcc's and liverhccseg's own 06_01_evaluate_run.sh (with METRICS_SUBDIR=
ablations for the 4 non-headline rungs) before this script can score it.

Usage:
  .venv/bin/python 06_13_ladder_summary.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/atlas-liver-hcc
REPO_ROOT = DATASET_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder_cross_dataset  # noqa: E402

ABLATIONS_ROOT = DATASET_ROOT / "8_results_atlas-liver-hcc/02_metrics/atlas_liver_hcc_model/t1w/ablations"

OOD_SOURCES = [
    REPO_ROOT / "lld-mmri-hcc/8_results_lld-mmri-hcc/02_metrics/atlas_liver_hcc_model/t1w",
    REPO_ROOT / "liverhccseg/8_results_liverhccseg/02_metrics/atlas_liver_hcc_model/t1w",
]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     "atlas-liver-hcc_t1w_baseline_20260804_062605"),
    ("+kmeans", "+ K-means intensity clustering",
     "ablations/atlas-liver-hcc_t1w_baseline_kmeans_20260829_165445"),
    ("+label_remap", "+ label remap",
     "ablations/atlas-liver-hcc_t1w_baseline_kmeans_label_remap_20260829_165445"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "ablations/atlas-liver-hcc_t1w_baseline_kmeans_label_remap_voronoi_20260829_165445"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "ablations/atlas-liver-hcc_t1w_v26_6_2_train050_val100_20260829_165445"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     "atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val000_20260804_062605"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     "atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val100_20260804_062605"),
]

if __name__ == "__main__":
    run_ladder_cross_dataset(task_name="atlas-liver-hcc", contrast_label="t1w",
                              ood_sources=OOD_SOURCES, ablations_root=ABLATIONS_ROOT,
                              rungs=RUNGS)
