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

2026-09-01: the pooled OOD Dice/HD95 figure now uses the SAME contrast_groups
tree (T1_family/ce_t1w_external/t2wi/dwi, equal weight per contrast-phase group)
as the headline atlas-liver-hcc_cross_dataset_t1w_01_results.yaml config -- loaded
directly from that YAML file (not hand-copied) so the two can never silently
disagree. Previously used a flat, case-count-weighted mean, which gave lld-mmri-hcc's
942 T1-ish cases (post-2026-09-01 phase extension) ~17x the weight of liverhccseg's
~56 -- exactly the vote-stacking problem contrast_groups exists to prevent, just not
applied here until now.

Usage:
  .venv/bin/python 06_13_ladder_summary.py
"""
import os
import sys
from pathlib import Path

import yaml

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/atlas-liver-hcc
REPO_ROOT = DATASET_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder_cross_dataset, _dataset_name  # noqa: E402

ABLATIONS_ROOT = DATASET_ROOT / "8_results_atlas-liver-hcc/02_metrics/atlas_liver_hcc_model/t1w/ablations"

OOD_SOURCES = [
    REPO_ROOT / "lld-mmri-hcc/8_results_lld-mmri-hcc/02_metrics/atlas_liver_hcc_model/t1w",
    REPO_ROOT / "liverhccseg/8_results_liverhccseg/02_metrics/atlas_liver_hcc_model/t1w",
]

# Load contrast_groups + sources (incl. the in-domain atlas_t1w source, which
# contrast_groups' "atlas_own" leaf needs) straight from the headline config --
# same tree, no drift. PROJECT_ROOT must be set (env.sh normally does this; set
# a sane default here since this script can be invoked standalone).
os.environ.setdefault("PROJECT_ROOT", str(REPO_ROOT.parent))
_CFG_PATH = (DATASET_ROOT / "5_scripts_atlas-liver-hcc" / "06_evaluate" / "configs"
             / "atlas-liver-hcc_cross_dataset_t1w_01_results.yaml")
_cfg = yaml.safe_load(_CFG_PATH.read_text())
CONTRAST_GROUPS = _cfg["contrast_groups"] if _cfg.get("use_contrast_groups", True) else None
IN_DOMAIN_GROUP = _cfg.get("in_domain_group")  # excluded from the pooled OOD figure below
PREFIXED_SOURCES = [
    (Path(os.path.expandvars(s["metrics_dir"])), s.get("column_prefix", ""))
    for s in _cfg["sources"]
]


def _leaf_column_names(node) -> list:
    """Collect every raw (prefixed) column-name leaf under a contrast_groups
    subtree, ignoring dict/list structure -- used only to derive which raw
    '<dataset>/<item>' labels the in_domain_group covers, for
    exclude_contrast_labels below."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        if "column" in node:
            return [node["column"]]
        out = []
        for v in node.values():
            out += _leaf_column_names(v)
        return out
    if isinstance(node, list):
        out = []
        for n in node:
            out += _leaf_column_names(n)
        return out
    return []


# Raw '<dataset>/<item>' labels covered by the in-domain group (e.g. "CE_T1w_all"),
# derived from the config itself (not hand-listed) so it can't drift -- these are
# excluded from the per-contrast breakdown PNG/table (see run_ladder_cross_dataset's
# exclude_contrast_labels docstring: showing them next to a mean that no longer
# includes them would be misleading, not just incomplete).
EXCLUDE_CONTRAST_LABELS = []
if IN_DOMAIN_GROUP and CONTRAST_GROUPS and IN_DOMAIN_GROUP in CONTRAST_GROUPS:
    for prefixed_col in _leaf_column_names(CONTRAST_GROUPS[IN_DOMAIN_GROUP]):
        for metrics_root, prefix in PREFIXED_SOURCES:
            if prefix and prefixed_col.startswith(prefix):
                EXCLUDE_CONTRAST_LABELS.append(f"{_dataset_name(metrics_root)}/{prefixed_col[len(prefix):]}")
                break

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
                              rungs=RUNGS, contrast_groups=CONTRAST_GROUPS,
                              prefixed_sources=PREFIXED_SOURCES,
                              in_domain_group=IN_DOMAIN_GROUP,
                              exclude_contrast_labels=EXCLUDE_CONTRAST_LABELS)
