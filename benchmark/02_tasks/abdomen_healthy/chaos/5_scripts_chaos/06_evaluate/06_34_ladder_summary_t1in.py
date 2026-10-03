#!/usr/bin/env python3
"""
CHAOS T1in causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py (see that module's
docstring for why this used to be copy-pasted per dataset and no longer is).

CROSS-DATASET (2026-10-03, Paul: every ladder must be cross-dataset). The OOD
pool adds CHAOS's external cohorts, scored on volumes cropped to the CHAOS
field of view BEFORE prediction (the `fov_crop/` metrics, same as the headline
table): AMOS CT + MRI and SLIVER07 CT. Pooled by TRUE contrast via the engine's
grouped mode (as the I-SPY2 ladders do), so CT gets one vote however many
cohorts supply it: ct = CHAOS ct + AMOS ct + SLIVER07 ct; AMOS's MRI is its own
T2w group.

GUARD: the engine silently skips a source that has no run dir for a rung, which
would mix rungs scored with and without the external cohorts and make the
rung-to-rung deltas meaningless. This wrapper refuses to run until every rung
exists in every external source (pass --allow-partial only for inspection).

Usage:
  .venv/bin/python 06_34_ladder_summary_t1in.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/abdomen_healthy/chaos
sys.path.insert(0, str(DATASET_ROOT.parent.parent.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_chaos/02_metrics/chaos_model/t1in"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

ABDOMEN = DATASET_ROOT.parent
EXTRA_OOD_SOURCES = [
    ABDOMEN / "amos/8_results_amos/02_metrics/chaos_model/t1in/fov_crop",
    ABDOMEN / "sliver07/8_results_sliver07/02_metrics/chaos_model/t1in/fov_crop",
]
OOD_GROUPS = {"t1out": ["t1out"], "t2spir": ["t2spir"],
              "ct": ["ct", "amos/ct", "sliver07/ct"], "t2w": ["amos/mri"]}

IN_DOMAIN = "t1in"
OOD_CONTRASTS = ["t1out", "t2spir", "ct"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     "nnUNet_chaos_t1in_baseline_20260614_153230"),
    ("+kmeans", "+ K-means intensity clustering",
     "nnUNet_chaos_t1in_baseline_kmeans_train050_val000_20260715_081839"),
    ("+label_remap", "+ label remap",
     "nnUNet_chaos_t1in_baseline_kmeans_label_remap_train050_val000_20260715_081919"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "nnUNet_chaos_t1in_baseline_kmeans_label_remap_voronoi_train050_val000_20260715_081959"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "nnUNet_chaos_t1in_v26_6_2_train050_val000_20260715_082040"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     "auglab_chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     "auglab_chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420"),
]

def _missing_external(rungs, sources):
    """(rung label, source) pairs with no run dir -- see GUARD in the docstring."""
    from ladder_ood_common import resolve_run_dir
    return [(label, str(src)) for label, _, key in rungs for src in sources
            if not resolve_run_dir(src, key).is_dir()]


if __name__ == "__main__":
    missing = _missing_external(RUNGS, EXTRA_OOD_SOURCES)
    if missing and "--allow-partial" not in sys.argv:
        sys.exit("Not all rungs are predicted on the external cohorts yet; refusing to "
                 "mix pooled and unpooled rungs:\n  "
                 + "\n  ".join(f"{l}  <-  {src}" for l, src in missing))
    run_ladder(task_name="CHAOS T1in", contrast_label="t1in",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS,
              extra_ood_sources=EXTRA_OOD_SOURCES, ood_groups=OOD_GROUPS)
