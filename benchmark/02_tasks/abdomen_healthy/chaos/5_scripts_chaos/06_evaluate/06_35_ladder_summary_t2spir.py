#!/usr/bin/env python3
"""
CHAOS T2spir causal-ablation ladder. Thin wrapper over the shared engine in
00_commun_scripts/00_03_evaluate/ladder_ood_common.py (retrofitted 2026-08-03;
see that module's docstring). OOD = mean over held-out contrasts (t1in, t1out,
ct), excluding the training contrast (t2spir).

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
  .venv/bin/python 06_35_ladder_summary_t2spir.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/abdomen_healthy/chaos
sys.path.insert(0, str(DATASET_ROOT.parent.parent.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_chaos/02_metrics/chaos_model/t2spir"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

ABDOMEN = DATASET_ROOT.parent
EXTRA_OOD_SOURCES = [
    ABDOMEN / "amos/8_results_amos/02_metrics/chaos_model/t2spir/fov_crop",
    ABDOMEN / "sliver07/8_results_sliver07/02_metrics/chaos_model/t2spir/fov_crop",
]
OOD_GROUPS = {"t1in": ["t1in"], "t1out": ["t1out"],
              "ct": ["ct", "amos/ct", "sliver07/ct"], "t2w": ["amos/mri"]}

IN_DOMAIN = "t2spir"
OOD_CONTRASTS = ["t1in", "t1out", "ct"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     "chaos_t2spir_baseline_20260620_111146"),
    ("+kmeans", "+ K-means intensity clustering",
     "ablations/baseline_kmeans_train050_val000"),
    ("+label_remap", "+ label remap",
     "ablations/baseline_kmeans_label_remap_train050_val000"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "ablations/baseline_kmeans_label_remap_voronoi_train050_val000"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "ablations/chaos_t2spir_v26_6_2_train050_val000_20261005_222654"),   # val000 (real-image checkpoint selection, like rung 4) 2026-10-06; was ..._val100_20261003_123018
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     "chaos_t2spir_auglabAug_v26_6_2_train050_val000_20260710_054202"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     "chaos_t2spir_auglabAug_v26_6_2_train050_val100_20260723_194053"),
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
    run_ladder(task_name="CHAOS T2spir", contrast_label="t2spir",
              metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
              in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS,
              extra_ood_sources=EXTRA_OOD_SOURCES, ood_groups=OOD_GROUPS)
