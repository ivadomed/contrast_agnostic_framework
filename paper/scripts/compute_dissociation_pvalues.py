#!/usr/bin/env python3
"""
Computes the paired significance of the fill-swap step (rung 3, "+voronoi
noise fill" -> rung 4, "v26_6_2 real fill") for each of the 8 causal-ablation
ladders (all 4 headline tasks x both training modalities), for
tab:dissociation in the paper (which reports Delta Dice + p, moving the
Delta HD95 column to supplementary per user request).

Reuses load_case_means/resolve_run_dir from the shared ladder engine and
wilcoxon_p/holm from stat_tests.py -- no new statistical machinery.

Usage:
  .venv/bin/python compute_dissociation_pvalues.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "datasets/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(REPO / "datasets/00_commun_scripts/00_03_evaluate"))
from ladder_ood_common import load_case_means, resolve_run_dir  # noqa: E402
from stat_tests import holm, wilcoxon_p  # noqa: E402


def _load_wrapper(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WRAPPERS = [
    ("Open-MS FLAIR", "datasets/open-ms/5_scripts_open-ms/06_evaluate/06_14_ladder_summary_ood.py"),
    ("Open-MS T1w", "datasets/open-ms/5_scripts_open-ms/06_evaluate/06_18_ladder_summary_t1w.py"),
    ("Brats-GLI T1n", "datasets/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_13_ladder_summary.py"),
    ("Brats-GLI T2w", "datasets/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_14_ladder_summary_t2w.py"),
    ("CHAOS T1in", "datasets/chaos/5_scripts_chaos/06_evaluate/06_34_ladder_summary_t1in.py"),
    ("CHAOS T2spir", "datasets/chaos/5_scripts_chaos/06_evaluate/06_33_ladder_summary_t2spir.py"),
    ("ON-Harmony T1w", "datasets/on-harmony/5_scripts_on-harmony/06_evaluate/06_10_ladder_summary.py"),
    ("ON-Harmony T2w", "datasets/on-harmony/5_scripts_on-harmony/06_evaluate/06_11_ladder_summary_t2w.py"),
]

pvals = []
names = []
for name, rel in WRAPPERS:
    mod = _load_wrapper(REPO / rel)
    rung3_key = mod.RUNGS[3][2]  # "+voronoi (noise fill)"
    rung4_key = mod.RUNGS[4][2]  # "v26_6_2 (real fill)"
    dir3 = resolve_run_dir(mod.METRICS_ROOT, rung3_key)
    dir4 = resolve_run_dir(mod.METRICS_ROOT, rung4_key)
    cases3 = load_case_means(dir3, "dice")
    cases4 = load_case_means(dir4, "dice")
    x, y = [], []
    for contrast in mod.OOD_CONTRASTS:
        c3 = cases3.get(contrast, {})
        c4 = cases4.get(contrast, {})
        for case_id in sorted(set(c3) & set(c4)):
            x.append(c3[case_id])
            y.append(c4[case_id])
    p = wilcoxon_p(np.array(x), np.array(y))
    pvals.append(p)
    names.append(name)
    print(f"{name}: n={len(x)} p_raw={p:.4g}")

adj = holm(pvals)
print("\nHolm-corrected:")
for name, p in zip(names, adj):
    print(f"  {name}: p={p:.4g}")
