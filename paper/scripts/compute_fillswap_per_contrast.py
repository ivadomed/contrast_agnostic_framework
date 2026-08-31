#!/usr/bin/env python3
"""
Breaks the fill-swap step (rung 3 "+voronoi noise fill" -> rung 4 "v26_6_2 real
fill") down PER individual held-out eval contrast, instead of pooling all OOD
contrasts into one number as compute_dissociation_pvalues.py does. Motivation:
Open-MS T1w and Brats-GLI T2w show no significant *pooled* fill-swap effect
(paper/compute_dissociation_pvalues.py) -- this checks whether that is because
the effect is genuinely flat everywhere, or because it is significant but
opposite-signed on different held-out contrasts and cancels in the pool.

Reuses load_case_means/resolve_run_dir from the shared ladder engine and
wilcoxon_p/holm from stat_tests.py -- no new statistical machinery.

Usage:
  .venv/bin/python compute_fillswap_per_contrast.py
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
from stat_tests import holm, wilcoxon_p, fmt_p  # noqa: E402


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

for name, rel in WRAPPERS:
    mod = _load_wrapper(REPO / rel)
    rung3_key = mod.RUNGS[3][2]
    rung4_key = mod.RUNGS[4][2]
    dir3 = resolve_run_dir(mod.METRICS_ROOT, rung3_key)
    dir4 = resolve_run_dir(mod.METRICS_ROOT, rung4_key)
    cases3 = load_case_means(dir3, "dice")
    cases4 = load_case_means(dir4, "dice")

    print(f"\n=== {name} (train contrast excluded; per-contrast fill-swap effect) ===")
    pvals, deltas, ns, contrasts = [], [], [], []
    for contrast in mod.OOD_CONTRASTS:
        c3 = cases3.get(contrast, {})
        c4 = cases4.get(contrast, {})
        common = sorted(set(c3) & set(c4))
        x = np.array([c3[k] for k in common])
        y = np.array([c4[k] for k in common])
        p = wilcoxon_p(x, y) if len(x) else float("nan")
        delta = float(np.mean(y) - np.mean(x)) * 100 if len(x) else float("nan")
        pvals.append(p)
        deltas.append(delta)
        ns.append(len(common))
        contrasts.append(contrast)

    adj = holm(pvals)
    for c, n, d, p in zip(contrasts, ns, deltas, adj):
        sign = "+" if d >= 0 else ""
        flag = " ***SIG***" if (np.isfinite(p) and p < 0.05) else ""
        print(f"  {c:8s} n={n:3d}  Delta Dice={sign}{d:6.2f}  p(Holm,within-contrast-family)={fmt_p(p)}{flag}")

    pooled_x, pooled_y = [], []
    for contrast in mod.OOD_CONTRASTS:
        c3 = cases3.get(contrast, {}); c4 = cases4.get(contrast, {})
        for k in sorted(set(c3) & set(c4)):
            pooled_x.append(c3[k]); pooled_y.append(c4[k])
    pooled_p = wilcoxon_p(np.array(pooled_x), np.array(pooled_y))
    pooled_delta = float(np.mean(pooled_y) - np.mean(pooled_x)) * 100
    print(f"  [pooled across all OOD contrasts: Delta Dice={pooled_delta:+.2f}, p={fmt_p(pooled_p)}]")
