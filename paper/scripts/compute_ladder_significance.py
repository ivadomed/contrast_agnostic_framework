#!/usr/bin/env python3
"""
Computes, for each causal-ablation ladder, the within-ladder Holm-corrected
paired Wilcoxon significance of every rung-transition (rung i vs rung i-1,
i=1..6 -- a family of 6 tests per ladder), for tab:ladder-dice (supplementary
Table 9). Marks significant DECREASES with an extra flag so the LaTeX can
render the "***^\\downarrow" style used for Brats-GLI T1n's Voronoi rung and
ON-Harmony's fill-swap rung.

Reuses load_case_means/resolve_run_dir from the shared ladder engine and
wilcoxon_p/holm from stat_tests.py -- no new statistical machinery.

Usage:
  .venv/bin/python compute_ladder_significance.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "datasets/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(REPO / "datasets/00_commun_scripts/00_03_evaluate"))
from ladder_ood_common import load_case_means, resolve_run_dir, rung_means  # noqa: E402
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

STARS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]


def stars(p):
    for thresh, s in STARS:
        if p < thresh:
            return s
    return ""


for name, rel in WRAPPERS:
    mod = _load_wrapper(REPO / rel)
    print(f"\n=== {name} ===")
    pvals, decreases = [], []
    ood_vals = []
    for i in range(1, len(mod.RUNGS)):
        prev_key = mod.RUNGS[i - 1][2]
        cur_key = mod.RUNGS[i][2]
        dir_prev = resolve_run_dir(mod.METRICS_ROOT, prev_key)
        dir_cur = resolve_run_dir(mod.METRICS_ROOT, cur_key)
        cases_prev = load_case_means(dir_prev, "dice")
        cases_cur = load_case_means(dir_cur, "dice")
        x, y = [], []
        for contrast in mod.OOD_CONTRASTS:
            cp = cases_prev.get(contrast, {})
            cc = cases_cur.get(contrast, {})
            for case_id in sorted(set(cp) & set(cc)):
                x.append(cp[case_id])
                y.append(cc[case_id])
        x, y = np.array(x), np.array(y)
        p = wilcoxon_p(x, y) if len(x) else float("nan")
        pvals.append(p)
        decreases.append(bool(len(y) and np.mean(y) < np.mean(x)))
        if i == 1:
            ood_prev, _ = rung_means(mod.METRICS_ROOT, prev_key, "dice", mod.IN_DOMAIN, mod.OOD_CONTRASTS)
            ood_vals.append(ood_prev)
        ood_cur, _ = rung_means(mod.METRICS_ROOT, cur_key, "dice", mod.IN_DOMAIN, mod.OOD_CONTRASTS)
        ood_vals.append(ood_cur)
    adj = holm(pvals)
    for i, (p, dec) in enumerate(zip(adj, decreases)):
        label = f"{mod.RUNGS[i][0]} -> {mod.RUNGS[i+1][0]}"
        s = stars(p)
        flag = " (DECREASE)" if (s and dec) else ""
        print(f"  {label}: p={fmt_p(p)} {s}{flag}")

    print("  LaTeX cells:")
    cells = [f"{ood_vals[0]:.1f}"]
    for i, (p, dec) in enumerate(zip(adj, decreases)):
        v = f"{ood_vals[i+1]:.1f}"
        s = stars(p)
        if s and dec:
            cells.append(f"${v}^{{{s}}}{{\\downarrow}}$")
        elif s:
            cells.append(f"${v}^{{{s}}}$")
        else:
            cells.append(v)
    print("   " + " & ".join(cells) + " \\\\")
