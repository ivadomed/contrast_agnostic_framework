#!/usr/bin/env python3
"""
Per-rung-transition significance for tab:ladder-full (supplementary): for each
causal-ablation ladder, is each step's change from the rung before it
significant, and in which direction?

Reads each ladder's `ladder_series.json` `rung_step_significance`, written by
the shared engine (00_commun_scripts/00_03_evaluate/ladder_ood_common.py),
which runs the test over the ladder's OWN pooled OOD set and Holm-corrects
within that ladder's family of transitions.

Rewritten 2026-09-13. The previous version imported each dataset's wrapper
module and rebuilt the OOD pool from its METRICS_ROOT/OOD_CONTRASTS globals.
That had two failure modes, both silent: it could not reach a wrapper that
builds its rungs inside main() (toothfairy2's does), and it ignored any extra
cross-dataset source the wrapper pooled in (I-SPY2 pools the external Duke
cohort), so its numbers would have disagreed with the same ladder's own
reported figures. Same reason and same fix as compute_dissociation_pvalues.py.

Usage:
  .venv/bin/python compute_ladder_significance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "datasets/00_commun_scripts/00_00_utils"))
from stat_tests import fmt_p  # noqa: E402

M = "datasets/{ds}/8_results_{ds}/02_metrics/{model}/{contrast}/ablations/ladder_series.json"

LADDERS = [
    ("Open-MS FLAIR",    M.format(ds="open-ms", model="open_ms_model", contrast="flair")),
    ("Open-MS T1w",      M.format(ds="open-ms", model="open_ms_model", contrast="t1w")),
    ("Brats-GLI T1n",    M.format(ds="brats2024-glioma", model="brats2024_glioma_model", contrast="t1n")),
    ("Brats-GLI T2w",    M.format(ds="brats2024-glioma", model="brats2024_glioma_model", contrast="t2w")),
    ("CHAOS T1in",       M.format(ds="chaos", model="chaos_model", contrast="t1in")),
    ("CHAOS T2spir",     M.format(ds="chaos", model="chaos_model", contrast="t2spir")),
    ("ON-Harmony T1w",   M.format(ds="on-harmony", model="on_harmony_model", contrast="T1w")),
    ("ON-Harmony T2w",   M.format(ds="on-harmony", model="on_harmony_model", contrast="T2w")),
    ("ToothFairy2 CBCT", M.format(ds="toothfairy2", model="toothfairy2_model", contrast="cbct")),
    ("I-SPY2 T1-CE",     M.format(ds="ispy2", model="ispy2_model", contrast="t1wce")),
    ("I-SPY2 T2w",       M.format(ds="ispy2", model="ispy2_model", contrast="t2w")),
]

STARS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]


def stars(p):
    if not np.isfinite(p):
        return ""
    for thresh, s in STARS:
        if p < thresh:
            return s
    return ""


def main():
    for name, rel in LADDERS:
        path = REPO / rel
        if not path.exists():
            print(f"\n=== {name} ===\n  (no ladder_series.json at {rel})")
            continue
        d = json.loads(path.read_text())
        steps = (d.get("rung_step_significance") or {}).get("dice")
        if not steps:
            print(f"\n=== {name} ===\n  (no rung_step_significance -- regenerate this "
                  f"ladder with the current engine)")
            continue
        series = d["dice"]
        print(f"\n=== {name} ===")
        for st in steps:
            s = stars(st["p_holm"])
            flag = " (DECREASE)" if (s and st["decrease"]) else ""
            print(f"  {st['from']} -> {st['to']}: p={fmt_p(st['p_holm'])} {s}{flag}"
                  f"   n={st['n_cases']}")

        cells = [f"{series[0]:.1f}"]
        for i, st in enumerate(steps):
            v = f"{series[i + 1]:.1f}"
            s = stars(st["p_holm"])
            if s and st["decrease"]:
                cells.append(rf"${v}^{{{s}}}{{\downarrow}}$")
            elif s:
                cells.append(f"${v}^{{{s}}}$")
            else:
                cells.append(v)
        print("  LaTeX cells:\n   " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
