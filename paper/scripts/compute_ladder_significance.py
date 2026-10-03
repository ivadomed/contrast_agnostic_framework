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
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils"))
from stat_tests import fmt_p  # noqa: E402

M = ("benchmark/02_tasks/{task}/{ds}/8_results_{ds}/02_metrics/{model}/{contrast}"
     "/ablations/ladder_series.json")
# Same mapping as make_per_contrast_curves.py; the template lacked the
# 2026-09-27 task folder, so every ladder silently resolved to a missing file.
TASK_OF = {"chaos": "abdomen_healthy", "on-harmony": "brain_healthy",
           "toothfairy2": "mandible_healthy", "brats2024-glioma": "brain_tumor",
           "open-ms": "brain_ms", "ispy2": "breast_cancer"}


def _j(ds, model, contrast):
    return M.format(task=TASK_OF[ds], ds=ds, model=model, contrast=contrast)


# All 14 training modalities, matching compute_dissociation_pvalues.py.
LADDERS = [
    ("Open-MS FLAIR",    _j("open-ms", "open_ms_model", "flair")),
    ("Open-MS T1w",      _j("open-ms", "open_ms_model", "t1w")),
    ("BraTS-GLI T1n",    _j("brats2024-glioma", "brats2024_glioma_model", "t1n")),
    ("BraTS-GLI T1c",    _j("brats2024-glioma", "brats2024_glioma_model", "t1c")),
    ("BraTS-GLI T2w",    _j("brats2024-glioma", "brats2024_glioma_model", "t2w")),
    ("BraTS-GLI FLAIR",  _j("brats2024-glioma", "brats2024_glioma_model", "t2f")),
    ("CHAOS T1in",       _j("chaos", "chaos_model", "t1in")),
    ("CHAOS T2spir",     _j("chaos", "chaos_model", "t2spir")),
    ("ON-Harmony T1w",   _j("on-harmony", "on_harmony_model", "T1w")),
    ("ON-Harmony T2w",   _j("on-harmony", "on_harmony_model", "T2w")),
    ("ON-Harmony DWI",   _j("on-harmony", "on_harmony_model", "dwi_ap")),
    ("ToothFairy2 CBCT", _j("toothfairy2", "toothfairy2_model", "cbct")),
    ("I-SPY2 T1-CE",     _j("ispy2", "ispy2_model", "t1wce")),
    ("I-SPY2 T2w",       _j("ispy2", "ispy2_model", "t2w")),
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
