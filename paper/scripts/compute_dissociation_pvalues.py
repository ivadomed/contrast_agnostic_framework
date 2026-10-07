#!/usr/bin/env python3
"""
Builds tab:dissociation: the fill-swap step (rung 3 "+voronoi noise fill" ->
rung 4 "v26_6_2 real fill") for every causal-ablation ladder.

ONE correction family (2026-10-07): significance is claimed at the TASK level
only -- the seven task-pooled paired Wilcoxon tests that annotate fig:ladder
(make_per_contrast_curves.panel_pooled: every training modality of the task,
same-patient pairs merged), Holm-corrected across the seven tasks. The table
prints one such row per task, then its per-setting rows as a descriptive
breakdown with the engine's UNCORRECTED pooled p and no significance marks.
Before this the settings formed a second family of sixteen, so MS read as
significant in the figure and not in the table for no reason but family size.

Reads each ladder's `ladder_series.json` -- written by the shared engine
(00_commun_scripts/00_03_evaluate/ladder_ood_common.py), which computes the
per-contrast and pooled fill-swap Wilcoxon tests itself and stores them under
`fill_swap_significance`. That JSON is the single source of truth; this script
only selects which ladders are paper rows, applies the across-ladder Holm
correction (which no individual ladder can do, since it cannot see the others),
and formats the table.

Rewritten 2026-09-09 (was: import each dataset's wrapper module and re-derive
the statistic from METRICS_ROOT/OOD_CONTRASTS). The import approach could not
reach cross-dataset ladders whose wrappers build their rungs inside main()
rather than at module level -- toothfairy2's does -- and it maintained a second
implementation of a test the engine already runs. Reading the JSON removes both
problems and makes adding a task a one-line change. Verified equivalent on the
original 8 rows: the engine's raw pooled p reproduces the old import-based
value exactly (e.g. CHAOS T1in 1.721e-4).

Usage:
  .venv/bin/python compute_dissociation_pvalues.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils"))
from stat_tests import holm  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_per_contrast_curves as mpc  # noqa: E402  (PANELS, load, panel_pooled: the figure's own test)

# LADDER_PENDING (comma-separated task names, as in make_per_contrast_curves): tasks whose ladder
# awaits a retrain print \pending cells instead of stale numbers.
PENDING = {t.strip() for t in os.environ.get("LADDER_PENDING", "").split(",") if t.strip()}

M = ("benchmark/02_tasks/{task}/{ds}/8_results_{ds}/02_metrics/{model}/{contrast}"
     "/ablations/ladder_series.json")
# Same mapping as make_per_contrast_curves.py's TASK_OF. Before this existed the
# template lacked the 2026-09-27 task folder, so EVERY row silently resolved to
# a missing file and the script printed an empty table with only a warning.
TASK_OF = {"totalseg-pelvic": "pelvis_healthy", "chaos": "abdomen_healthy", "on-harmony": "brain_healthy",
           "toothfairy2": "mandible_healthy", "brats2024-glioma": "brain_tumor",
           "open-ms": "brain_ms", "ispy2": "breast_cancer"}


def _j(ds, model, contrast):
    return M.format(task=TASK_OF[ds], ds=ds, model=model, contrast=contrast)


# (paper row label, boundary type, ladder_series.json path)
#
# ONE ROW PER (task, training modality) -- every modality the headline table
# pools, so this family matches tab:meta's. 14 rows since 2026-10-02: BraTS
# trains on all four of its contrasts and ON-Harmony on three, and a
# Holm correction over a smaller family than the table claims would be
# silently anti-conservative.
#
#  * ToothFairy2 trains on CBCT only, so its OOD axis is necessarily
#    cross-DATASET (HaN-Seg + PDDCA). Footnoted in the table.
#  * I-SPY2's rows pool its external cohorts (duke-breast-mri, ispy1,
#    acrin6698) by TRUE contrast, inside the ladder wrappers themselves.
ROWS = [
    ("Abdomen T1in",       "tissue interface",            _j("chaos", "chaos_model", "t1in")),
    ("Abdomen T2spir",     "tissue interface",            _j("chaos", "chaos_model", "t2spir")),
    ("Brain T1w",   "tissue interface",            _j("on-harmony", "on_harmony_model", "T1w")),
    ("Brain T2w",   "tissue interface",            _j("on-harmony", "on_harmony_model", "T2w")),
    ("Brain DWI",   "tissue interface",            _j("on-harmony", "on_harmony_model", "dwi_ap")),
    ("Mandible CBCT",    r"tissue interface$^\ddagger$", _j("toothfairy2", "toothfairy2_model", "cbct")),
    ("Pelvis CT",        "tissue interface",            _j("totalseg-pelvic", "totalseg_pelvic_model", "ct")),
    ("Pelvis MRI",       "tissue interface",            _j("totalseg-pelvic", "totalseg_pelvic_model", "mri")),
    ("MS FLAIR",    "no tissue interface",         _j("open-ms", "open_ms_model", "flair")),
    ("MS T1w",      "no tissue interface",         _j("open-ms", "open_ms_model", "t1w")),
    ("Glioma T1n",    "no tissue interface",         _j("brats2024-glioma", "brats2024_glioma_model", "t1n")),
    ("Glioma T1c",    "no tissue interface",         _j("brats2024-glioma", "brats2024_glioma_model", "t1c")),
    ("Glioma T2w",    "no tissue interface",         _j("brats2024-glioma", "brats2024_glioma_model", "t2w")),
    ("Glioma FLAIR",  "no tissue interface",         _j("brats2024-glioma", "brats2024_glioma_model", "t2f")),
    ("Breast T1-CE",     "no tissue interface",         _j("ispy2", "ispy2_model", "t1wce")),
    ("Breast T2w",       "no tissue interface",         _j("ispy2", "ispy2_model", "t2w")),
]

# Supplementary: the one Duke arm held OUT of the pooled I-SPY2 rows above,
# because it is cross-DATASET at the SAME contrast the model trained on rather
# than held-out-contrast evidence. Reported so the exclusion is visible, not
# silent -- it is by far the largest fill-swap effect anywhere in the study.
SUPPLEMENTARY = [
    ("Duke T1WCE-trained $\\to$ t1wce (cross-dataset, SAME contrast)",
     "benchmark/02_tasks/breast_cancer/duke-breast-mri/8_results_duke-breast-mri/02_metrics/ispy2_model/t1wce/ablations/t1wce_uniap/ladder_series.json"),  # L-R + A-P crop, the current standard (2026-10-01)
]

FILL = 4  # index of the "v26_6_2 (real fill)" rung; delta is FILL-1 -> FILL


def read(rel: str, metric: str = "dice"):
    p = REPO / rel
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    series = d.get(metric, [])
    sig = (d.get("fill_swap_significance") or {}).get(metric)
    if len(series) <= FILL or not sig:
        return None
    return {
        "before": series[FILL - 1],
        "delta": series[FILL] - series[FILL - 1],
        # Relative to where this step starts (Paul, 2026-10-03): +3 points at 80% Dice
        # is a small boost to a nearly solved task, +5 at 40% is a large share of the
        # model's competence. The table and text lead with this, absolute delta beside it.
        "rel": 100.0 * (series[FILL] - series[FILL - 1]) / series[FILL - 1],
        "p_raw": sig["pooled_p"],
        "n": sig.get("n_cases"),
        "per_contrast": sig.get("per_contrast", {}),
    }


# Paper row label -> figure panel title (the task it belongs to).
TASK_OF_ROW = {label: label.split(" ")[0] for label, _, _ in ROWS}


def task_rows(metric: str):
    """One row per task: the exact test make_per_contrast_curves annotates on fig:ladder
    (panel_pooled), plus the panel-average noise-fill level it divides by for the
    relative gain. delta is printed as the metric's own change (HD95 negative = better),
    whereas panel_pooled returns an 'improvement'."""
    higher = metric == "dice"
    out = []
    for title, grp, items in mpc.PANELS:
        ladders = [d for d in (mpc.load(rel) for _, rel in items) if d is not None]
        if not ladders:
            continue
        p, dl = mpc.panel_pooled(ladders, metric, higher)
        before = float(np.nanmean([np.asarray(d[metric][:mpc.N_RUNGS], float)[FILL - 1] for d in ladders]))
        delta = dl if higher else -dl
        out.append({"task": title, "group": grp, "before": before, "delta": delta,
                    "rel": 100.0 * delta / before if before else float("nan"), "p_raw": p})
    return out


def fmt_p(p):
    """2 significant figures, matching the table's existing style (6.9e-4, 0.37)."""
    if not np.isfinite(p):
        return "---"
    return f"{p:.2g}".replace("e-0", "e-")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="dice", choices=("dice", "hd95"))
    metric = ap.parse_args().metric
    got = [(label, btype, read(rel, metric)) for label, btype, rel in ROWS]
    missing = [l for l, _, r in got if r is None]
    rows = [(l, b, r) for l, b, r in got if r is not None]
    if missing:
        print(f"WARNING: no ladder data for: {', '.join(missing)}\n", file=sys.stderr)

    # THE family: one task-pooled test per task (the figure's own), Holm over the tasks.
    tasks = task_rows(metric)
    adj_task = holm([t["p_raw"] for t in tasks])
    task_p = {t["task"]: pa for t, pa in zip(tasks, adj_task)}

    print(f"Fill-swap step. Family = {len(tasks)} task-level tests, Holm-corrected ({metric}); "
          f"setting rows uncorrected\n")
    print(f"{'task / setting':<22} {'boundary':<22} {'before':>7} {'delta':>7} {'rel%':>7} {'p_raw':>10} {'p_holm':>10} {'n':>5}")
    for t, pa in zip(tasks, adj_task):
        print(f"{t['task'].upper():<22} {t['group']:<22} {t['before']:7.1f} {t['delta']:+7.2f} {t['rel']:+7.1f} "
              f"{t['p_raw']:10.3g} {pa:10.3g} {'-':>5}")
        for label, btype, r in rows:
            if TASK_OF_ROW[label] != t["task"]:
                continue
            plain = btype.replace(r"$^\dagger$", "+").replace(r"$^\ddagger$", "*")
            print(f"  {label:<20} {plain:<22} {r['before']:7.1f} {r['delta']:+7.2f} {r['rel']:+7.1f} "
                  f"{r['p_raw']:10.3g} {'(uncorr.)':>10} {r['n'] or '-':>5}")

    print(f"\n--- LaTeX rows for tab:dissociation ({metric}) ---")
    # Columns: task or setting & level before the step & absolute delta & relative delta (%)
    # & p. Task rows: Holm-corrected over the seven tasks, * = significant. Setting rows
    # (indented): the engine's uncorrected pooled p, never starred.
    prev = None
    for t, pa in zip(tasks, adj_task):
        if t["group"] != prev:
            if prev is not None:
                print(r"\midrule")
            title = "Appearance-defined" if t["group"] == "no_interface" else "Interface-bounded"
            print(rf"\multicolumn{{5}}{{l}}{{\emph{{{title}}}}} \\")
        prev = t["group"]
        if t["task"] in PENDING:
            print(rf"\textbf{{{t['task']}}} (all settings) & \pending & \pending & \pending & \pending \\")
        else:
            d, rel = f"{t['delta']:+.2f}", f"{t['rel']:+.1f}"
            if pa < 0.05:
                d += "^{*}"
            print(rf"\textbf{{{t['task']}}} (all settings) & {t['before']:.1f} & ${d}$ & ${rel}$ & {fmt_p(pa)} \\")
        for label, btype, r in rows:
            if TASK_OF_ROW[label] != t["task"]:
                continue
            mark = btype[btype.index("$"):] if "$" in btype else ""
            if t["task"] in PENDING:
                print(rf"\quad {label + mark} & \pending & \pending & \pending & \pending \\")
            else:
                print(rf"\quad {label + mark} & {r['before']:.1f} & ${r['delta']:+.2f}$ & ${r['rel']:+.1f}$ & {fmt_p(r['p_raw'])} \\")

    print("\n--- supplementary (external-cohort breast confirmation, not in tab:dissociation) ---")
    sup = [(l, read(rel)) for l, rel in SUPPLEMENTARY]
    sup = [(l, r) for l, r in sup if r is not None]
    if sup:
        sadj = holm([r["p_raw"] for _, r in sup])
        for (l, r), pa in zip(sup, sadj):
            print(f"  {l:<62} {r['delta']:+6.2f}  p_raw={r['p_raw']:.3g}  "
                  f"p_holm={pa:.3g}  n={r['n']}")

    print("\n--- per-contrast (drives fig:ladder's per-curve colouring) ---")
    for label, _, r in rows:
        if r["per_contrast"]:
            cells = "  ".join(f"{c}={p:.3g}" for c, p in r["per_contrast"].items())
            print(f"  {label:<20} {cells}")


if __name__ == "__main__":
    main()
