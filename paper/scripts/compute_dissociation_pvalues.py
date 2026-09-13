#!/usr/bin/env python3
"""
Builds tab:dissociation: the fill-swap step (rung 3 "+voronoi noise fill" ->
rung 4 "v26_6_2 real fill") for every causal-ablation ladder, Holm-corrected
across the whole family.

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
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "datasets/00_commun_scripts/00_00_utils"))
from stat_tests import holm  # noqa: E402

M = "datasets/{ds}/8_results_{ds}/02_metrics/{model}/{contrast}/ablations/ladder_series.json"

# (paper row label, boundary type, ladder_series.json path)
#
# ONE ROW PER (task, training modality), each on that task's OWN held-out
# contrast axis. Two notes on what is deliberately NOT here:
#
#  * ToothFairy2 trains on CBCT only, so it has no held-out in-house contrast
#    and its OOD axis is necessarily cross-DATASET (HaN-Seg CT + MR). Footnoted
#    in the table rather than silently mixed in with the cross-contrast rows.
#  * The Duke-breast-MRI (MAMA-MIA) cohort is an external confirmation of the
#    I-SPY2 result, not a separate task, so it gets no row of its own. Its
#    genuinely held-out-contrast arms are instead POOLED INTO the two I-SPY2
#    rows as extra equally-weighted OOD items by the ladder wrappers themselves
#    (ispy2 06_04/06_05 `extra_ood_sources`) -- duke pre-contrast for the
#    T1WCE-trained ladder, duke t1wce + pre-contrast for the T2w-trained one.
#    The one combination that is NOT held-out-contrast evidence -- the
#    T1WCE-trained model tested on duke t1wce, i.e. cross-DATASET at the SAME
#    contrast -- is deliberately excluded from the pool and reported in the
#    supplement below, so every row of this table stays one estimand.
ROWS = [
    ("CHAOS T1in",      "tissue interface",
     M.format(ds="chaos", model="chaos_model", contrast="t1in")),
    ("CHAOS T2spir",    "tissue interface",
     M.format(ds="chaos", model="chaos_model", contrast="t2spir")),
    ("ON-Harmony T1w",  r"tissue interface$^\dagger$",
     M.format(ds="on-harmony", model="on_harmony_model", contrast="T1w")),
    ("ON-Harmony T2w",  r"tissue interface$^\dagger$",
     M.format(ds="on-harmony", model="on_harmony_model", contrast="T2w")),
    ("ToothFairy2 CBCT", r"tissue interface$^\ddagger$",
     M.format(ds="toothfairy2", model="toothfairy2_model", contrast="cbct")),
    ("Open-MS FLAIR",   "no tissue interface",
     M.format(ds="open-ms", model="open_ms_model", contrast="flair")),
    ("Open-MS T1w",     "no tissue interface",
     M.format(ds="open-ms", model="open_ms_model", contrast="t1w")),
    ("Brats-GLI T1n",   "no tissue interface",
     M.format(ds="brats2024-glioma", model="brats2024_glioma_model", contrast="t1n")),
    ("Brats-GLI T2w",   "no tissue interface",
     M.format(ds="brats2024-glioma", model="brats2024_glioma_model", contrast="t2w")),
    ("I-SPY2 T1WCE",    "no tissue interface",
     M.format(ds="ispy2", model="ispy2_model", contrast="t1wce")),
    ("I-SPY2 T2w",      "no tissue interface",
     M.format(ds="ispy2", model="ispy2_model", contrast="t2w")),
]

# Supplementary: the one Duke arm held OUT of the pooled I-SPY2 rows above,
# because it is cross-DATASET at the SAME contrast the model trained on rather
# than held-out-contrast evidence. Reported so the exclusion is visible, not
# silent -- it is by far the largest fill-swap effect anywhere in the study.
SUPPLEMENTARY = [
    ("Duke T1WCE-trained $\\to$ t1wce (cross-dataset, SAME contrast)",
     M.format(ds="duke-breast-mri", model="ispy2_model", contrast="t1wce")),
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
        "delta": series[FILL] - series[FILL - 1],
        "p_raw": sig["pooled_p"],
        "n": sig.get("n_cases"),
        "per_contrast": sig.get("per_contrast", {}),
    }


def fmt_p(p):
    """2 significant figures, matching the table's existing style (6.9e-4, 0.37)."""
    if not np.isfinite(p):
        return "---"
    return f"{p:.2g}".replace("e-0", "e-")


def main():
    got = [(label, btype, read(rel)) for label, btype, rel in ROWS]
    missing = [l for l, _, r in got if r is None]
    rows = [(l, b, r) for l, b, r in got if r is not None]
    if missing:
        print(f"WARNING: no ladder data for: {', '.join(missing)}\n", file=sys.stderr)

    adj = holm([r["p_raw"] for _, _, r in rows])

    print(f"Fill-swap step, Holm-corrected across {len(rows)} ladders (dice)\n")
    print(f"{'task':<20} {'boundary':<32} {'dDice':>7} {'p_raw':>10} {'p_holm':>10} {'n':>5}")
    for (label, btype, r), pa in zip(rows, adj):
        plain = btype.replace(r"$^\dagger$", "+").replace(r"$^\ddagger$", "*")
        print(f"{label:<20} {plain:<32} {r['delta']:+7.2f} {r['p_raw']:10.3g} "
              f"{pa:10.3g} {r['n'] or '-':>5}")

    print("\n--- LaTeX rows for tab:dissociation ---")
    prev = None
    for (label, btype, r), pa in zip(rows, adj):
        if prev is not None and btype.startswith("no") != prev.startswith("no"):
            print(r"\midrule")
        bold = abs(r["delta"]) >= 4.0          # match existing convention: bold the LARGE effects
        d = f"{r['delta']:+.2f}"
        d = rf"$\mathbf{{{d}}}$" if bold else f"${d}$"
        print(f"{label:<20}& {btype} & {d} & {fmt_p(pa)} \\\\")
        prev = btype

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
