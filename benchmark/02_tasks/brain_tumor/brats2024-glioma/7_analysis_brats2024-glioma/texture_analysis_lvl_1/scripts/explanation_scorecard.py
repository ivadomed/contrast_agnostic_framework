#!/usr/bin/env python
"""
Scorecard: every significant noise-fill -> real-fill change (BraTS, Holm-significant OOD cells) vs every
non-flawed analysis. Cell colour: green = the analysis points the same way as the observed change (a
possible explanation), red = it points the other way, orange = too weak to call / not tested on that cell.

Excluded as flawed: full-GT FLATTEN (border leakage), learned-synthesis R^2 (DL-as-evidence), early breast run.

Rules (fixed before colouring):
  level measures (NGF similarity, eta^2(train|eval)): z-score within region over its 9 OOD cells;
      z > +0.25 predicts HELPS, z < -0.25 predicts HURTS, else orange.
  gap / signed measures (visibility gap, ramp gap, confusion margin, texture-energy gap, anatomy-SI gap):
      predicted sign = sign(value); orange if |value| < 0.25 * SD of that measure over all 36 OOD cells.
  error signature (descriptive mechanism check, SNFH/RC only): failure green if recall drops significantly
      (p<0.05) while precision does not drop significantly; red if precision drops significantly (over-call);
      success green if recall or precision rises significantly and neither drops significantly; else orange.
  causal interventions (border-preserving only): green where a causal test supports an explanation of
      that cell's change, orange where untested.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs"
D = OUT / "data"
KEY = ["train", "eval", "region"]
GREEN, ORANGE, RED = "#4c9f70", "#f0b356", "#d1584c"


def load():
    sig = pd.read_csv(D / "region_fill_swap_significance.csv")
    sig = sig[sig["family"] == "OOD"].copy()
    t = sig[KEY + ["mean_delta_pts", "p_holm", "significant", "direction"]]
    t = t.merge(pd.read_csv(D / "ngf_vs_fill_swap_pooled.csv")[KEY + ["ngf_region"]], on=KEY)
    t = t.merge(pd.read_csv(D / "containment_vs_fill_swap_pooled.csv")[KEY + ["eta2_train_given_eval"]], on=KEY)
    st = pd.read_csv(D / "step_affine_cells.csv")
    vis = st.groupby(["eval", "region"])["step_auc"].mean().to_dict()
    t["vis_gap"] = [vis[(e, r)] - vis[(tr, r)] for tr, e, r in zip(t["train"], t["eval"], t["region"])]
    t = t.merge(pd.read_csv(D / "internal_ramp_cells.csv")[KEY + ["R_gap"]], on=KEY)
    t = t.merge(pd.read_csv(D / "region_surround_cells.csv")[KEY + ["margin"]], on=KEY)
    t = t.merge(pd.read_csv(D / "anisotropy_vs_fill_swap.csv")[KEY + ["E_mean_eval_minus_train"]], on=KEY)
    txt = (OUT / "tables" / "anatomy_vs_pathology_texture.md").read_text().split("## raw/")[0]
    si = {(m[1], m[2]): float(m[3]) for m in re.finditer(
        r"\| (t1n|t1c|t2w|t2f) \| (SNFH|RC|ET|NCR) \| \d+ \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|", txt)}
    t["SI_gap"] = [si[(e, r)] - si[(tr, r)] for tr, e, r in zip(t["train"], t["eval"], t["region"])]
    err = {}
    for tr, f in (("t2w", "error_characterization_summary.csv"), ("t1n", "error_characterization_t1n_summary.csv"),
                  ("t2f", "error_characterization_t2f_summary.csv")):
        e = pd.read_csv(D / f)
        for r in e.itertuples():
            err[(tr, r.eval, r.region, r.metric)] = (r.median_diff, r.p)
    return t, err


def verdict_sign(pred_sign, outcome_sign):
    if pred_sign == 0:
        return "O"
    return "G" if pred_sign == outcome_sign else "R"


def main():
    t, err = load()
    for c in ("ngf_region", "eta2_train_given_eval"):
        t[c + "_z"] = t.groupby("region")[c].transform(lambda s: (s - s.mean()) / s.std())
    gap_cols = ["vis_gap", "R_gap", "margin", "E_mean_eval_minus_train", "SI_gap"]
    sd = {c: t[c].std() for c in gap_cols}
    cells = t[t["significant"]].copy()
    cells["out"] = np.sign(cells["mean_delta_pts"]).astype(int)
    cells = cells.sort_values("mean_delta_pts").reset_index(drop=True)

    cols = [("Texture similarity\n(NGF)", "ngf_region_z", "level", "{:+.2f}z"),
            ("Containment\nη²(train|eval)", "eta2_train_given_eval_z", "level", "{:+.2f}z"),
            ("Visibility gap\n(step AUC eval−train)", "vis_gap", "gap", "{:+.3f}"),
            ("Internal ramp gap\n(R eval−train)", "R_gap", "gap", "{:+.3f}"),
            ("Region vs surroundings\nconfusion margin", "margin", "gap", "{:+.3f}"),
            ("Texture energy gap\n(eval−train)", "E_mean_eval_minus_train", "gap", "{:+.3f}"),
            ("Anatomy coupling gap\n(SI eval−train)", "SI_gap", "gap", "{:+.2f}")]
    V = {name: [] for name, *_ in cols}
    T = {name: [] for name, *_ in cols}
    for r in cells.itertuples():
        for name, col, kind, fmt in cols:
            v = getattr(r, col)
            if kind == "level":
                ps = 1 if v > 0.25 else (-1 if v < -0.25 else 0)
            else:
                ps = 0 if abs(v) < 0.25 * sd[col] else int(np.sign(v))
            V[name].append(verdict_sign(ps, r.out))
            T[name].append(fmt.format(v))

    # error signature
    name = "Error signature\n(mechanism check)"
    V[name], T[name] = [], []
    for r in cells.itertuples():
        rec = err.get((r.train, r.eval, r.region, "recall"))
        pre = err.get((r.train, r.eval, r.region, "precision"))
        if rec is None or pre is None:
            V[name].append("O"); T[name].append("not computed"); continue
        rd, rp = rec; pd_, pp = pre
        rec_dn, rec_up = rd < 0 and rp < 0.05, rd > 0 and rp < 0.05
        pre_dn, pre_up = pd_ < 0 and pp < 0.05, pd_ > 0 and pp < 0.05
        if r.out < 0:
            v = "G" if rec_dn and not pre_dn else ("R" if pre_dn else "O")
        else:
            v = "G" if (rec_up or pre_up) and not (rec_dn or pre_dn) else ("R" if (rec_dn and pre_dn) else "O")
        V[name].append(v); T[name].append(f"rec {rd:+.2f} / prec {pd_:+.2f}")

    # causal interventions (border-preserving)
    name = "GT-mask intervention\n(leaky, see FINDINGS §12)"
    manual = {("t1n", "t2f", "SNFH"): ("G", "core→iid noise:\nnoise↑ real↓ (p=4e-6)"),
              ("t2w", "t2f", "SNFH"): ("G", "core→iid noise:\nDiD +0.03 (p=2e-4)"),
              ("t2w", "t1n", "SNFH"): ("G", "dark step rescues\nreal-fill (p=2e-5)")}
    V[name], T[name] = [], []
    for r in cells.itertuples():
        v, s = manual.get((r.train, r.eval, r.region), ("O", "not tested"))
        V[name].append(v); T[name].append(s)

    names = list(V)
    rows = [f"{r.train}→{r.eval}  {r.region}  ({r.mean_delta_pts:+.1f})" for r in cells.itertuples()]
    colour = {"G": GREEN, "O": ORANGE, "R": RED}
    fig, ax = plt.subplots(figsize=(2.0 * len(names) + 3.2, 0.55 * len(rows) + 2.6))
    for j, n in enumerate(names):
        for i in range(len(rows)):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colour[V[n][i]], ec="white", lw=1.5))
            ax.text(j + 0.5, i + 0.5, T[n][i], ha="center", va="center", fontsize=6.6,
                    color="white" if V[n][i] != "O" else "#333", wrap=True)
    ax.set_xlim(0, len(names)); ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(names)) + 0.5); ax.set_xticklabels(names, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(rows, fontsize=8.5)
    for i, r in enumerate(cells.itertuples()):
        ax.get_yticklabels()[i].set_color(RED if r.out < 0 else GREEN)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fail = [i for i, r in enumerate(cells.itertuples()) if r.out < 0]
    succ = [i for i, r in enumerate(cells.itertuples()) if r.out > 0]
    split = "   ".join(f"{n.split(chr(10))[0]}: failures {sum(V[n][i]=='G' for i in fail)}/{len(fail)}, successes {sum(V[n][i]=='G' for i in succ)}/{len(succ)}" for n in names)
    counts = "   ".join(f"{n.split(chr(10))[0]}: {V[n].count('G')}G/{V[n].count('O')}O/{V[n].count('R')}R" for n in names)
    fig.text(0.01, 0.01, "Rows: Holm-significant noise→real changes (Δ Dice pts; red label = real-fill worse). "
             "Green = analysis agrees with the observed direction, red = contradicts, orange = too weak / untested.\n"
             + counts + "\nGreen on failures vs successes (baseline 'always helps' = 0/4 vs 10/10):  " + split, fontsize=7, color="#333", va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUT / "plots" / "explanation_scorecard.png", dpi=160, bbox_inches="tight")

    md = ["# Explanation scorecard — significant noise→real changes (BraTS)", "",
          "G = agrees with observed direction, O = too weak / untested, R = contradicts. Rules in the script docstring.", "",
          "| cell | " + " | ".join(n.replace(chr(10), " ") for n in names) + " |",
          "|---|" + "---|" * len(names)]
    for i, row in enumerate(rows):
        md.append(f"| {row} | " + " | ".join(f"{V[n][i]} ({T[n][i]})" for n in names) + " |")
    md += ["", "| analysis | G | O | R |", "|---|--:|--:|--:|"]
    md += [f"| {n.replace(chr(10), ' ')} | {V[n].count('G')} | {V[n].count('O')} | {V[n].count('R')} |" for n in names]
    md += ["", "Green on failures / successes: " + split]
    (OUT / "tables" / "explanation_scorecard.md").write_text("\n".join(md))
    print("\n".join(md[-(len(names) + 2):]))


if __name__ == "__main__":
    main()
