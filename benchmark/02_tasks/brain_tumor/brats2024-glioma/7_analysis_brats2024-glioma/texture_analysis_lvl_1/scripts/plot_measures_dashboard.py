#!/usr/bin/env python
"""
One-page dashboard of every candidate explanation vs the real-fill ablation outcome.

Rows = the 36 cross-contrast cells (train->eval x region), grouped by region and sorted by the
real-fill gain (delta = real-fill - noise-fill Dice). Left: the gain as a bar coloured by
significance outcome. Right: one column per measure, coloured by its WITHIN-REGION rank
(so columns with different units are comparable) and annotated with the raw value. A measure
that explained the gain would shade monotonically down each region block, in step with the bar.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
OUT = THIS_DIR.parent / "outputs"
DATA, PLOTS = OUT / "data", OUT / "plots"
REGIONS = ["SNFH", "RC", "ET", "NCR"]
REGION_NAME = {"SNFH": "edema", "RC": "resection cavity", "ET": "enhancing tumor", "NCR": "necrotic core"}
HELP, HURT, NS = "#2f7d6b", "#c0392b", "#b5b5b5"
KEY = ["train", "eval", "region"]


def load() -> tuple[pd.DataFrame, list]:
    t = pd.read_csv(DATA / "combine_similarity_confusion.csv")
    eta = pd.read_csv(DATA / "containment_vs_fill_swap_pooled.csv")[KEY + ["eta2_eval_given_train"]]
    syn = pd.read_csv(DATA / "synthesis_vs_fill_swap_pooled.csv")[KEY + ["r2_train_given_eval"]]
    ani = pd.read_csv(DATA / "anisotropy_vs_fill_swap.csv")[KEY + ["anisotropy_eval", "E_mean_eval_minus_train"]]
    t = t.merge(eta, on=KEY).merge(syn, on=KEY).merge(ani, on=KEY)
    sep = t.groupby(["eval", "region"])["sep_E"].first().to_dict()          # per-contrast property
    t["sep_T"] = [sep.get((tr, r), np.nan) for tr, r in zip(t["train"], t["region"])]
    t["sep_gap"] = t["sep_E"] - t["sep_T"]
    cols = [  # (column, label, group)
        ("NGF", "NGF\nsimilarity", "same-region similarity"),
        ("eta_fwd", "η²\n(train|eval)", "same-region similarity"),
        ("eta2_eval_given_train", "η²\n(eval|train)", "same-region similarity"),
        ("r2_train_given_eval", "synth R²\n(train|eval)", "same-region similarity"),
        ("margin", "confusion\nmargin", "region vs surroundings"),
        ("sep_E", "separability\n(eval)", "region vs surroundings"),
        ("sep_gap", "separability\ngap (eval−train)*", "region vs surroundings"),
        ("anisotropy_eval", "isotropy\n(eval)", "per-contrast image property"),
        ("E_mean_eval_minus_train", "texture energy\n(eval−train)", "per-contrast image property"),
    ]
    return t, cols


def main():
    t, cols = load()
    t["region"] = pd.Categorical(t["region"], REGIONS, ordered=True)
    t = t.sort_values(["region", "delta"], ascending=[True, False]).reset_index(drop=True)
    ranks = np.column_stack([t.groupby("region", observed=True)[c].rank(pct=True).to_numpy() for c, _, _ in cols])

    n = len(t)
    fig = plt.figure(figsize=(16, 0.36 * n + 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, len(cols)], wspace=0.02)
    axb, axh = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    y = np.arange(n)
    colors = [{"success": HELP, "failure": HURT}.get(o, NS) for o in t["outcome"]]
    axb.barh(y, 100 * t["delta"], color=colors, height=0.72)
    axb.axvline(0, color="#333", lw=0.8)
    axb.set_ylim(n - 0.5, -0.5)
    axb.set_yticks(y)
    axb.set_yticklabels([f"{r.train}→{r.eval}" for r in t.itertuples()], fontsize=8.5)
    axb.set_xlabel("real-fill gain Δ Dice (pts)", fontsize=9)
    axb.tick_params(axis="x", labelsize=8)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    axh.imshow(ranks, aspect="auto", cmap="RdBu", vmin=0, vmax=1, interpolation="nearest")
    for i in range(n):
        for j, (c, _, _) in enumerate(cols):
            v = t.loc[i, c]
            if np.isfinite(v):
                axh.text(j, i, f"{v:+.2f}" if c in ("margin", "sep_gap", "E_mean_eval_minus_train") else f"{v:.2f}",
                         ha="center", va="center", fontsize=7,
                         color="white" if abs(ranks[i, j] - 0.5) > 0.35 else "#222")
    axh.set_yticks([])
    axh.set_xticks(range(len(cols)))
    axh.set_xticklabels([lab for _, lab, _ in cols], fontsize=8)
    axh.xaxis.tick_top()
    for x in (3.5, 6.5):
        axh.axvline(x, color="#222", lw=1.5)
    bounds = t.groupby("region", observed=True).size().cumsum().to_numpy()[:-1]
    starts = np.r_[0, bounds]
    for b in bounds:
        for ax in (axb, axh):
            ax.axhline(b - 0.5, color="#222", lw=1.5)
    ends = np.r_[bounds, n]
    trans = matplotlib.transforms.blended_transform_factory(axb.transAxes, axb.transData)
    for r, s, e in zip(REGIONS, starts, ends):
        axb.text(-0.55, (s + e - 1) / 2, REGION_NAME[r], transform=trans, rotation=90, ha="center",
                 va="center", fontsize=10, fontweight="bold")
    groups = [("same-region similarity", 1.5), ("region vs surroundings", 5.0), ("per-contrast image property", 7.5)]
    for g, x in groups:
        axh.text(x, -1.9, g, ha="center", fontsize=9.5, fontweight="bold", color="#333")

    handles = [plt.Rectangle((0, 0), 1, 1, color=HELP, label="real-fill significantly better"),
               plt.Rectangle((0, 0), 1, 1, color=HURT, label="real-fill significantly worse"),
               plt.Rectangle((0, 0), 1, 1, color=NS, label="n.s.")]
    fig.legend(handles=handles, loc="lower left", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.02, 0.0))
    fig.text(0.98, 0.005,
             "Cell colour = rank of the value within its region (blue high, red low); text = raw value.\n"
             "A measure explaining the gain would shade steadily down each region block, in step with the bars.\n"
             "* separability gap was derived after looking at the data (exploratory).",
             ha="right", va="bottom", fontsize=8, color="#444")
    fig.suptitle("Every candidate explanation vs. the real-fill gain (BraTS2024-glioma, 36 cross-contrast cells)",
                 fontsize=12, y=0.965)
    fig.savefig(PLOTS / "measures_dashboard.png", dpi=150, bbox_inches="tight")
    print(f"Wrote {PLOTS / 'measures_dashboard.png'}")
    t.to_csv(DATA / "measures_dashboard.csv", index=False)


if __name__ == "__main__":
    main()
