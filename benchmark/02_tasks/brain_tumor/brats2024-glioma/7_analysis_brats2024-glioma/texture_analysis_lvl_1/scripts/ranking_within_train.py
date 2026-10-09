#!/usr/bin/env python
"""
Within-training-contrast ranking test (Paul, 2026-09-24): fix the training contrast and the
region, rank the 3 held-out eval contrasts by a cross-contrast similarity measure and by
transfer quality, and ask whether the orders agree. Comparing only within a row removes the
"mirror twin" problem of the pooled comparison (t2w->t1n and t1n->t2w share one symmetric NGF
value but sit in different rows) and any per-training-contrast offset.

Measures (pooled over patients, from the existing tables):
  NGF                  symmetric texture similarity           (ngf_vs_fill_swap_pooled.csv)
  eta2(train|eval)     training contrast recoverable from eval (containment_vs_fill_swap_pooled.csv)
  eta2(eval|train)     reverse direction
Targets (mean Dice over GT-present patients, folds 0-2, same population as the ablation table):
  realfill Dice, noisefill Dice, delta = realfill - noisefill.
Noise-fill never saw real texture, so it is the reference: if texture similarity drives
transfer, NGF should order the real-fill model's Dice better than the noise-fill model's.

Per row: Kendall S = concordant - discordant eval-contrast pairs (3 pairs, S in {-3,-1,1,3}),
tau = S/3. Pooled over rows: sum of S, exact one-sided p (agreement) under independent random
orderings within each row (exact convolution of the 6-permutation null per row).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from compute_cross_contrast_ngf import FOLDS, load_rung_eval  # noqa: E402

OUT = THIS_DIR.parent / "outputs"
DATA, TABLES, PLOTS = OUT / "data", OUT / "tables", OUT / "plots"
TRAINS = ("t1n", "t2w", "t2f")
REGIONS = ("SNFH", "RC", "ET", "NCR")
MEASURES = {"NGF": "ngf", "eta2(train|eval)": "eta_fwd", "eta2(eval|train)": "eta_rev"}
TARGETS = {"realfill Dice": "dice_realfill", "noisefill Dice": "dice_voronoi", "delta": "delta"}


def dice_table() -> pd.DataFrame:
    present = pd.read_csv(DATA / "gt_region_presence.csv")
    present = present[present["gt_vox"] > 0][["case", "region"]]
    rows = []
    for tr in TRAINS:
        per = {}
        for rung in ("voronoi", "realfill"):
            df = load_rung_eval(tr, rung)
            assert set(df["fold"]) <= set(FOLDS)
            df = df[df["label"].isin(REGIONS)].rename(columns={"group": "eval", "label": "region"})
            df = df.merge(present, on=["case", "region"])
            per[rung] = df.groupby(["eval", "case", "region"])["dice"].mean()
        paired = pd.concat({"dice_voronoi": per["voronoi"], "dice_realfill": per["realfill"]}, axis=1).dropna()
        m = paired.groupby(["eval", "region"]).mean().reset_index()
        m["train"] = tr
        rows.append(m)
    d = pd.concat(rows, ignore_index=True)
    d = d[d["train"] != d["eval"]]
    d["delta"] = d["dice_realfill"] - d["dice_voronoi"]
    return d


def kendall_S(x, y) -> int:
    s = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        a, b = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        s += int(a * b)
    return s


def row_null(n=3):
    """Distribution of S for a fixed x and a uniformly random ordering of y (no ties)."""
    x = np.arange(n)
    vals = [kendall_S(x, np.array(p)) for p in itertools.permutations(range(n))]
    u, c = np.unique(vals, return_counts=True)
    return dict(zip(u.tolist(), (c / c.sum()).tolist()))


def pooled_p(S_obs: int, n_rows: int) -> float:
    one = row_null()
    dist = {0: 1.0}
    for _ in range(n_rows):
        new = {}
        for s, p in dist.items():
            for v, q in one.items():
                new[s + v] = new.get(s + v, 0.0) + p * q
        dist = new
    return float(sum(p for s, p in dist.items() if s >= S_obs))


def main():
    ngf = pd.read_csv(DATA / "ngf_vs_fill_swap_pooled.csv")[["region", "train", "eval", "ngf_region"]]
    eta = pd.read_csv(DATA / "containment_vs_fill_swap_pooled.csv")[
        ["region", "train", "eval", "eta2_train_given_eval", "eta2_eval_given_train"]]
    t = (dice_table().merge(ngf, on=["train", "eval", "region"])
         .merge(eta, on=["train", "eval", "region"])
         .rename(columns={"ngf_region": "ngf", "eta2_train_given_eval": "eta_fwd",
                          "eta2_eval_given_train": "eta_rev"}))
    t.to_csv(DATA / "ranking_within_train_cells.csv", index=False)

    row_rows = []
    for (tr, region), g in t.groupby(["train", "region"]):
        g = g.sort_values("eval")
        rec = dict(train=tr, region=region, evals=",".join(g["eval"]))
        for mname, mcol in MEASURES.items():
            for tname, tcol in TARGETS.items():
                rec[f"{mname} vs {tname}"] = kendall_S(g[mcol].to_numpy(), g[tcol].to_numpy())
        row_rows.append(rec)
    rows = pd.DataFrame(row_rows)
    rows["region"] = pd.Categorical(rows["region"], REGIONS, ordered=True)
    rows = rows.sort_values(["region", "train"]).reset_index(drop=True)
    rows.to_csv(DATA / "ranking_within_train_rows.csv", index=False)

    combos = [f"{m} vs {tt}" for m in MEASURES for tt in TARGETS]
    summ = []
    for scope, sub in [("all regions", rows), ("edema (SNFH) only", rows[rows["region"] == "SNFH"])]:
        for c in combos:
            S = int(sub[c].sum())
            summ.append(dict(scope=scope, comparison=c, n_rows=len(sub), S=S,
                             tau=S / (3 * len(sub)), p_one_sided=pooled_p(S, len(sub))))
    summ = pd.DataFrame(summ)
    summ.to_csv(DATA / "ranking_within_train_summary.csv", index=False)

    lines = ["# Within-training-contrast ranking: do eval contrasts order the same way by similarity and by transfer?", "",
             "Each row fixes a training contrast and a region and ranks its 3 held-out eval contrasts. "
             "S = concordant − discordant pairs (−3..3); pooled τ = ΣS / (3·rows); p = exact one-sided "
             "probability of ΣS this large under random orderings within each row. Dice = mean over "
             "GT-present patients, folds 0–2.", "", "## Pooled", "",
             "| scope | comparison | rows | ΣS | τ | p (agreement) |", "|---|---|--:|--:|--:|--:|"]
    for _, r in summ.iterrows():
        lines.append(f"| {r['scope']} | {r['comparison']} | {r['n_rows']} | {r['S']:+d} | {r['tau']:+.2f} | {r['p_one_sided']:.3g} |")
    lines += ["", "## Per row (edema first)", ""]
    for (region, tr), g in t.assign(region=pd.Categorical(t["region"], REGIONS, ordered=True)).sort_values(
            ["region", "train", "eval"]).groupby(["region", "train"], observed=True):
        lines.append(f"**{region}, trained on {tr}**")
        lines.append("")
        lines.append("| eval | realfill Dice | noisefill Dice | Δ | NGF | η²(train\\|eval) | η²(eval\\|train) |")
        lines.append("|---|--:|--:|--:|--:|--:|--:|")
        for _, r in g.sort_values("ngf", ascending=False).iterrows():
            lines.append(f"| {r['eval']} | {100*r['dice_realfill']:.1f} | {100*r['dice_voronoi']:.1f} | "
                         f"{100*r['delta']:+.1f} | {r['ngf']:.3f} | {r['eta_fwd']:.3f} | {r['eta_rev']:.3f} |")
        rr = rows[(rows["train"] == tr) & (rows["region"] == region)].iloc[0]
        lines.append(f"\nKendall S (NGF vs realfill / noisefill / Δ): {rr['NGF vs realfill Dice']:+d} / "
                     f"{rr['NGF vs noisefill Dice']:+d} / {rr['NGF vs delta']:+d}\n")
    (TABLES / "ranking_within_train.md").write_text("\n".join(lines))

    mat = rows[combos].to_numpy(float) / 3
    fig, ax = plt.subplots(figsize=(12, 6.2))
    im = ax.imshow(mat, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([c.replace(" vs ", "\nvs\n") for c in combos], fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r} | train {tr}" for r, tr in zip(rows["region"], rows["train"])], fontsize=8.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(mat[i, j]) > 0.6 else "black")
    for x in (2.5, 5.5):
        ax.axvline(x, color="#333", linewidth=1.2)
    for y in (2.5, 5.5, 8.5):
        ax.axhline(y, color="#333", linewidth=0.6)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Kendall τ within row (3 eval contrasts)")
    edema = summ[summ["scope"] == "all regions"].set_index("comparison")
    ax.set_title("Within each training contrast × region: does similarity rank the eval contrasts like transfer does?\n"
                 f"pooled τ, all rows — NGF vs realfill {edema.loc['NGF vs realfill Dice','tau']:+.2f}, "
                 f"vs noisefill {edema.loc['NGF vs noisefill Dice','tau']:+.2f}, vs Δ {edema.loc['NGF vs delta','tau']:+.2f}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "ranking_within_train.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(t[t["region"] == "SNFH"].sort_values(["train", "ngf"], ascending=[True, False])[
        ["train", "eval", "dice_realfill", "dice_voronoi", "delta", "ngf", "eta_fwd", "eta_rev"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
