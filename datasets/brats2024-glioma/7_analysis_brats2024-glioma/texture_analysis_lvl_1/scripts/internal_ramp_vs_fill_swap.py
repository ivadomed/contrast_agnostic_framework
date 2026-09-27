#!/usr/bin/env python
"""
H9 — internal ramp. Noise-fill training shows only FLAT regions (constant + iid noise); real-fill
shows the training contrast's real within-region structure (up to affine incl. sign). Error
analysis: the T1n-trained noise-fill model does not recognise FLAIR edema (72% called background)
and FLAIR edema has a strong depth ramp (bright deep inside, fading to the border), while T1n
edema is flat.

R(patient, contrast, region) = share of within-region intensity variance explained by depth to
the region border = SS_between(depth bins) / SS_total over bins -7..-1 (bin -7 = deepest stored),
from the per-bin sufficient statistics of compute_boundary_profiles.py
(outputs/data/boundary_profile_shard*.csv: n, sum_z, sum_z2 of z-scored intensity). Ratio of
sums of squares => invariant to scale, offset and sign of the intensities (what real-fill
randomises). Cell values = mean over GT-present patients.

PRE-REGISTERED (written before computing any outcome, 2026-09-24):
  P1  noise-fill Dice falls with the eval ramp: within-row tau(noise Dice, -R_eval) > 0.
  P2  real-fill gain rises with ramp gap: Spearman over the 36 OOD cells rho(delta, R_eval - R_train) > 0;
      within rows this is tau(delta, R_eval) > 0.
  P3  (descriptive) edema: R(t2f) > R(t2w) > R(t1c), R(t1n) lowest.
  P4  failures (edema t2w->t1n, t2f->t1n; RC t2w->t1n, t2f->t1c) all have R_train > R_eval.
Tests: within (train, region) rows of 3 eval contrasts, Kendall S pooled over 12 rows, exact
one-sided p (ranking_within_train.kendall_S / pooled_p).
Outputs: data/internal_ramp_{patient,cells}.csv, tables/internal_ramp_vs_fill_swap.md,
plots/internal_ramp_vs_fill_swap.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from ranking_within_train import kendall_S, pooled_p  # noqa: E402

OUT = THIS_DIR.parent / "outputs"
DATA, TABLES, PLOTS = OUT / "data", OUT / "tables", OUT / "plots"
REGIONS = ["SNFH", "RC", "ET", "NCR"]
HELP, HURT, NS = "#2f7d6b", "#c0392b", "#9a9a9a"
MIN_INSIDE_VOX = 200


def ramp_table() -> pd.DataFrame:
    b = pd.concat([pd.read_csv(f) for f in sorted(DATA.glob("boundary_profile_shard*.csv"))], ignore_index=True)
    b = b.drop_duplicates(["patient", "contrast", "region", "bin"])
    b = b[b["bin"] <= -1]
    rows = []
    for (p, c, r), g in b.groupby(["patient", "contrast", "region"]):
        N = g["n"].sum()
        if N < MIN_INSIDE_VOX or len(g) < 2:
            continue
        S, S2 = g["sum_z"].sum(), g["sum_z2"].sum()
        ss_tot = S2 - S * S / N
        ss_between = (g["sum_z"] ** 2 / g["n"]).sum() - S * S / N
        k = len(g)
        rows.append(dict(patient=p, contrast=c, region=r, n_inside=int(N), n_bins=k,
                         R=ss_between / ss_tot if ss_tot > 0 else np.nan,
                         R_adj=1 - ((ss_tot - ss_between) / (N - k)) / (ss_tot / (N - 1)) if ss_tot > 0 else np.nan))
    return pd.DataFrame(rows)


def row_test(t, x, y):
    S, n = 0, 0
    for _, g in t.groupby(["train", "region"]):
        if len(g) == 3 and g[[x, y]].notna().all().all():
            S += kendall_S(g[x].to_numpy(), g[y].to_numpy())
            n += 1
    return S / (3 * n), pooled_p(S, n), n


def main():
    rp = ramp_table()
    rp.to_csv(DATA / "internal_ramp_patient.csv", index=False)
    present = pd.read_csv(DATA / "gt_region_presence.csv")
    present = present[present["gt_vox"] > 0][["case", "region"]].rename(columns={"case": "patient"})
    rp = rp.merge(present, on=["patient", "region"])
    Rc = rp.groupby(["contrast", "region"])["R_adj"].mean()

    cells = pd.read_csv(DATA / "ranking_within_train_cells.csv")[["train", "eval", "region", "dice_realfill", "dice_voronoi", "delta"]]
    sig = pd.read_csv(DATA / "region_fill_swap_significance.csv")
    sig = sig[sig["family"] == "OOD"][["train", "eval", "region", "direction"]]
    cells = cells.merge(sig, on=["train", "eval", "region"])
    cells["outcome"] = cells["direction"].map({"HELPS": "success", "HURTS": "failure"}).fillna("n.s.")
    # paired per-patient gap (same patients in both contrasts), then averaged
    wide = rp.pivot_table(index=["patient", "region"], columns="contrast", values="R_adj")
    gaps = []
    for r in cells.itertuples():
        w = wide.xs(r.region, level="region")[[r.train, r.eval]].dropna()
        gaps.append(((w[r.eval] - w[r.train]).mean(), len(w)))
    cells["R_eval"] = [Rc.get((e, r), np.nan) for e, r in zip(cells["eval"], cells["region"])]
    cells["R_train"] = [Rc.get((t, r), np.nan) for t, r in zip(cells["train"], cells["region"])]
    cells["R_gap"] = [g for g, _ in gaps]
    cells["n_gap"] = [n for _, n in gaps]
    cells["neg_R_eval"] = -cells["R_eval"]
    cells.to_csv(DATA / "internal_ramp_cells.csv", index=False)

    p1 = row_test(cells, "neg_R_eval", "dice_voronoi")
    p2r = row_test(cells, "R_eval", "delta")
    rho2, p2c = spearmanr(cells["R_gap"], cells["delta"])
    fails = cells[cells["outcome"] == "failure"]
    p4 = f"{int((fails['R_gap'] < 0).sum())}/{len(fails)}"
    succ = cells[cells["outcome"] == "success"]
    extra_real = row_test(cells, "neg_R_eval", "dice_realfill")

    L = ["# H9 internal ramp vs. the real-fill ablation (BraTS2024-glioma)", "",
         "R = share of within-region intensity variance explained by depth to the border (bias-corrected, "
         "bins −7..−1; invariant to scale/offset/sign). Predictions fixed before computing (see script docstring).", "",
         "## Pre-registered predictions", "",
         f"- **P1** within-row τ(noise-fill Dice, −R_eval) = {p1[0]:+.2f}, p = {p1[1]:.3g} ({p1[2]} rows) → "
         f"**{'holds' if p1[1] < 0.05 and p1[0] > 0 else 'does not hold'}**",
         f"- **P2** Spearman ρ(Δ, R_eval − R_train) over 36 cells = {rho2:+.2f}, p(two-sided) = {p2c:.3g}; "
         f"within-row τ(Δ, R_eval) = {p2r[0]:+.2f}, p = {p2r[1]:.3g} → "
         f"**{'holds' if (rho2 > 0 and p2c / 2 < 0.05) else 'does not hold'}**",
         f"- **P3** edema R by contrast: " + ", ".join(f"{c} {Rc.get((c, 'SNFH'), np.nan):.3f}" for c in ("t2f", "t2w", "t1c", "t1n")),
         f"- **P4** failures with R_train > R_eval: **{p4}**; successes with R_eval > R_train: "
         f"{int((succ['R_gap'] > 0).sum())}/{len(succ)}", "",
         f"Exploratory: within-row τ(real-fill Dice, −R_eval) = {extra_real[0]:+.2f}, p = {extra_real[1]:.3g}", "",
         "## Mean R per contrast × region", "", "```\n" + Rc.unstack().round(3).to_string() + "\n```", "",
         "## Cells", "", "| region | train→eval | Δ | outcome | noise Dice | real Dice | R_train | R_eval | gap (paired) |",
         "|---|---|--:|---|--:|--:|--:|--:|--:|"]
    for r in cells.assign(region=pd.Categorical(cells["region"], REGIONS, ordered=True)).sort_values(["region", "R_gap"]).itertuples():
        L.append(f"| {r.region} | {r.train}→{r.eval} | {100*r.delta:+.1f} | {r.outcome} | {r.dice_voronoi:.2f} | "
                 f"{r.dice_realfill:.2f} | {r.R_train:.3f} | {r.R_eval:.3f} | {r.R_gap:+.3f} |")
    (TABLES / "internal_ramp_vs_fill_swap.md").write_text("\n".join(L))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    col = {"success": HELP, "failure": HURT, "n.s.": NS}
    mk = {"SNFH": "o", "RC": "s", "ET": "^", "NCR": "D"}
    for r in cells.itertuples():
        axes[0].scatter(r.R_eval, r.dice_voronoi, color=col[r.outcome], marker=mk[r.region], s=60, edgecolors="white")
        axes[1].scatter(r.R_gap, 100 * r.delta, color=col[r.outcome], marker=mk[r.region], s=60, edgecolors="white")
        if r.region == "SNFH" or r.outcome != "n.s.":
            axes[1].annotate(f"{r.train}→{r.eval} {r.region}", (r.R_gap, 100 * r.delta), fontsize=7,
                             xytext=(4, 3), textcoords="offset points")
    axes[0].set_xlabel("internal ramp R in the eval contrast"); axes[0].set_ylabel("noise-fill Dice")
    axes[0].set_title(f"P1: τ(noise Dice, −R_eval) = {p1[0]:+.2f} (p={p1[1]:.2g})", fontsize=10)
    axes[1].axhline(0, color="#777", ls="--", lw=1); axes[1].axvline(0, color="#777", ls=":", lw=1)
    axes[1].set_xlabel("ramp gap R_eval − R_train (paired)"); axes[1].set_ylabel("real-fill gain Δ Dice (pts)")
    axes[1].set_title(f"P2: ρ(Δ, gap) = {rho2:+.2f} (p={p2c:.2g}); P4 failures with negative gap: {p4}", fontsize=10)
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", color=HELP, ls="", label="success"),
               plt.Line2D([], [], marker="o", color=HURT, ls="", label="failure"),
               plt.Line2D([], [], marker="o", color=NS, ls="", label="n.s.")] + \
              [plt.Line2D([], [], marker=m, color="#555", ls="", label=k) for k, m in mk.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("H9 — does the region's internal depth ramp explain noise-fill failures and real-fill gains?", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(PLOTS / "internal_ramp_vs_fill_swap.png", dpi=150, bbox_inches="tight")
    print("\n".join(L))


if __name__ == "__main__":
    main()
