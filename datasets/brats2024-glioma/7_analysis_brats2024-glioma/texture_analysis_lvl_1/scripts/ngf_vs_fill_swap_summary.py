#!/usr/bin/env python
"""
Side-by-side summary: per-region fill-swap ablation result (rung4 noise-fill -> rung5 real-fill,
from region_fill_swap_significance.py) vs. cross-contrast NGF between the training and eval
contrast in that region, POOLED over patients (the question "is SNFH texture on t2f the same as
SNFH texture on t1n", not a per-patient correlation).

Outputs (outputs/):
  data/ngf_vs_fill_swap_pooled.csv       all 36 OOD cells
  tables/ngf_vs_fill_swap_pooled.md      same, readable, grouped by region
  plots/ngf_vs_fill_swap_pooled.png      pooled NGF per cell (95% CI), coloured by ablation outcome
  plots/ngf_vs_fill_swap_scatter_significant.png   per-patient scatter, significant cells only,
                                                    split into real-fill FAILURE / SUCCESS

NGF pooling: mean over patients of the per-patient region NGF (ngf_all, symmetric), 95% CI by
patient bootstrap. Reference column: NGF of the same contrast pair in healthy brain of the same
patients — how much the two contrasts agree in normal tissue, so a region's NGF can be read as
"more/less agreement than normal tissue". NGF(A,B) = NGF(B,A): train->eval and eval->train
cells share one NGF value by construction.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

THIS_DIR = Path(__file__).resolve().parent
OUT = THIS_DIR.parent / "outputs"
DATA, TABLES, PLOTS = OUT / "data", OUT / "tables", OUT / "plots"

REGIONS = ["SNFH", "RC", "ET", "NCR"]
REGION_NAME = {"SNFH": "edema (SNFH)", "RC": "resection cavity (RC)",
               "ET": "enhancing tumor (ET)", "NCR": "necrotic core (NCR)"}
HELP, HURT, NS = "#2f7d6b", "#c0392b", "#9a9a9a"   # match ladder_ood_common IMPROVE/WORSEN/FLAT
N_BOOT, SEED = 5000, 0


def pair_key(a, b):
    return tuple(sorted((a, b)))


def boot_ci(x, rng):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    means = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_pooled(ngf: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    ngf = ngf.assign(pair=[pair_key(a, b) for a, b in zip(ngf["contrast_a"], ngf["contrast_b"])])
    rows, ci_cache = [], {}
    for _, s in sig[sig["family"] == "OOD"].iterrows():
        pk = pair_key(s["train"], s["eval"])
        reg = ngf[(ngf["region"] == s["region"]) & (ngf["pair"] == pk)]
        healthy = ngf[(ngf["region"] == "healthy") & (ngf["pair"] == pk)
                      & ngf["patient"].isin(reg["patient"])]
        if (s["region"], pk) not in ci_cache:
            ci_cache[(s["region"], pk)] = boot_ci(reg["ngf_all"], rng)
        lo, hi = ci_cache[(s["region"], pk)]
        rows.append(dict(region=s["region"], train=s["train"], eval=s["eval"],
                         delta_dice_pts=s["mean_delta_pts"], p_holm=s["p_holm"],
                         outcome={"HELPS": "success", "HURTS": "failure"}.get(s["direction"], "n.s."),
                         n_dice=int(s["n"]), n_ngf=len(reg),
                         ngf_region=reg["ngf_all"].mean(), ngf_ci_lo=lo, ngf_ci_hi=hi,
                         ngf_healthy=healthy["ngf_all"].mean(),
                         ngf_region_minus_healthy=reg["ngf_all"].mean() - healthy["ngf_all"].mean()))
    out = pd.DataFrame(rows)
    out["region"] = pd.Categorical(out["region"], REGIONS, ordered=True)
    return out.sort_values(["region", "ngf_region"], ascending=[True, False]).reset_index(drop=True)


def write_markdown(pooled: pd.DataFrame, path: Path):
    lines = ["# Real-fill ablation vs. cross-contrast NGF, pooled over patients (BraTS2024-glioma)", "",
             "Ablation: rung4 noise-fill → rung5 real-fill, mean Δ Dice (points), paired Wilcoxon, "
             "patient unit, GT-present patients, Holm over 36 OOD cells. NGF: symmetric texture "
             "similarity between the training and eval contrast inside the region, mean over "
             "patients [95% bootstrap CI]; chance level ≈ 0.33, identical texture = 1. "
             "'healthy' = same contrast pair in healthy brain of the same patients.", ""]
    for region in REGIONS:
        sub = pooled[pooled["region"] == region]
        lines += [f"## {REGION_NAME[region]}", "",
                  "| train→eval | Δ Dice (pts) | p (Holm) | outcome | NGF region [95% CI] | NGF healthy | region − healthy |",
                  "|---|--:|--:|---|--:|--:|--:|"]
        for _, r in sub.iterrows():
            outcome = {"success": "**real-fill success**", "failure": "**real-fill FAILURE**"}.get(r["outcome"], "n.s.")
            lines.append(f"| {r['train']}→{r['eval']} | {r['delta_dice_pts']:+.1f} | {r['p_holm']:.3g} | "
                         f"{outcome} | {r['ngf_region']:.3f} [{r['ngf_ci_lo']:.3f}, {r['ngf_ci_hi']:.3f}] | "
                         f"{r['ngf_healthy']:.3f} | {r['ngf_region_minus_healthy']:+.3f} |")
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def plot_pooled(pooled: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(16, 5.2), sharex=True)
    color = {"success": HELP, "failure": HURT, "n.s.": NS}
    for ax, region in zip(axes, REGIONS):
        sub = pooled[pooled["region"] == region].iloc[::-1]
        y = np.arange(len(sub))
        for yi, (_, r) in zip(y, sub.iterrows()):
            c = color[r["outcome"]]
            ax.plot([r["ngf_ci_lo"], r["ngf_ci_hi"]], [yi, yi], color=c, linewidth=2)
            ax.scatter(r["ngf_region"], yi, s=70 if r["outcome"] != "n.s." else 40, color=c,
                       edgecolors="white", linewidths=1.2, zorder=3,
                       marker={"success": "^", "failure": "v", "n.s.": "o"}[r["outcome"]])
            ax.scatter(r["ngf_healthy"], yi, s=28, marker="|", color="#333", zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{t}→{e}  ({d:+.1f})" for t, e, d in
                            zip(sub["train"], sub["eval"], sub["delta_dice_pts"])], fontsize=8)
        ax.axvline(1 / 3, color="#bbb", linestyle=":", linewidth=1)
        ax.set_title(REGION_NAME[region], fontsize=10)
        ax.set_xlabel("NGF train vs eval contrast (pooled)", fontsize=8.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color="#eee")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], marker="^", color=HELP, linestyle="", markersize=8, label="real-fill success (sig.)"),
               plt.Line2D([], [], marker="v", color=HURT, linestyle="", markersize=8, label="real-fill failure (sig.)"),
               plt.Line2D([], [], marker="o", color=NS, linestyle="", markersize=6, label="n.s."),
               plt.Line2D([], [], marker="|", color="#333", linestyle="", markersize=9, label="healthy-brain NGF, same pair"),
               plt.Line2D([], [], color="#bbb", linestyle=":", label="chance (1/3)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Is the region's texture the same in the training and eval contrast? "
                 "Pooled NGF (95% CI) per cell, labelled with the real-fill ablation outcome "
                 "(Δ Dice pts in brackets)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_scatter_significant(per_patient: pd.DataFrame, pooled: pd.DataFrame, path: Path):
    groups = [("failure", "REAL-FILL FAILURE (real-fill significantly worse)", HURT),
              ("success", "REAL-FILL SUCCESS (real-fill significantly better)", HELP)]
    ncol = 5
    n_rows = [int(np.ceil((pooled["outcome"] == g).sum() / ncol)) for g, _, _ in groups]
    row_h, header_h = 3.7, 0.8
    heights = [row_h * nr + header_h for nr in n_rows]
    fig = plt.figure(figsize=(3.2 * ncol, sum(heights)))
    sfigs = fig.subfigures(len(groups), 1, height_ratios=heights, hspace=0.04)
    for sfig, (g, title, c), nr, h in zip(sfigs, groups, n_rows, heights):
        cells = pooled[pooled["outcome"] == g].sort_values("delta_dice_pts", ascending=(g == "failure"))
        sfig.set_facecolor({"failure": "#fbeeec", "success": "#ecf5f2"}[g])
        sfig.suptitle(title, fontsize=12, fontweight="bold", color=c, y=1 - 0.25 / h)
        axes = np.atleast_1d(sfig.subplots(nr, ncol)).ravel()
        sfig.subplots_adjust(top=1 - (header_h + 0.35) / h, bottom=0.55 / h, left=0.05, right=0.98,
                             hspace=0.85, wspace=0.35)
        for ax in axes[len(cells):]:
            ax.axis("off")
        for ax, (_, r) in zip(axes, cells.iterrows()):
            sub = per_patient[(per_patient["train"] == r["train"]) & (per_patient["eval"] == r["eval"])
                              & (per_patient["region"] == r["region"])]
            x, y = sub["ngf_all"].to_numpy(), 100 * sub["delta_dice"].to_numpy()
            ax.scatter(x, y, s=14, alpha=0.75, color=c, edgecolors="white", linewidths=0.3)
            ax.axhline(0, color="#888", linestyle="--", linewidth=0.8)
            rho, p_rho = spearmanr(x, y)
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept, color="#222", linewidth=1.8 if p_rho < 0.05 else 1.1)
            ax.set_title(f"{r['train']}→{r['eval']}  {r['region']}   n={len(sub)}\n"
                         f"ablation Δ {r['delta_dice_pts']:+.1f} pts (p={r['p_holm']:.2g})\n"
                         f"NGF vs Δ: ρ={rho:+.2f} (p={p_rho:.2g})",
                         fontsize=8.5, fontweight="bold" if p_rho < 0.05 else "normal")
            ax.set_xlabel("NGF (per patient)", fontsize=7.5)
            ax.set_ylabel("Δ Dice (pts)", fontsize=7.5)
            ax.tick_params(labelsize=7)
            ax.set_facecolor("white")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    for d in (DATA, TABLES, PLOTS):
        d.mkdir(parents=True, exist_ok=True)
    ngf = pd.read_csv(DATA / "cross_contrast_ngf_per_patient.csv")
    sig = pd.read_csv(DATA / "region_fill_swap_significance.csv")
    per_patient = pd.read_csv(DATA / "ngf_vs_ladder_per_patient.csv")

    pooled = build_pooled(ngf, sig)
    pooled.to_csv(DATA / "ngf_vs_fill_swap_pooled.csv", index=False)
    print(f"Wrote {DATA / 'ngf_vs_fill_swap_pooled.csv'}")
    write_markdown(pooled, TABLES / "ngf_vs_fill_swap_pooled.md")
    plot_pooled(pooled, PLOTS / "ngf_vs_fill_swap_pooled.png")
    plot_scatter_significant(per_patient, pooled, PLOTS / "ngf_vs_fill_swap_scatter_significant.png")

    pd.set_option("display.width", 200)
    print(pooled[["region", "train", "eval", "delta_dice_pts", "outcome", "ngf_region",
                  "ngf_ci_lo", "ngf_ci_hi", "ngf_healthy", "ngf_region_minus_healthy"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
