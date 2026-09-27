#!/usr/bin/env python
"""
EXPLORATORY (post hoc): combine same-region cross-contrast texture similarity with the
region-vs-surroundings confusion margin, per OOD cell (train, eval, region).

  NGF      symmetric, voxel-paired similarity of the region in train vs eval contrast
  eta_fwd  eta^2(train | eval), directional voxel-paired recoverability
  margin   fingerprint distance(train region, eval surround) - distance(train region, eval region)
           (> 0: learned texture points to the region; < 0: to the surroundings)
Questions: (1) are the measures redundant / opposed / independent across cells?
(2) does the 2x2 split (similar? x confusable?) separate success from failure?
(3) does a combined within-row rank score order the real-fill gain better than either alone?
Both inputs were already inspected against the outcomes, so nothing here is confirmatory.
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


def row_tau(t, col):
    S, n = 0, 0
    for _, g in t.groupby(["train", "region"]):
        if len(g) == 3 and g[col].notna().all():
            S += kendall_S(g[col].to_numpy(), g["delta"].to_numpy())
            n += 1
    return S / (3 * n), pooled_p(S, n), n


def main():
    rs = pd.read_csv(DATA / "region_surround_cells.csv")
    ngf = pd.read_csv(DATA / "ngf_vs_fill_swap_pooled.csv")[["train", "eval", "region", "ngf_region"]]
    eta = pd.read_csv(DATA / "containment_vs_fill_swap_pooled.csv")[["train", "eval", "region", "eta2_train_given_eval"]]
    t = (rs.merge(ngf, on=["train", "eval", "region"]).merge(eta, on=["train", "eval", "region"])
         .rename(columns={"ngf_region": "NGF", "eta2_train_given_eval": "eta_fwd"}))
    # within-row ranks (1 = lowest of the 3 eval contrasts) and a combined score
    for c in ("NGF", "margin", "eta_fwd"):
        t[f"rk_{c}"] = t.groupby(["train", "region"])[c].rank()
    t["combo_NGF_margin"] = t["rk_NGF"] + t["rk_margin"]
    t["combo_eta_margin"] = t["rk_eta_fwd"] + t["rk_margin"]
    t.to_csv(DATA / "combine_similarity_confusion.csv", index=False)

    L = ["# Exploratory: texture similarity × region-vs-surroundings confusion (BraTS2024-glioma)", "",
         "Post hoc — both measures were already inspected against the outcomes. NGF = same-region cross-contrast "
         "texture similarity; η²(train|eval) = directional recoverability; margin > 0 = learned texture points to the "
         "eval region rather than its surroundings. Δ = real-fill − noise-fill Dice.", "",
         "## 1. Redundant, opposed or independent? (Spearman across the 36 cross-contrast cells)", "",
         "| pair | all regions | edema only |", "|---|--:|--:|"]
    e = t[t["region"] == "SNFH"]
    for a, b in [("NGF", "margin"), ("eta_fwd", "margin"), ("NGF", "eta_fwd"), ("NGF", "d_RR")]:
        L.append(f"| {a} vs {b} | {spearmanr(t[a], t[b])[0]:+.2f} | {spearmanr(e[a], e[b])[0]:+.2f} |")

    L += ["", "## 2. 2×2 split per region (similar = NGF above the region's median; confusable = margin < 0)", "",
          "| region | quadrant | cells (Δ Dice pts, outcome) |", "|---|---|---|"]
    quad_rows = []
    for region in REGIONS:
        g = t[t["region"] == region]
        med = g["NGF"].median()
        for sim in (True, False):
            for conf in (False, True):
                q = g[((g["NGF"] > med) == sim) & ((g["margin"] < 0) == conf)]
                name = f"{'similar' if sim else 'dissimilar'} & {'CONFUSABLE' if conf else 'not confusable'}"
                cells = ", ".join(f"{r.train}→{r.eval} ({100*r.delta:+.1f}, {r.outcome})" for r in q.itertuples()) or "—"
                L.append(f"| {region} | {name} | {cells} |")
                for r in q.itertuples():
                    quad_rows.append(dict(region=region, quadrant=name, outcome=r.outcome))
    qd = pd.DataFrame(quad_rows)
    L += ["", "Outcome counts by quadrant (all regions):", "",
          "| quadrant | success | failure | n.s. |", "|---|--:|--:|--:|"]
    for name, g in qd.groupby("quadrant"):
        vc = g["outcome"].value_counts()
        L.append(f"| {name} | {vc.get('success', 0)} | {vc.get('failure', 0)} | {vc.get('n.s.', 0)} |")

    L += ["", "## 3. Within-row ranking vs Δ (τ pooled over 12 train×region rows; exact one-sided p)", "",
          "| score | τ | p |", "|---|--:|--:|"]
    res = {}
    for col in ("NGF", "eta_fwd", "margin", "combo_NGF_margin", "combo_eta_margin"):
        tau, p, n = row_tau(t, col)
        res[col] = (tau, p)
        L.append(f"| {col} | {tau:+.2f} | {p:.3g} |")

    L += ["", "## Edema cells", "", "| train→eval | Δ | outcome | NGF | η²(train\\|eval) | margin | sep_E |",
          "|---|--:|---|--:|--:|--:|--:|"]
    for r in e.sort_values("delta").itertuples():
        L.append(f"| {r.train}→{r.eval} | {100*r.delta:+.1f} | {r.outcome} | {r.NGF:.3f} | {r.eta_fwd:.3f} | "
                 f"{r.margin:+.3f} | {r.sep_E:.3f} |")
    (TABLES / "combine_similarity_confusion.md").write_text("\n".join(L))

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    col = {"success": HELP, "failure": HURT, "n.s.": NS}
    mk = {"success": "^", "failure": "v", "n.s.": "o"}
    for ax, region in zip(axes, REGIONS):
        g = t[t["region"] == region]
        ax.axhline(0, color="#777", ls="--", lw=1)
        ax.axvline(g["NGF"].median(), color="#bbb", ls=":", lw=1)
        for r in g.itertuples():
            ax.scatter(r.NGF, r.margin, s=90 if r.outcome != "n.s." else 50, marker=mk[r.outcome],
                       color=col[r.outcome], edgecolors="white", linewidths=1, zorder=3)
            ax.annotate(f"{r.train}→{r.eval}\n{100*r.delta:+.1f}", (r.NGF, r.margin), fontsize=7,
                        xytext=(5, 3), textcoords="offset points", color="#333")
        ax.set_title(region, fontsize=10)
        ax.set_xlabel("NGF (same-region similarity)", fontsize=8.5)
        ax.set_ylabel("margin (< 0: confusable with surroundings)", fontsize=8.5)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], marker="^", color=HELP, ls="", ms=8, label="real-fill success"),
               plt.Line2D([], [], marker="v", color=HURT, ls="", ms=8, label="real-fill failure"),
               plt.Line2D([], [], marker="o", color=NS, ls="", ms=6, label="n.s.")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Similarity (x) × confusion (y) per cell — exploratory; dotted = region median NGF", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(PLOTS / "combine_similarity_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    main()
