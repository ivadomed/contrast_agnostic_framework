#!/usr/bin/env python
"""
Pooled comparison: per-region fill-swap ablation outcome (region_fill_swap_significance.csv)
vs. directional containment eta^2 (compute_correlation_ratio.py), per region x train->eval.

Hypothesis (Paul, 2026-09-24): a train->eval transfer succeeds when what the model learned from
the TRAINING contrast is recoverable from the EVAL contrast. Predictor: eta^2(train | eval)
("training contrast predicted from eval contrast"). Reverse eta^2(eval | train) and the
asymmetry  eta^2(train|eval) - eta^2(eval|train)  are reported alongside.

Primary estimate: Kelley-corrected eps2, raw intensities, K=32 bins; pooled = mean over patients,
95% CI by patient bootstrap (asymmetry CI: paired bootstrap). Robustness: the asymmetry's sign
under every (variant in {raw, highpass}) x (K in {16,32,64}) combination, reported as the
fraction of the 6 combinations agreeing with the primary sign. Healthy-brain eta^2 for the same
ordered pair (same patients) is given as reference.

Outputs: data/containment_vs_fill_swap_pooled.csv, tables/containment_vs_fill_swap_pooled.md,
plots/containment_vs_fill_swap_pooled.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
OUT = THIS_DIR.parent / "outputs"
DATA, TABLES, PLOTS = OUT / "data", OUT / "tables", OUT / "plots"
REGIONS = ["SNFH", "RC", "ET", "NCR"]
REGION_NAME = {"SNFH": "edema (SNFH)", "RC": "resection cavity (RC)",
               "ET": "enhancing tumor (ET)", "NCR": "necrotic core (NCR)"}
HELP, HURT, NS = "#2f7d6b", "#c0392b", "#9a9a9a"
PRIMARY = ("raw", 32)
METRIC = "eps2"
N_BOOT, SEED = 5000, 0


def load_cr() -> pd.DataFrame:
    files = sorted(DATA.glob("correlation_ratio_shard*.csv"))
    if not files:
        raise SystemExit("no correlation_ratio_shard*.csv — run compute_correlation_ratio.py first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def wide(cr, region, variant, K, a, b):
    """Per-patient Series of eta^2(a|b) and eta^2(b|a), aligned on patient."""
    s = cr[(cr["region"] == region) & (cr["variant"] == variant) & (cr["K"] == K)]
    fwd = s[(s["pred"] == a) & (s["cond"] == b)].set_index("patient")[METRIC]
    rev = s[(s["pred"] == b) & (s["cond"] == a)].set_index("patient")[METRIC]
    idx = fwd.index.intersection(rev.index)
    return fwd.loc[idx], rev.loc[idx]


def boot_mean_ci(x, rng):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    m = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def build(cr: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    combos = [(v, k) for v in ("raw", "highpass") for k in (16, 32, 64)]
    rows = []
    for _, s in sig[sig["family"] == "OOD"].iterrows():
        tr, ev, region = s["train"], s["eval"], s["region"]
        fwd, rev = wide(cr, region, *PRIMARY, tr, ev)          # fwd = eta2(train|eval)
        h_fwd, h_rev = wide(cr, "healthy", *PRIMARY, tr, ev)
        h_fwd, h_rev = h_fwd[h_fwd.index.isin(fwd.index)], h_rev[h_rev.index.isin(fwd.index)]
        asym = fwd - rev
        prim_sign = np.sign(asym.mean())
        agree = []
        for v, k in combos:
            f2, r2 = wide(cr, region, v, k, tr, ev)
            if len(f2) >= 5:
                agree.append(np.sign((f2 - r2).mean()) == prim_sign)
        f_lo, f_hi = boot_mean_ci(fwd, rng)
        a_lo, a_hi = boot_mean_ci(asym, rng)
        rows.append(dict(region=region, train=tr, eval=ev,
                         delta_dice_pts=s["mean_delta_pts"], p_holm=s["p_holm"],
                         outcome={"HELPS": "success", "HURTS": "failure"}.get(s["direction"], "n.s."),
                         n=len(fwd),
                         eta2_train_given_eval=fwd.mean(), ci_lo=f_lo, ci_hi=f_hi,
                         eta2_eval_given_train=rev.mean(),
                         asym=asym.mean(), asym_ci_lo=a_lo, asym_ci_hi=a_hi,
                         asym_sign_robust=f"{int(np.sum(agree))}/{len(agree)}",
                         healthy_train_given_eval=h_fwd.mean(), healthy_eval_given_train=h_rev.mean()))
    out = pd.DataFrame(rows)
    out["region"] = pd.Categorical(out["region"], REGIONS, ordered=True)
    return out.sort_values(["region", "eta2_train_given_eval"], ascending=[True, False]).reset_index(drop=True)


def write_md(t: pd.DataFrame, path: Path):
    lines = ["# Real-fill ablation vs. directional containment η² (BraTS2024-glioma)", "",
             "**η²(train | eval)** = fraction of the training contrast's within-region intensity variance "
             "predictable from the eval contrast (correlation ratio, Roche et al. 1998; Kelley bias-corrected, "
             "raw intensities, K=32 equal-frequency bins; mean over patients [95% bootstrap CI]). "
             "Hypothesis: high η²(train|eval) → what the model learned is present at test time → real-fill "
             "succeeds. **asym** = η²(train|eval) − η²(eval|train) [paired 95% CI]; *robust* = how many of the "
             "6 (raw/highpass × K=16/32/64) settings give the same sign. Ablation: rung4 noise-fill → rung5 "
             "real-fill, Δ Dice points, Holm-corrected over 36 OOD cells.", ""]
    for region in REGIONS:
        sub = t[t["region"] == region]
        lines += [f"## {REGION_NAME[region]}", "",
                  "| train→eval | Δ Dice | p (Holm) | outcome | η²(train\\|eval) [CI] | η²(eval\\|train) | asym [CI] | robust | healthy η²(train\\|eval) |",
                  "|---|--:|--:|---|--:|--:|--:|:-:|--:|"]
        for _, r in sub.iterrows():
            oc = {"success": "**success**", "failure": "**FAILURE**"}.get(r["outcome"], "n.s.")
            lines.append(f"| {r['train']}→{r['eval']} | {r['delta_dice_pts']:+.1f} | {r['p_holm']:.2g} | {oc} | "
                         f"{r['eta2_train_given_eval']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | "
                         f"{r['eta2_eval_given_train']:.3f} | {r['asym']:+.3f} [{r['asym_ci_lo']:+.3f}, {r['asym_ci_hi']:+.3f}] | "
                         f"{r['asym_sign_robust']} | {r['healthy_train_given_eval']:.3f} |")
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def plot(t: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(17, 5.4), sharex=True)
    col = {"success": HELP, "failure": HURT, "n.s.": NS}
    mk = {"success": "^", "failure": "v", "n.s.": "o"}
    for ax, region in zip(axes, REGIONS):
        sub = t[t["region"] == region].iloc[::-1]
        for yi, (_, r) in enumerate(sub.iterrows()):
            c = col[r["outcome"]]
            ax.plot([r["eta2_eval_given_train"], r["eta2_train_given_eval"]], [yi, yi],
                    color=c, alpha=0.35, linewidth=3, solid_capstyle="round")
            ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=c, linewidth=1.2)
            ax.scatter(r["eta2_eval_given_train"], yi, s=36, facecolors="white", edgecolors=c,
                       linewidths=1.4, zorder=3)
            ax.scatter(r["eta2_train_given_eval"], yi, s=80 if r["outcome"] != "n.s." else 45,
                       marker=mk[r["outcome"]], color=c, edgecolors="white", linewidths=1, zorder=4)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f"{a}→{b}  ({d:+.1f})" for a, b, d in
                            zip(sub["train"], sub["eval"], sub["delta_dice_pts"])], fontsize=8)
        ax.set_title(REGION_NAME[region], fontsize=10)
        ax.set_xlabel("η² (bias-corrected, pooled)", fontsize=8.5)
        ax.grid(axis="x", color="#eee")
        ax.tick_params(axis="x", labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], marker="^", color=HELP, ls="", ms=8, label="η²(train|eval), real-fill success"),
               plt.Line2D([], [], marker="v", color=HURT, ls="", ms=8, label="η²(train|eval), real-fill failure"),
               plt.Line2D([], [], marker="o", color=NS, ls="", ms=6, label="η²(train|eval), n.s."),
               plt.Line2D([], [], marker="o", color="#555", mfc="white", ls="", ms=6, label="reverse η²(eval|train)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Is the training contrast's structure recoverable from the eval contrast?  "
                 "Filled = η²(train|eval) with 95% CI, hollow = reverse; Δ Dice pts in brackets",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    for d in (DATA, TABLES, PLOTS):
        d.mkdir(parents=True, exist_ok=True)
    cr = load_cr()
    print(f"Loaded {len(cr)} rows, {cr['patient'].nunique()} patients")
    sig = pd.read_csv(DATA / "region_fill_swap_significance.csv")
    t = build(cr, sig)
    t.to_csv(DATA / "containment_vs_fill_swap_pooled.csv", index=False)
    write_md(t, TABLES / "containment_vs_fill_swap_pooled.md")
    plot(t, PLOTS / "containment_vs_fill_swap_pooled.png")
    pd.set_option("display.width", 220)
    print(t[["region", "train", "eval", "delta_dice_pts", "outcome", "n", "eta2_train_given_eval",
             "eta2_eval_given_train", "asym", "asym_ci_lo", "asym_ci_hi", "asym_sign_robust",
             "healthy_train_given_eval"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
