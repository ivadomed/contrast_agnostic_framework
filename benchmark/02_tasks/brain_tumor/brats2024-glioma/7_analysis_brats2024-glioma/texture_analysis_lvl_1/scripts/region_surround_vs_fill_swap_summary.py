#!/usr/bin/env python
"""
Region-vs-surroundings texture confusion vs. the real-fill ablation (see
compute_region_surround_texture.py for the hypothesis and the fingerprint).

Per patient p, cell (train T, eval E, region R), fingerprints A(contrast, mask):
  d_RR    = |A(T,R) - A(E,R)|      learned region texture vs eval region texture
  d_RS    = |A(T,R) - A(E,S)|      learned region texture vs eval SURROUNDINGS
  margin  = d_RS - d_RR            > 0: learned texture points to the right place;
                                   < 0: it looks more like the surroundings (misleading)
  sep_E   = |A(E,R) - A(E,S)|      is the region distinguishable from its surroundings on eval?
  margin_pop: same as margin with A(T,R) replaced by the leave-one-patient-out MEAN over the
              other patients (closer to the population texture the model learned).

PRE-REGISTERED (written before any outcome was looked at, 2026-09-24):
  primary config = highpass, all 3 axes, surroundings = full ring (1,5] voxels.
    (chosen on descriptor dynamic range: raw fingerprints saturate at 0.8-0.94 because
     1-4-voxel correlations are dominated by slow shading/resampling smoothness)
  robustness = raw; in-plane axes only (axes 0,1: removes the t2w/t2f thick-slice S-I
     signature from cross-contrast distances); ring_healthy; ring widths 3 and 8.
  predictions: P1 the 4 significant failures have negative margin or the lowest margin in
     their row; P2 within-row Kendall tau(margin, delta) > 0 (pooled); P3 in edema,
     margin(t2w->t1c) > margin(t2w->t1n).

Outputs: data/region_surround_cells.csv, data/region_surround_robustness.csv,
tables/region_surround_vs_fill_swap.md, plots/region_surround_vs_fill_swap.png
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
from compute_region_surround_texture import LAGS  # noqa: E402
from ranking_within_train import kendall_S, pooled_p  # noqa: E402

OUT = THIS_DIR.parent / "outputs"
DATA, TABLES, PLOTS = OUT / "data", OUT / "tables", OUT / "plots"
REGIONS = ["SNFH", "RC", "ET", "NCR"]
REGION_NAME = {"SNFH": "edema (SNFH)", "RC": "resection cavity (RC)",
               "ET": "enhancing tumor (ET)", "NCR": "necrotic core (NCR)"}
HELP, HURT, NS = "#2f7d6b", "#c0392b", "#9a9a9a"
N_BOOT, SEED = 5000, 0
AXES_ALL, AXES_INPLANE = (0, 1, 2), (0, 1)
PRIMARY = dict(variant="highpass", surround="ring", d_out=5.0, axes=AXES_ALL)
CONFIGS = {
    "primary (highpass, 3 axes, ring 5)": PRIMARY,
    "raw": dict(PRIMARY, variant="raw"),
    "in-plane axes only": dict(PRIMARY, axes=AXES_INPLANE),
    "healthy-only ring": dict(PRIMARY, surround="ring_healthy"),
    "ring width 3": dict(PRIMARY, d_out=3.0),
    "ring width 8": dict(PRIMARY, d_out=8.0),
}


def cols_for(axes):
    return [f"acf_ax{a}_lag{k}" for a in axes for k in LAGS]


def fingerprints(fp: pd.DataFrame, cfg) -> tuple[dict, dict]:
    """{(patient, contrast, region): vector} for the region and for its surroundings."""
    cols = cols_for(cfg["axes"])
    sub = fp[fp["variant"] == cfg["variant"]]
    reg = sub[sub["part"] == "region"]
    sur = sub[(sub["part"] == cfg["surround"]) & (np.isclose(sub["d_out"], cfg["d_out"]))]

    def to_dict(df):
        out = {}
        for key, v in zip(zip(df["patient"], df["contrast"], df["region"]), df[cols].to_numpy(float)):
            if np.all(np.isfinite(v)):
                out[key] = v
        return out
    return to_dict(reg), to_dict(sur)


def per_patient(fp: pd.DataFrame, cells: pd.DataFrame, cfg) -> pd.DataFrame:
    R, S = fingerprints(fp, cfg)
    rows = []
    for _, c in cells.iterrows():
        tr, ev, region = c["train"], c["eval"], c["region"]
        pats = sorted({p for (p, con, r) in R if con == tr and r == region}
                      & {p for (p, con, r) in R if con == ev and r == region}
                      & {p for (p, con, r) in S if con == ev and r == region})
        if not pats:
            continue
        tmpl = np.stack([R[(p, tr, region)] for p in pats])
        for i, p in enumerate(pats):
            a_tr, a_er, a_es = R[(p, tr, region)], R[(p, ev, region)], S[(p, ev, region)]
            loo = (tmpl.sum(0) - tmpl[i]) / (len(pats) - 1) if len(pats) > 1 else a_tr
            d_rr, d_rs = np.linalg.norm(a_tr - a_er), np.linalg.norm(a_tr - a_es)
            rows.append(dict(train=tr, eval=ev, region=region, patient=p, d_RR=d_rr, d_RS=d_rs,
                             margin=d_rs - d_rr, sep_E=np.linalg.norm(a_er - a_es),
                             margin_pop=np.linalg.norm(loo - a_es) - np.linalg.norm(loo - a_er)))
    return pd.DataFrame(rows)


def boot_ci(x, rng):
    x = np.asarray(x, float)
    m = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pooled(pp: pd.DataFrame, cells: pd.DataFrame, rng) -> pd.DataFrame:
    out = []
    for (tr, ev, region), g in pp.groupby(["train", "eval", "region"]):
        lo, hi = boot_ci(g["margin"], rng)
        out.append(dict(train=tr, eval=ev, region=region, n=len(g), margin=g["margin"].mean(),
                        margin_lo=lo, margin_hi=hi, margin_pop=g["margin_pop"].mean(),
                        sep_E=g["sep_E"].mean(), d_RR=g["d_RR"].mean(), d_RS=g["d_RS"].mean()))
    return pd.DataFrame(out).merge(cells, on=["train", "eval", "region"])


def tests(t: pd.DataFrame) -> dict:
    res = {}
    for m in ("margin", "margin_pop", "sep_E", "d_RR"):
        S_tot, n_rows = 0, 0
        for _, g in t.groupby(["train", "region"]):
            if len(g) == 3:
                S_tot += kendall_S(g[m].to_numpy(), g["delta"].to_numpy())
                n_rows += 1
        rho, p_rho = spearmanr(t[m], t["delta"])
        res[m] = dict(tau_rows=S_tot / (3 * n_rows) if n_rows else np.nan, n_rows=n_rows,
                      p_rows=pooled_p(S_tot, n_rows) if n_rows else np.nan, rho_cells=rho, p_cells=p_rho)
    fails = t[t["outcome"] == "failure"]
    p1 = []
    for _, f in fails.iterrows():
        row = t[(t["train"] == f["train"]) & (t["region"] == f["region"])]
        p1.append(bool(f["margin"] < 0 or f["margin"] == row["margin"].min()))
    res["P1"] = f"{sum(p1)}/{len(p1)} failures negative or lowest-in-row"
    e = t[(t["region"] == "SNFH") & (t["train"] == "t2w")].set_index("eval")["margin"]
    res["P3"] = bool(e.get("t1c", np.nan) > e.get("t1n", np.nan))
    return res


def write_md(t, robust, prim_tests, path):
    L = ["# Region-vs-surroundings texture confusion vs. the real-fill ablation (BraTS2024-glioma)", "",
         "Texture fingerprint = normalized spatial autocorrelation (lags 1–4 voxels per axis; exactly invariant to "
         "the per-region scale/offset/sign remaps real-fill applies). **margin** = distance(learned region texture on "
         "the TRAINING contrast, eval SURROUNDINGS) − distance(learned region texture, eval REGION): > 0 the learned "
         "texture points to the right place, < 0 it looks more like the surroundings. **sep_E** = region vs "
         "surroundings distance on the eval contrast. Mean over patients [95% bootstrap CI]. Δ = real-fill − "
         "noise-fill Dice (points). Primary config and predictions were fixed before looking at outcomes.", "",
         "## Pre-registered predictions (primary config)", "",
         f"- P1 (failures negative or lowest margin in row): **{prim_tests['P1']}**",
         f"- P2 (within-row τ(margin, Δ) > 0): **τ = {prim_tests['margin']['tau_rows']:+.2f}**, "
         f"p = {prim_tests['margin']['p_rows']:.3g} over {prim_tests['margin']['n_rows']} rows",
         f"- P3 (edema, margin t2w→t1c > t2w→t1n): **{'holds' if prim_tests['P3'] else 'does NOT hold'}**", "",
         "## Robustness (same tests, each configuration)", "",
         "| configuration | τ(margin,Δ) rows | p | ρ(margin,Δ) cells | τ(margin_pop,Δ) | τ(sep_E,Δ) | P1 | P3 |",
         "|---|--:|--:|--:|--:|--:|---|---|"]
    for _, r in robust.iterrows():
        L.append(f"| {r['config']} | {r['tau_margin']:+.2f} | {r['p_margin']:.3g} | {r['rho_margin']:+.2f} | "
                 f"{r['tau_margin_pop']:+.2f} | {r['tau_sep_E']:+.2f} | {r['P1']} | {'yes' if r['P3'] else 'no'} |")
    L += ["", "## Per cell (primary config)", ""]
    for region in REGIONS:
        sub = t[t["region"] == region].sort_values(["train", "margin"], ascending=[True, False])
        L += [f"### {REGION_NAME[region]}", "",
              "| train→eval | Δ Dice | outcome | margin [CI] | margin_pop | sep_E | d(train R, eval R) | d(train R, eval S) | n |",
              "|---|--:|---|--:|--:|--:|--:|--:|--:|"]
        for _, r in sub.iterrows():
            oc = {"success": "**success**", "failure": "**FAILURE**"}.get(r["outcome"], "n.s.")
            L.append(f"| {r['train']}→{r['eval']} | {100*r['delta']:+.1f} | {oc} | {r['margin']:+.3f} "
                     f"[{r['margin_lo']:+.3f}, {r['margin_hi']:+.3f}] | {r['margin_pop']:+.3f} | {r['sep_E']:.3f} | "
                     f"{r['d_RR']:.3f} | {r['d_RS']:.3f} | {r['n']} |")
        L.append("")
    path.write_text("\n".join(L))
    print(f"Wrote {path}")


def plot(t, path):
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(17, 5.4), sharex=True)
    col = {"success": HELP, "failure": HURT, "n.s.": NS}
    mk = {"success": "^", "failure": "v", "n.s.": "o"}
    for ax, region in zip(axes, REGIONS):
        sub = t[t["region"] == region].sort_values("margin").reset_index(drop=True)
        for yi, r in sub.iterrows():
            c = col[r["outcome"]]
            ax.plot([r["margin_lo"], r["margin_hi"]], [yi, yi], color=c, linewidth=1.6)
            ax.scatter(r["margin"], yi, s=80 if r["outcome"] != "n.s." else 45, marker=mk[r["outcome"]],
                       color=c, edgecolors="white", linewidths=1, zorder=3)
        ax.axvline(0, color="#555", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f"{a}→{b}  ({100*d:+.1f})" for a, b, d in zip(sub["train"], sub["eval"], sub["delta"])],
                           fontsize=8)
        ax.set_title(REGION_NAME[region], fontsize=10)
        ax.set_xlabel("margin  (< 0: learned texture looks like the surroundings)", fontsize=8)
        ax.grid(axis="x", color="#eee")
        ax.tick_params(axis="x", labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], marker="^", color=HELP, ls="", ms=8, label="real-fill success (sig.)"),
               plt.Line2D([], [], marker="v", color=HURT, ls="", ms=8, label="real-fill failure (sig.)"),
               plt.Line2D([], [], marker="o", color=NS, ls="", ms=6, label="n.s.")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Does the texture learned on the training contrast look more like the eval region or its surroundings? "
                 "(primary config; 95% CI; Δ Dice pts in brackets)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    files = sorted(DATA.glob("region_surround_texture_shard*.csv"))
    fp = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {len(fp)} fingerprint rows from {len(files)} shards, {fp['patient'].nunique()} patients")
    sig = pd.read_csv(DATA / "region_fill_swap_significance.csv")
    sig = sig[sig["family"] == "OOD"][["train", "eval", "region", "p_holm", "direction"]]
    sig["outcome"] = sig["direction"].map({"HELPS": "success", "HURTS": "failure"}).fillna("n.s.")
    dice = pd.read_csv(DATA / "ranking_within_train_cells.csv")[["train", "eval", "region", "dice_realfill",
                                                                   "dice_voronoi", "delta"]]
    cells = sig.merge(dice, on=["train", "eval", "region"])

    rng = np.random.default_rng(SEED)
    robust, prim_t, prim_tests = [], None, None
    for name, cfg in CONFIGS.items():
        pp = per_patient(fp, cells, cfg)
        t = pooled(pp, cells, rng)
        te = tests(t)
        robust.append(dict(config=name, tau_margin=te["margin"]["tau_rows"], p_margin=te["margin"]["p_rows"],
                           rho_margin=te["margin"]["rho_cells"], tau_margin_pop=te["margin_pop"]["tau_rows"],
                           tau_sep_E=te["sep_E"]["tau_rows"], P1=te["P1"], P3=te["P3"]))
        if cfg is PRIMARY:
            prim_t, prim_tests = t, te
            pp.to_csv(DATA / "region_surround_per_patient.csv", index=False)
    robust = pd.DataFrame(robust)
    prim_t.to_csv(DATA / "region_surround_cells.csv", index=False)
    robust.to_csv(DATA / "region_surround_robustness.csv", index=False)
    write_md(prim_t, robust, prim_tests, TABLES / "region_surround_vs_fill_swap.md")
    plot(prim_t, PLOTS / "region_surround_vs_fill_swap.png")

    pd.set_option("display.width", 220)
    print(robust.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print({k: v for k, v in prim_tests.items()})
    print()
    print(prim_t[prim_t["region"] == "SNFH"].sort_values(["train", "margin"], ascending=[True, False])[
        ["train", "eval", "delta", "outcome", "n", "margin", "margin_lo", "margin_hi", "margin_pop", "sep_E",
         "d_RR", "d_RS"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
