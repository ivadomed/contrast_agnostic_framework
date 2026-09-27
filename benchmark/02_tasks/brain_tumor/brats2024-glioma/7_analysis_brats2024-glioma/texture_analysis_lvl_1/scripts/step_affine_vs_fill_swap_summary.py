#!/usr/bin/env python
"""
H6 "step vs affine-texture" summary — merges compute_step_affine.py's per-patient measures
with the real-fill/noise-fill ablation Dice and tests the pre-registered predictions below.
See compute_step_affine.py's own docstring for the full measure definitions (M1 step_d,
M2 step_auc, M3 edge_salience, M4 affine_r2) and reuse notes.

PRE-REGISTERED PREDICTIONS (written before computing any outcome, 2026-09-24; copied
verbatim from compute_step_affine.py so this file is self-contained):
  P1: noise-fill Dice tracks eval step (M1 step_d, M2 step_auc, M3 edge_salience): tau > 0.
  P2: real-fill Dice tracks affine_r2 (M4, highpass) MORE than noise-fill Dice does.
  P3: gain Delta = real - noise tracks z(M4 highpass) - z(M1) within rows: tau > 0.
  P4 (edema/SNFH, t2w-trained): step on t1n > step on t1c.
  Primary config: ring width d_out=5, M4 on highpass. Robustness: d_out in {3, 8} for
  P1 (step_d/step_auc only, edge_salience does not depend on d_out); raw (un-highpassed)
  M4 for P2/P3.
  Tests: within each (train, region) row (12 rows total: 3 trains x 4 regions), rank the 3
  held-out eval contrasts and compute Kendall S/tau (ranking_within_train.kendall_S), pooled
  over rows with an exact one-sided permutation p (ranking_within_train.pooled_p) — identical
  machinery to ranking_within_train.py's own NGF/eta2 tests. Also Spearman across all 36
  pooled cells (train != eval, 3 trains x 3 evals x 4 regions).
  Secondary (between-patient, exploratory power is low — 4 tests, Holm-corrected): per-patient
  Spearman of noise-fill Dice vs eval step_d (d_out=5) in cells (t2w->t1n, SNFH),
  (t2w->t1c, SNFH), (t2f->t1n, SNFH), (t1n->t2f, SNFH).

Dice source: ranking_within_train.dice_table() (reused, not re-derived) — it already
restricts to GT-present patients via gt_region_presence.csv (verified: patient_region_deltas.csv
itself is NOT GT-restricted, it has dice=0 rows for GT-absent regions; dice_table() applies the
gt_vox > 0 filter before averaging, which is what this script also uses everywhere else).

Outputs: outputs/data/step_affine_cells.csv, outputs/data/step_affine_robustness.csv,
outputs/data/step_affine_secondary_per_patient.csv, outputs/tables/step_affine_vs_fill_swap.md,
outputs/plots/step_affine_vs_fill_swap.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[6]  # THIS_DIR=scripts; parents: lvl1, 7_analysis, dataset, datasets, repo root
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "datasets" / "00_commun_scripts" / "00_00_utils"))
from ranking_within_train import kendall_S, pooled_p, dice_table, REGIONS, TRAINS  # noqa: E402
from compute_cross_contrast_ngf import load_rung_eval  # noqa: E402
from stat_tests import holm  # noqa: E402

OUT = THIS_DIR.parent / "outputs"
DATA, TABLES, PLOTS = OUT / "data", OUT / "tables", OUT / "plots"
REGION_NAME = {"SNFH": "edema (SNFH)", "RC": "resection cavity (RC)",
               "ET": "enhancing tumor (ET)", "NCR": "necrotic core (NCR)"}
HELP, HURT, NS = "#2f7d6b", "#c0392b", "#9a9a9a"
PRIMARY_D_OUT = 5.0
D_OUTS_ROBUST = (3.0, 8.0)
SECONDARY_CELLS = [("t2w", "t1n"), ("t2w", "t1c"), ("t2f", "t1n"), ("t1n", "t2f")]


# ───────────────────────── loading ─────────────────────────
def load_step() -> pd.DataFrame:
    files = sorted(DATA.glob("step_affine_step_shard*.csv"))
    if not files:
        sys.exit(f"No step_affine_step_shard*.csv in {DATA} — run the shards first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n_before = len(df)
    # A shard killed between its step-append and affine-append re-appends that patient's step
    # rows on the resumed run (done_patients() in compute_step_affine.py only skips a patient
    # once BOTH csvs have it) — dedupe defensively so a partial rerun can't double-weight a
    # patient in the cell means below.
    df = df.drop_duplicates(subset=["patient", "contrast", "region", "d_out"], keep="last")
    if len(df) != n_before:
        print(f"  dropped {n_before - len(df)} duplicate step rows (partial-rerun artifact)")
    print(f"Loaded {len(df)} step rows from {len(files)} shard(s), {df['patient'].nunique()} patients")
    return df


def load_affine() -> pd.DataFrame:
    files = sorted(DATA.glob("step_affine_affine_shard*.csv"))
    if not files:
        sys.exit(f"No step_affine_affine_shard*.csv in {DATA} — run the shards first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["patient", "region", "contrast_a", "contrast_b", "variant"], keep="last")
    if len(df) != n_before:
        print(f"  dropped {n_before - len(df)} duplicate affine rows (partial-rerun artifact)")
    print(f"Loaded {len(df)} affine rows from {len(files)} shard(s), {df['patient'].nunique()} patients")
    return df


def gt_present() -> pd.DataFrame:
    p = pd.read_csv(DATA / "gt_region_presence.csv")
    return p[p["gt_vox"] > 0][["case", "region"]].rename(columns={"case": "patient"})


# ───────────────────────── cell-level aggregation ─────────────────────────
def step_cell_table(step_df: pd.DataFrame, present: pd.DataFrame, d_out: float) -> pd.DataFrame:
    df = step_df[step_df["d_out"] == d_out].merge(present, on=["patient", "region"])
    g = df.groupby(["contrast", "region"]).agg(
        step_d=("step_d", "mean"), step_auc=("step_auc", "mean"), edge_salience=("edge_salience", "mean"),
        n_measure=("step_d", lambda x: int(x.notna().sum()))).reset_index()
    return g.rename(columns={"contrast": "eval"})


def affine_lookup(affine_df: pd.DataFrame, present: pd.DataFrame, variant: str) -> dict:
    df = affine_df[affine_df["variant"] == variant].merge(present, on=["patient", "region"])
    g = df.groupby(["contrast_a", "contrast_b", "region"])["affine_r2"].mean().reset_index()
    lut = {}
    for _, r in g.iterrows():
        lut[(r["contrast_a"], r["contrast_b"], r["region"])] = r["affine_r2"]
        lut[(r["contrast_b"], r["contrast_a"], r["region"])] = r["affine_r2"]
    return lut


def build_cells(step_df: pd.DataFrame, affine_df: pd.DataFrame, present: pd.DataFrame) -> pd.DataFrame:
    cells = dice_table()  # train, eval, region, dice_voronoi, dice_realfill, delta (train != eval)
    step5 = step_cell_table(step_df, present, PRIMARY_D_OUT)
    cells = cells.merge(step5, on=["eval", "region"], how="left")
    aff_hp = affine_lookup(affine_df, present, "highpass")
    aff_raw = affine_lookup(affine_df, present, "raw")
    cells["affine_r2_hp"] = [aff_hp.get((t, e, r), np.nan) for t, e, r in zip(cells["train"], cells["eval"], cells["region"])]
    cells["affine_r2_raw"] = [aff_raw.get((t, e, r), np.nan) for t, e, r in zip(cells["train"], cells["eval"], cells["region"])]

    def zwithin(col):
        return cells.groupby("region")[col].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0)

    cells["z_affine_hp"] = zwithin("affine_r2_hp")
    cells["z_step_d"] = zwithin("step_d")
    cells["combined_score"] = cells["z_affine_hp"] - cells["z_step_d"]

    sig = pd.read_csv(DATA / "region_fill_swap_significance.csv")
    sig = sig[sig["family"] == "OOD"][["train", "eval", "region", "direction"]]
    sig["outcome"] = sig["direction"].map({"HELPS": "success", "HURTS": "failure"}).fillna("n.s.")
    cells = cells.merge(sig[["train", "eval", "region", "outcome"]], on=["train", "eval", "region"], how="left")
    cells["outcome"] = cells["outcome"].fillna("n.s.")
    return cells


# ───────────────────────── row/cell tests ─────────────────────────
def row_test(df: pd.DataFrame, xcol: str, ycol: str) -> dict:
    S_tot, n_rows = 0, 0
    for (_tr, _region), g in df.groupby(["train", "region"]):
        gg = g.dropna(subset=[xcol, ycol])
        if len(gg) == 3:
            S_tot += kendall_S(gg[xcol].to_numpy(), gg[ycol].to_numpy())
            n_rows += 1
    tau = S_tot / (3 * n_rows) if n_rows else float("nan")
    p_rows = pooled_p(S_tot, n_rows) if n_rows else float("nan")
    sub = df.dropna(subset=[xcol, ycol])
    rho, p_cells = spearmanr(sub[xcol], sub[ycol]) if len(sub) >= 3 else (float("nan"), float("nan"))
    return dict(S=S_tot, n_rows=n_rows, tau=tau, p_rows=p_rows, n_cells=len(sub),
                rho_cells=float(rho), p_cells=float(p_cells))


def p4_test(step_df: pd.DataFrame, present: pd.DataFrame, measure: str = "step_d") -> dict:
    """P4 primary = step_d; also called with step_auc / edge_salience (same directional claim
    on the other two "step" measures, reported alongside, not substituted post-hoc)."""
    pres_snfh = present[present["region"] == "SNFH"][["patient"]]
    sub = step_df[(step_df["region"] == "SNFH") & (step_df["d_out"] == PRIMARY_D_OUT)
                  & (step_df["contrast"].isin(["t1n", "t1c"]))]
    sub = sub.merge(pres_snfh, on="patient")
    wide = sub.pivot(index="patient", columns="contrast", values=measure).dropna()
    if len(wide) < 2:
        return dict(measure=measure, n=len(wide), mean_t1n=float("nan"), mean_t1c=float("nan"),
                    mean_diff=float("nan"), p_one_sided=float("nan"), holds=False, direction_correct=False)
    stat, p = wilcoxon(wide["t1n"], wide["t1c"], alternative="greater")
    direction_correct = bool(wide["t1n"].mean() > wide["t1c"].mean())
    return dict(measure=measure, n=len(wide), mean_t1n=float(wide["t1n"].mean()), mean_t1c=float(wide["t1c"].mean()),
                mean_diff=float((wide["t1n"] - wide["t1c"]).mean()), p_one_sided=float(p),
                direction_correct=direction_correct, holds=bool(direction_correct and p < 0.05))


def per_patient_dice_voronoi(train: str, eval_c: str, region: str, present: pd.DataFrame) -> pd.Series:
    df = load_rung_eval(train, "voronoi")
    df = df[(df["label"] == region) & (df["group"] == eval_c)]
    pres = present[present["region"] == region][["patient"]].rename(columns={"patient": "case"})
    df = df.merge(pres, on="case")
    return df.groupby("case")["dice"].mean()


def secondary_check(step_df: pd.DataFrame, present: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tr, ev in SECONDARY_CELLS:
        dice_p = per_patient_dice_voronoi(tr, ev, "SNFH", present)
        step_p = (step_df[(step_df.region == "SNFH") & (step_df.d_out == PRIMARY_D_OUT)
                           & (step_df.contrast == ev)]
                  .set_index("patient")["step_d"])
        merged = pd.concat([dice_p.rename("dice_voronoi"), step_p.rename("step_d")], axis=1).dropna()
        if len(merged) >= 3:
            rho, p_two = spearmanr(merged["step_d"], merged["dice_voronoi"])
            p_one = p_two / 2.0 if rho > 0 else 1.0 - p_two / 2.0
        else:
            rho, p_one = float("nan"), float("nan")
        rows.append(dict(train=tr, eval=ev, region="SNFH", n=len(merged), rho=float(rho),
                          p_one_sided_raw=float(p_one)))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_one_sided_raw"].tolist())
    return df


# ───────────────────────── robustness ─────────────────────────
def robustness(step_df, affine_df, present, cells) -> pd.DataFrame:
    rows = []
    # P1 robustness: ring width
    for d in (PRIMARY_D_OUT,) + D_OUTS_ROBUST:
        step_d_tab = step_cell_table(step_df, present, d)
        c = dice_table().merge(step_d_tab, on=["eval", "region"], how="left")
        for measure in ("step_d", "step_auc"):
            te = row_test(c, measure, "dice_voronoi")
            rows.append(dict(config=f"P1 d_out={int(d)}", measure=measure, **te))
    te = row_test(cells, "edge_salience", "dice_voronoi")
    rows.append(dict(config="P1 (d_out-invariant)", measure="edge_salience", **te))
    # P2 robustness: raw vs highpass, vs realfill AND vs voronoi
    for variant, col in (("highpass", "affine_r2_hp"), ("raw", "affine_r2_raw")):
        for target in ("dice_realfill", "dice_voronoi"):
            te = row_test(cells, col, target)
            rows.append(dict(config=f"P2 {variant} vs {target}", measure=col, **te))
    # P3 robustness: raw vs highpass combined score
    for variant, col in (("highpass", "affine_r2_hp"), ("raw", "affine_r2_raw")):
        cc = cells.copy()
        cc[f"z_{col}"] = cc.groupby("region")[col].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0)
        cc["combined_score_r"] = cc[f"z_{col}"] - cc["z_step_d"]
        te = row_test(cc, "combined_score_r", "delta")
        rows.append(dict(config=f"P3 {variant}", measure="combined_score", **te))
    # M4 "*_core" variant (advisor review, 2026-09-24): restricted to voxels >3 voxels inside
    # the region boundary, to check affine_r2 isn't dominated by a shared boundary step under
    # the highpass kernel. Same P2/P3 tests, core affine_r2 in place of the full-region one.
    aff_hp_core = affine_lookup(affine_df, present, "highpass_core")
    aff_raw_core = affine_lookup(affine_df, present, "raw_core")
    cc = cells.copy()
    cc["affine_r2_hp_core"] = [aff_hp_core.get((t, e, r), np.nan) for t, e, r in zip(cc["train"], cc["eval"], cc["region"])]
    cc["affine_r2_raw_core"] = [aff_raw_core.get((t, e, r), np.nan) for t, e, r in zip(cc["train"], cc["eval"], cc["region"])]
    for variant, col in (("highpass_core", "affine_r2_hp_core"), ("raw_core", "affine_r2_raw_core")):
        for target in ("dice_realfill", "dice_voronoi"):
            te = row_test(cc, col, target)
            rows.append(dict(config=f"P2 {variant} vs {target}", measure=col, **te))
        cc[f"z_{col}"] = cc.groupby("region")[col].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0)
        cc["combined_score_core"] = cc[f"z_{col}"] - cc["z_step_d"]
        te = row_test(cc, "combined_score_core", "delta")
        rows.append(dict(config=f"P3 {variant}", measure="combined_score", **te))
    return pd.DataFrame(rows)


# ───────────────────────── output ─────────────────────────
def write_md(cells, rob, p1, p2r, p2n, p3, p4_list, secondary, diag_rho, diag_p, path):
    L = ["# H6: step vs affine-texture (BraTS2024-glioma)", "",
         "Pre-registered predictions (see compute_step_affine.py / this script's docstrings, written "
         "before any outcome was computed). Primary config: ring width d_out=5, M4 on highpass images. "
         "Dice = mean over GT-present patients, folds 0-2 (via ranking_within_train.dice_table()).", "",
         "**Population caveat:** cell-level Dice averages over all GT-present patients for that "
         "(train, eval, region); the M1-M4 measures are NaN below MIN_VOX=200 voxels (eroded region or "
         "ring too small), which drops more patients for the smaller regions (NCR, ET). See "
         "`n_measure` in the per-cell table below — it can be noticeably smaller than the Dice "
         "population for those regions; treat comparisons there as noisier.", "",
         f"**Diagnostic (not pre-registered):** affine_r2 (highpass) and step_d are correlated across "
         f"the 36 pooled cells (Spearman rho = {diag_rho:+.2f}, p = {diag_p:.3g}) — a highpass sigma=2 "
         "kernel can leave boundary-step residue inside the eroded region, and r^2 is sign-invariant, "
         "so a shared step can inflate affine_r2 as well as step_d. This weakens the independence "
         "P3's z(M4)-z(M1) combination assumes; the `*_core` M4 variant in Robustness (voxels >3 "
         "from the boundary) is the check for this.", "",
         "## P1 — noise-fill Dice tracks eval step (tau > 0 expected)", "",
         "| measure | rows | S | tau | p (rows) | n cells | rho (cells) | p (cells) |",
         "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for name, te in p1.items():
        L.append(f"| {name} | {te['n_rows']} | {te['S']:+d} | {te['tau']:+.2f} | {te['p_rows']:.3g} | "
                 f"{te['n_cells']} | {te['rho_cells']:+.2f} | {te['p_cells']:.3g} |")
    p1_holds = all(te['tau'] > 0 for te in p1.values())
    L += ["", f"**P1 {'holds' if p1_holds else 'does NOT hold'}** for all three measures "
              f"(all tau > 0: {p1_holds})."]

    L += ["", "## P2 — real-fill Dice tracks affine_r2 (highpass) MORE than noise-fill Dice does", "",
          "| target | tau | p (rows) | rho (cells) | p (cells) |",
          "|---|--:|--:|--:|--:|",
          f"| real-fill Dice | {p2r['tau']:+.2f} | {p2r['p_rows']:.3g} | {p2r['rho_cells']:+.2f} | {p2r['p_cells']:.3g} |",
          f"| noise-fill Dice | {p2n['tau']:+.2f} | {p2n['p_rows']:.3g} | {p2n['rho_cells']:+.2f} | {p2n['p_cells']:.3g} |",
          "", "No formal test of the difference between the two taus is applied (none was "
              "pre-registered) — this is a comparison of two point estimates, reported as such.",
          f"**P2 {'supported' if p2r['tau'] > p2n['tau'] else 'NOT supported'} by the point estimates** "
              f"(tau real-fill {p2r['tau']:+.2f} vs noise-fill {p2n['tau']:+.2f})"
              f"{'; both are small/near zero, treat as weak evidence either way' if max(abs(p2r['tau']), abs(p2n['tau'])) < 0.2 else ''}."]

    L += ["", "## P3 — gain Delta = real - noise tracks z(affine_r2 highpass) - z(step_d) within rows", "",
          "| tau | p (rows) | rho (cells) | p (cells) |", "|--:|--:|--:|--:|",
          f"| {p3['tau']:+.2f} | {p3['p_rows']:.3g} | {p3['rho_cells']:+.2f} | {p3['p_cells']:.3g} |",
          "", f"**P3 {'holds' if p3['tau'] > 0 else 'does NOT hold'}** (tau {p3['tau']:+.2f})."]

    L += ["", "## P4 — edema (SNFH), t2w-trained: step on t1n > step on t1c "
              "(primary M1 step_d; M2 step_auc, M3 edge_salience reported alongside)", "",
          "| measure | n | mean(t1n) | mean(t1c) | mean diff | p (one-sided) | verdict |",
          "|---|--:|--:|--:|--:|--:|---|"]
    for p4 in p4_list:
        verdict = ("**holds**" if p4["holds"] else
                   ("direction correct, n.s." if p4["direction_correct"] else "**does NOT hold**"))
        L.append(f"| {p4['measure']} | {p4['n']} | {p4['mean_t1n']:.3f} | {p4['mean_t1c']:.3f} | "
                 f"{p4['mean_diff']:+.3f} | {p4['p_one_sided']:.3g} | {verdict} |")
    L.append("")
    L.append("\"Holds\" = mean(t1n) > mean(t1c) AND one-sided p < 0.05 (pre-registered criterion); "
             "\"direction correct, n.s.\" is reported separately, not folded into \"holds\".")

    L += ["", "## Robustness (P1 ring width; P2/P3 raw vs highpass)", "",
          "| config | measure | rows | tau | p (rows) | rho (cells) |",
          "|---|---|--:|--:|--:|--:|"]
    for _, r in rob.iterrows():
        L.append(f"| {r['config']} | {r['measure']} | {r['n_rows']} | {r['tau']:+.2f} | "
                 f"{r['p_rows']:.3g} | {r['rho_cells']:+.2f} |")

    L += ["", "## Secondary (exploratory, between-patient, Holm over 4 tests)", "",
          "| train->eval | region | n | rho | p (raw one-sided) | p (Holm) |",
          "|---|---|--:|--:|--:|--:|"]
    for _, r in secondary.iterrows():
        L.append(f"| {r['train']}->{r['eval']} | {r['region']} | {r['n']} | {r['rho']:+.2f} | "
                 f"{r['p_one_sided_raw']:.3g} | {r['p_holm']:.3g} |")

    L += ["", "## Per cell (primary config, SNFH first)", ""]
    order = ["SNFH", "RC", "ET", "NCR"]
    for region in order:
        sub = cells[cells["region"] == region].sort_values(["train", "eval"])
        L += [f"### {REGION_NAME[region]}", "",
              "| train->eval | outcome | realfill | noisefill | Delta | step_d | step_auc | "
              "edge_sal | n (measure) | affine_r2 (hp) | affine_r2 (raw) | combined_score |",
              "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for _, r in sub.iterrows():
            oc = {"success": "**success**", "failure": "**FAILURE**"}.get(r["outcome"], "n.s.")
            L.append(f"| {r['train']}->{r['eval']} | {oc} | {100*r['dice_realfill']:.1f} | "
                     f"{100*r['dice_voronoi']:.1f} | {100*r['delta']:+.1f} | "
                     f"{r['step_d']:.3f} | {r['step_auc']:.3f} | {r['edge_salience']:.3f} | "
                     f"{int(r['n_measure']) if pd.notna(r['n_measure']) else 'NA'} | "
                     f"{r['affine_r2_hp']:.3f} | {r['affine_r2_raw']:.3f} | {r['combined_score']:+.2f} |")
        L.append("")
    path.write_text("\n".join(L))
    print(f"Wrote {path}")


REGION_MARKER = {"SNFH": "*", "RC": "s", "ET": "^", "NCR": "o"}
REGION_SIZE = {"SNFH": 130, "RC": 55, "ET": 55, "NCR": 55}


def plot(cells: pd.DataFrame, path: Path):
    col = {"success": HELP, "failure": HURT, "n.s.": NS}
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.8))
    panels = [
        ("step_d", "dice_voronoi", "step_d  (M1, ring d_out=5)", "noise-fill Dice", axes[0]),
        ("affine_r2_hp", "dice_realfill", "affine_r2  (M4, highpass)", "real-fill Dice", axes[1]),
        ("combined_score", "delta", "z(affine_r2 hp) - z(step_d)", "Delta Dice (real - noise)", axes[2]),
    ]
    for xcol, ycol, xlabel, ylabel, ax in panels:
        for region in ("NCR", "RC", "ET", "SNFH"):  # SNFH drawn last / on top, it's the primary region
            for outcome in ("n.s.", "success", "failure"):
                g = cells[(cells["outcome"] == outcome) & (cells["region"] == region)]
                if g.empty:
                    continue
                ax.scatter(g[xcol], g[ycol], c=col[outcome], s=REGION_SIZE[region],
                           marker=REGION_MARKER[region], edgecolors="white", linewidths=0.6, zorder=3,
                           alpha=1.0 if region == "SNFH" else 0.85)
        if ycol == "delta":
            ax.axhline(0, color="#888", linestyle="--", linewidth=0.8)
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.grid(color="#eee")
        ax.tick_params(labelsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    color_handles = [plt.Line2D([], [], marker="o", color=col[k], ls="", ms=8, label=k)
                     for k in ("success", "failure", "n.s.")]
    region_handles = [plt.Line2D([], [], marker=REGION_MARKER[r], color="#555", ls="", ms=9,
                                 label=REGION_NAME[r]) for r in ("SNFH", "RC", "ET", "NCR")]
    leg1 = fig.legend(handles=color_handles, loc="lower center", ncol=3, fontsize=9.5, frameon=False,
                      bbox_to_anchor=(0.30, -0.03), title="outcome (OOD significance)")
    fig.legend(handles=region_handles, loc="lower center", ncol=4, fontsize=9.5, frameon=False,
              bbox_to_anchor=(0.75, -0.03), title="region (marker)")
    fig.add_artist(leg1)
    fig.suptitle("H6 step-vs-affine-texture: does eval step explain noise-fill Dice, and affine_r2 explain "
                 "real-fill Dice / the real-vs-noise gain? (color = outcome, marker = region, SNFH = star)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    step_df = load_step()
    affine_df = load_affine()
    present = gt_present()

    # Deliverables named as requested: outputs/data/step_affine_per_patient*.csv (the merged,
    # deduped per-patient measures — separate from the per-rank shard files, which stay as
    # resumability checkpoints).
    step_df.to_csv(DATA / "step_affine_per_patient_step.csv", index=False)
    affine_df.to_csv(DATA / "step_affine_per_patient_affine.csv", index=False)

    cells = build_cells(step_df, affine_df, present)
    cells.to_csv(DATA / "step_affine_cells.csv", index=False)

    diag_rho, diag_p = spearmanr(cells["affine_r2_hp"], cells["step_d"], nan_policy="omit")

    p1 = {m: row_test(cells, m, "dice_voronoi") for m in ("step_d", "step_auc", "edge_salience")}
    p2r = row_test(cells, "affine_r2_hp", "dice_realfill")
    p2n = row_test(cells, "affine_r2_hp", "dice_voronoi")
    p3 = row_test(cells, "combined_score", "delta")
    p4_list = [p4_test(step_df, present, m) for m in ("step_d", "step_auc", "edge_salience")]
    secondary = secondary_check(step_df, present)
    secondary.to_csv(DATA / "step_affine_secondary_per_patient.csv", index=False)

    rob = robustness(step_df, affine_df, present, cells)
    rob.to_csv(DATA / "step_affine_robustness.csv", index=False)

    write_md(cells, rob, p1, p2r, p2n, p3, p4_list, secondary, float(diag_rho), float(diag_p),
             TABLES / "step_affine_vs_fill_swap.md")
    plot(cells, PLOTS / "step_affine_vs_fill_swap.png")

    pd.set_option("display.width", 220)
    print("\n=== P1 (noise-fill Dice vs eval step) ===")
    for m, te in p1.items():
        print(m, te)
    print("\n=== P2 (affine_r2 hp vs realfill / voronoi) ===")
    print("realfill:", p2r)
    print("voronoi :", p2n)
    print("\n=== P3 (combined score vs delta) ===")
    print(p3)
    print(f"diagnostic rho(affine_r2_hp, step_d) across cells = {diag_rho:+.3f}, p = {diag_p:.3g}")
    print("\n=== P4 (SNFH, t2w-trained: t1n step > t1c step) ===")
    for p4 in p4_list:
        print(p4)
    print("\n=== secondary (per-patient, Holm) ===")
    print(secondary.to_string(index=False))


if __name__ == "__main__":
    main()
