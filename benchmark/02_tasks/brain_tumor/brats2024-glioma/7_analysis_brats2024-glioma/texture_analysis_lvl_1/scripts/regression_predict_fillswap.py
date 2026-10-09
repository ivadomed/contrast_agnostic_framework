#!/usr/bin/env python
"""
Can image predictors PREDICT the BraTS noise-fill -> real-fill Dice change (delta)?  (nonlinear / regularised
regression + honest cross-validation + permutation nulls).  Complements predictor_combinations.py (linear LOO, null).

A. CELL level: 36 OOD cells, 7 scorecard predictors (explanation_scorecard.load()) [+ region one-hot as reference].
   ridge (RidgeCV), lasso (LassoCV), random forest (200 trees, depth 3).
   CV: leave-one-cell-out, leave-one-training-contrast-out (3 groups), leave-one-region-out.
   Null: N_PERM_A shuffles of delta across cells, full model+CV pipeline re-run.
B. PATIENT level: (patient x OOD cell) rows, delta_patient = dice_realfill - dice_voronoi (pts), GT-present only,
   per-patient predictors. Models ridge / HistGradientBoosting. CV: GroupKFold(5) by patient, leave-one-cell-out.
   Held-out predictions aggregated to cell means -> cell-level Spearman / sign accuracy on the 14 significant cells.
   Nulls: (a) permute the cell-mean part of delta across cells (within-cell residuals kept), (b) global row shuffle.
   Main = cell-identity-free features; reference = + train/eval/region one-hot.
Env: FS_NPERM_A, FS_NPERM_B (default 500, 200), FS_NPERM_B_REF (100), FS_NJOBS (8), FS_OUT_SUFFIX.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explanation_scorecard import load  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs"
D = OUT / "data"
SUF = os.environ.get("FS_OUT_SUFFIX", "")
NPA = int(os.environ.get("FS_NPERM_A", 500))
NPB = int(os.environ.get("FS_NPERM_B", 200))
NPR = int(os.environ.get("FS_NPERM_B_REF", 100))
NJ = int(os.environ.get("FS_NJOBS", 8))
SEED = 0
PRED7 = {"NGF": "ngf_region_z", "eta2": "eta2_train_given_eval_z", "visibility": "vis_gap", "ramp": "R_gap",
         "margin": "margin", "energy": "E_mean_eval_minus_train", "anatomySI": "SI_gap"}


# ------------------------------------------------------------------ models
def mk_ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))


def mk_lasso():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LassoCV(cv=5, n_alphas=30, max_iter=5000))


def mk_rf():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=2, random_state=0))


def mk_hgb():
    return HistGradientBoostingRegressor(max_depth=3, max_iter=60, learning_rate=0.08, min_samples_leaf=30,
                                         l2_regularization=1.0, early_stopping=False, random_state=0)


def splits_from_groups(g):
    g = np.asarray(g)
    return [(np.where(g != u)[0], np.where(g == u)[0]) for u in pd.unique(g)]


def cv_pred(mk, X, y, splits):
    p = np.full(len(y), np.nan)
    for tr, te in splits:
        m = mk()
        m.fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return p


# ------------------------------------------------------------------ metrics
def sp(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(spearmanr(a, b)[0])


def cell_stats(pc, yc, sig_idx):
    """pc/yc cell-level held-out pred and observed; sig_idx positions of the 14 significant cells."""
    s = np.sign(pc[sig_idx])
    o = np.sign(yc[sig_idx])
    return dict(rho=sp(pc, yc), r2=float(r2_score(yc, pc)), acc=float((s == o).mean()),
                fc=int(((s < 0) & (o < 0)).sum()))


# ------------------------------------------------------------------ A: cell level
def run_A_once(X, y, sig_idx, schemes, models):
    res = {}
    for mn, mk in models.items():
        for sn, sps in schemes.items():
            res[(mn, sn)] = cell_stats(cv_pred(mk, X, y, sps), y, sig_idx) | {}
    return res


def perm_A(seed, X, y, sig_idx, schemes, models):
    yp = np.random.default_rng(seed).permutation(y)
    return run_A_once(X, yp, sig_idx, schemes, models)


# ------------------------------------------------------------------ B: patient level
def build_patient_table(t):
    cells = t[["train", "eval", "region", "mean_delta_pts", "significant", "p_holm"]].copy()
    pr = pd.read_csv(D / "patient_region_deltas.csv").rename(columns={"case": "patient"})
    pres = pd.read_csv(D / "gt_region_presence.csv").rename(columns={"case": "patient"})
    pres = pres[pres["gt_vox"] > 0]
    df = pr.merge(cells[["train", "eval", "region"]], on=["train", "eval", "region"])
    df = df.merge(pres.rename(columns={"case": "patient"}), on=["patient", "region"])
    df["delta"] = (df["dice_realfill"] - df["dice_voronoi"]) * 100.0
    df["log_vox"] = np.log10(df["gt_vox"].astype(float))

    # NGF (symmetric pair)
    ng = pd.read_csv(D / "cross_contrast_ngf_per_patient.csv")
    ngd = {}
    for r in ng.itertuples():
        ngd[(r.patient, r.region, frozenset((r.contrast_a, r.contrast_b)))] = r.ngf_all
    df["ngf"] = [ngd.get((p, rg, frozenset((a, b))), np.nan)
                 for p, rg, a, b in zip(df["patient"], df["region"], df["train"], df["eval"])]
    # eta2 / eps2 (raw, K=32)
    cr = pd.read_csv(D / "correlation_ratio_shard0.csv")
    cr = cr[(cr["variant"] == "raw") & (cr["K"] == 32)].drop_duplicates(["patient", "region", "pred", "cond"], keep="last")
    e2 = {(r.patient, r.region, r.pred, r.cond): (r.eta2, r.eps2) for r in cr.itertuples()}
    for nm, (a, b), k in (("eta2_fwd", ("train", "eval"), 0), ("eta2_rev", ("eval", "train"), 0),
                          ("eps2_fwd", ("train", "eval"), 1), ("eps2_rev", ("eval", "train"), 1)):
        df[nm] = [e2.get((p, rg, x, y_), (np.nan, np.nan))[k]
                  for p, rg, x, y_ in zip(df["patient"], df["region"], df[a], df[b])]
    # confusion margin
    ms = pd.read_csv(D / "region_surround_per_patient.csv").drop_duplicates(["train", "eval", "region", "patient"], keep="last")
    df = df.merge(ms[["train", "eval", "region", "patient", "margin"]], on=["train", "eval", "region", "patient"], how="left")
    # ramp
    rp = pd.read_csv(D / "internal_ramp_patient.csv").drop_duplicates(["patient", "contrast", "region"], keep="last")
    rd = {(r.patient, r.contrast, r.region): r.R_adj for r in rp.itertuples()}
    df["R_eval"] = [rd.get((p, c, rg), np.nan) for p, c, rg in zip(df["patient"], df["eval"], df["region"])]
    df["R_train"] = [rd.get((p, c, rg), np.nan) for p, c, rg in zip(df["patient"], df["train"], df["region"])]
    df["R_gap"] = df["R_eval"] - df["R_train"]
    # visibility step AUC (d_out = 5)
    st = pd.read_csv(D / "step_affine_per_patient_step.csv")
    st = st[st["d_out"] == 5].drop_duplicates(["patient", "contrast", "region"], keep="last")
    sd = {(r.patient, r.contrast, r.region): r.step_auc for r in st.itertuples()}
    df["vis_eval"] = [sd.get((p, c, rg), np.nan) for p, c, rg in zip(df["patient"], df["eval"], df["region"])]
    df["vis_train"] = [sd.get((p, c, rg), np.nan) for p, c, rg in zip(df["patient"], df["train"], df["region"])]
    df["vis_gap"] = df["vis_eval"] - df["vis_train"]
    # texture energy
    an = pd.read_csv(D / "anisotropy_per_patient.csv").drop_duplicates(["patient", "contrast", "region"], keep="last")
    ed = {(r.patient, r.contrast, r.region): r.E_mean for r in an.itertuples()}
    df["E_eval"] = [ed.get((p, c, rg), np.nan) for p, c, rg in zip(df["patient"], df["eval"], df["region"])]
    df["E_train"] = [ed.get((p, c, rg), np.nan) for p, c, rg in zip(df["patient"], df["train"], df["region"])]
    df["E_gap"] = df["E_eval"] - df["E_train"]
    df["cell"] = df["train"] + ">" + df["eval"] + ":" + df["region"]
    return df, cells


FEATS_B = ["ngf", "eta2_fwd", "eta2_rev", "eps2_fwd", "eps2_rev", "margin", "R_eval", "R_train", "R_gap",
           "vis_eval", "vis_train", "vis_gap", "E_eval", "E_train", "E_gap", "log_vox"]


def agg_cells(pred, df, cell_order):
    return pd.Series(pred).groupby(df["cell"].to_numpy()).mean().reindex(cell_order).to_numpy()


def run_B_once(X, y, df, cell_order, sig_idx, cell_y, splitsets, models):
    res = {}
    for mn, mk in models.items():
        for sn, sps in splitsets.items():
            p = cv_pred(mk, X, y, sps)
            pc = agg_cells(p, df, cell_order)
            res[(mn, sn)] = dict(row_rho=sp(p, y), row_r2=float(r2_score(y, p)), **cell_stats(pc, cell_y, sig_idx))
    return res


def perm_B(seed, mode, X, y, df, cell_order, sig_idx, cell_idx, cell_mean, splitsets, models):
    rng = np.random.default_rng(seed)
    if mode == "cell":
        perm_means = rng.permutation(cell_mean)
        yp = y - cell_mean[cell_idx] + perm_means[cell_idx]
        cy = perm_means
    else:
        yp = rng.permutation(y)
        cy = pd.Series(yp).groupby(cell_idx).mean().reindex(range(len(cell_order))).to_numpy()
    return run_B_once(X, yp, df, cell_order, sig_idx, cy, splitsets, models)


def pval(null_vals, obs):
    null_vals = np.asarray(null_vals)
    return (1 + int((null_vals >= obs).sum())) / (len(null_vals) + 1)


def main():
    t0 = time.time()
    t, _ = load()
    for c in ("ngf_region", "eta2_train_given_eval"):
        t[c + "_z"] = t.groupby("region")[c].transform(lambda s: (s - s.mean()) / s.std())
    t = t.reset_index(drop=True)
    y = t["mean_delta_pts"].to_numpy(float)
    sig_idx = np.where(t["significant"].to_numpy())[0]
    sig_sign = np.sign(y[sig_idx])
    fail_n = int((sig_sign < 0).sum())
    print(f"cells={len(t)} sig={len(sig_idx)} failures={fail_n} regions={sorted(t['region'].unique())}", flush=True)
    L = ["# Regression: can image predictors predict the noise->real fill-swap change? (BraTS)", "",
         f"36 OOD cells; 14 Holm-significant (10 helps / {fail_n} fails; always-helps sign accuracy = {np.mean(sig_sign > 0):.2f}). "
         f"sklearn used; permutation nulls re-run the full model+CV pipeline. Run {time.strftime('%Y-%m-%d')}.", ""]

    # ---------------- A
    X7 = t[list(PRED7.values())].to_numpy(float)
    oh = pd.get_dummies(t["region"]).to_numpy(float)
    Xoh = np.column_stack([X7, oh])
    schemes = {"LOCO(cell)": splits_from_groups(np.arange(len(t))),
               "LO-train-contrast": splits_from_groups(t["train"]),
               "LO-region": splits_from_groups(t["region"])}
    modelsA = {"ridge": mk_ridge, "lasso": mk_lasso, "RF(d3,200)": mk_rf}
    A = {}
    for vname, X, sc in (("7 predictors", X7, schemes),
                         ("7 pred + region one-hot", Xoh, {k: v for k, v in schemes.items() if k != "LO-region"})):
        obs = run_A_once(X, y, sig_idx, sc, modelsA)
        nulls = Parallel(n_jobs=NJ)(delayed(perm_A)(SEED + i, X, y, sig_idx, sc, modelsA) for i in range(NPA))
        A[vname] = (obs, nulls)
        print(f"A {vname} done {time.time()-t0:.0f}s", flush=True)

    L += [f"## A. Cell level (36 cells, {NPA} permutations of delta across cells)", "",
          "| features | model | CV | Spearman | R2 | sign acc (14 sig) | failures caught (of 4) | p(Spearman) | p(sign acc) |",
          "|---|---|---|--:|--:|--:|--:|--:|--:|"]
    bestA = {}
    for vname, (obs, nulls) in A.items():
        for (mn, sn), s in obs.items():
            pr = pval([n[(mn, sn)]["rho"] for n in nulls], s["rho"])
            pa = pval([n[(mn, sn)]["acc"] for n in nulls], s["acc"])
            L.append(f"| {vname} | {mn} | {sn} | {s['rho']:+.2f} | {s['r2']:+.2f} | {s['acc']:.2f} | {s['fc']} | {pr:.3f} | {pa:.3f} |")
        bestA[vname] = max(obs.items(), key=lambda kv: kv[1]["rho"])
    L += [""]
    # held-out predictions for plot (A, best per CV scheme of 7-pred variant)
    plotA = {}
    for sn in ("LOCO(cell)", "LO-train-contrast"):
        cand = {mn: cv_pred(mk, X7, y, schemes[sn]) for mn, mk in modelsA.items()}
        bm = max(cand, key=lambda m: sp(cand[m], y))
        plotA[sn] = (bm, cand[bm])

    # ---------------- B
    df, cells = build_patient_table(t)
    cell_order = list(t["train"] + ">" + t["eval"] + ":" + t["region"])
    cidx_map = {c: i for i, c in enumerate(cell_order)}
    df["cell_idx"] = df["cell"].map(cidx_map)
    yB = df["delta"].to_numpy(float)
    cell_idx = df["cell_idx"].to_numpy(int)
    cell_mean = pd.Series(yB).groupby(cell_idx).mean().reindex(range(len(cell_order))).to_numpy()
    chk = sp(cell_mean, y)
    L += ["## B. Patient level", "",
          f"Rows = {len(df)} (patient x OOD cell, GT present), {df['patient'].nunique()} patients, {len(cell_order)} cells. "
          f"Cell means of patient delta vs scorecard cell delta: Spearman {chk:+.3f}, max |diff| "
          f"{np.nanmax(np.abs(cell_mean - y)):.2f} pts (differences = GT-present/fold filtering).", "",
          "Feature coverage (non-missing fraction): " + ", ".join(f"{f} {df[f].notna().mean():.2f}" for f in FEATS_B) +
          ". Missing values median-imputed (ridge) / native (HGB).", ""]
    # sig idx in the same cell order as t
    XB = df[FEATS_B].to_numpy(float)
    ohB = pd.concat([pd.get_dummies(df[c], prefix=c) for c in ("train", "eval", "region")], axis=1).to_numpy(float)
    XBoh = np.column_stack([XB, ohB])
    pat = df["patient"].to_numpy()
    gkf = list(GroupKFold(n_splits=5).split(XB, yB, pat))
    splitsB = {"GroupKFold(5, patient)": gkf, "LOCO(cell)": splits_from_groups(cell_idx)}
    modelsB = {"ridge": mk_ridge, "HGB(d3)": mk_hgb}
    B = {}
    for vname, X, nperm in (("cell-identity-free", XB, NPB), ("+ train/eval/region one-hot (reference)", XBoh, NPR)):
        obs = run_B_once(X, yB, df, cell_order, sig_idx, cell_mean, splitsB, modelsB)
        nl = {}
        for mode in ("cell", "row"):
            nl[mode] = Parallel(n_jobs=NJ)(delayed(perm_B)(SEED + 1000 + i, mode, X, yB, df, cell_order, sig_idx,
                                                          cell_idx, cell_mean, splitsB, modelsB) for i in range(nperm))
        B[vname] = (obs, nl, nperm)
        print(f"B {vname} done {time.time()-t0:.0f}s", flush=True)

    L += ["| features | model | CV | row Spearman | row R2 | cell Spearman | cell R2 | sign acc (14 sig) | failures caught (of 4) | "
          "p cell-Spearman (cell-label perm) | p cell-Spearman (row shuffle) | p row-Spearman (row shuffle) |",
          "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for vname, (obs, nl, nperm) in B.items():
        for (mn, sn), s in obs.items():
            pc = pval([n[(mn, sn)]["rho"] for n in nl["cell"]], s["rho"])
            pr = pval([n[(mn, sn)]["rho"] for n in nl["row"]], s["rho"])
            prr = pval([n[(mn, sn)]["row_rho"] for n in nl["row"]], s["row_rho"])
            L.append(f"| {vname} (n_perm={nperm}) | {mn} | {sn} | {s['row_rho']:+.2f} | {s['row_r2']:+.3f} | {s['rho']:+.2f} | "
                     f"{s['r2']:+.2f} | {s['acc']:.2f} | {s['fc']} | {pc:.3f} | {pr:.3f} | {prr:.3f} |")
    L += [""]

    # permutation importance for best main model (by cell Spearman)
    obs_main = B["cell-identity-free"][0]
    (bmn, bsn), bs = max(obs_main.items(), key=lambda kv: kv[1]["rho"])
    mkb = modelsB[bmn]
    imp = np.zeros(len(FEATS_B))
    for tr, te in splitsB[bsn]:
        m = mkb()
        m.fit(XB[tr], yB[tr])
        pi = permutation_importance(m, XB[te], yB[te], scoring="r2", n_repeats=5, random_state=0, n_jobs=1)
        imp += pi.importances_mean / len(splitsB[bsn])
    order = np.argsort(-imp)
    L += [f"### Permutation importance on held-out folds — best main model by cell Spearman: {bmn}, {bsn} "
          f"(cell Spearman {bs['rho']:+.2f}, row R2 {bs['row_r2']:+.3f}); mean drop in held-out R2 when the feature is shuffled", "",
          "| feature | importance |", "|---|--:|"] + [f"| {FEATS_B[i]} | {imp[i]:+.4f} |" for i in order] + [""]

    # ---------------- plot
    lab = np.where(y < 0, 0, 1)
    sigmask = t["significant"].to_numpy()
    out_c = np.where(sigmask & (y < 0), "#d1584c", np.where(sigmask, "#4c9f70", "#9a9a9a"))

    def panel(ax, pc, title):
        ax.scatter(pc, y, c=out_c, s=np.where(sigmask, 55, 28), edgecolor="k", lw=0.4, zorder=3)
        ax.axhline(0, c="#bbb", lw=0.8); ax.axvline(0, c="#bbb", lw=0.8)
        lo, hi = min(pc.min(), y.min()), max(pc.max(), y.max())
        ax.plot([lo, hi], [lo, hi], ls=":", c="#888", lw=0.8)
        ax.set_xlabel("held-out predicted delta (pts)"); ax.set_ylabel("observed delta (pts)")
        ax.set_title(f"{title}\nSpearman {sp(pc, y):+.2f}", fontsize=9)

    pB = {}
    for sn in splitsB:
        cand = {mn: agg_cells(cv_pred(mk, XB, yB, splitsB[sn]), df, cell_order) for mn, mk in modelsB.items()}
        bm = max(cand, key=lambda m: sp(cand[m], y))
        pB[sn] = (bm, cand[bm])
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 9))
    for ax, (sn, (bm, pc)) in zip(axs[0], plotA.items()):
        panel(ax, pc, f"A cell-level, 7 predictors, {bm}, CV={sn}")
    for ax, (sn, (bm, pc)) in zip(axs[1], pB.items()):
        panel(ax, pc, f"B patient-level (cell-identity-free), {bm}, CV={sn}\n(held-out preds averaged per cell)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc=c, mec="k", label=l) for c, l in
                        (("#d1584c", "significant failure (4)"), ("#4c9f70", "significant success (10)"),
                         ("#9a9a9a", "not significant (22)"))], loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "plots" / f"regression_predict_fillswap{SUF}.png", dpi=150)

    L += [f"Plot: plots/regression_predict_fillswap{SUF}.png. Runtime {time.time()-t0:.0f}s."]
    (OUT / "tables" / f"regression_predict_fillswap{SUF}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
