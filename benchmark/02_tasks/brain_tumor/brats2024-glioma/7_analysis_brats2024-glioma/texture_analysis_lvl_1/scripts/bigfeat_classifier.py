#!/usr/bin/env python
"""
Can a LARGE image-feature set predict the BraTS noise->real fill-swap Dice change for UNSEEN (train, eval, region) cells?

Data : outputs/data/bigfeat_shard*.csv (bigfeat_extract.py).  Rows = patient x OOD cell (train != eval, GT-present),
       target = delta_patient = (dice_realfill - dice_voronoi)*100.  Features: eval-contrast, train-contrast (same
       patient+region) and eval-train difference.  No cell one-hot.
Regression models (selection SelectKBest(f_regression,k) + scaling + imputation INSIDE every fold):
       HistGradientBoosting (shallow), RandomForest, ElasticNetCV.
Classification (rows of the 14 Holm-significant cells, label = sign of the cell outcome): L1-logistic, HGB, RF.
CV (headline = unseen cells): leave-one-cell-out, leave-one-training-contrast-out; reference: GroupKFold by patient.
Null: whole pipeline re-run on N permutations: row delta -> delta - own-cell mean + permuted cell mean
      (classification: cell labels permuted over the 14 cells).
Importance: permutation importance of the best model on held-out folds (6 folds of 6 cells).
Env: BF_NPERM (200) BF_NJOBS (8) BF_K (30).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import glob
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
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "outputs"
D = OUT / "data"
NPERM = int(os.environ.get("BF_NPERM", 200))
REG_ENABLED = os.environ.get("BF_SKIP_REG", "0") != "1"
NJ = int(os.environ.get("BF_NJOBS", 8))
K = int(os.environ.get("BF_K", 30))
SEED = 0
DROP = {"patient", "contrast", "region", "n_R", "n_ring", "brain_mean", "brain_std"}


# ---------------------------------------------------------------- data
def build():
    sh = sorted(glob.glob(str(D / "bigfeat_shard*.csv")))
    sh = [s for s in sh if "bigfeat_test" not in s]
    F = pd.concat([pd.read_csv(s) for s in sh], ignore_index=True).drop_duplicates(["patient", "contrast", "region"], keep="last")
    fcols = [c for c in F.columns if c not in DROP]
    sig = pd.read_csv(D / "region_fill_swap_significance.csv")
    sig = sig[sig["train"] != sig["eval"]].reset_index(drop=True)
    pr = pd.read_csv(D / "patient_region_deltas.csv").rename(columns={"case": "patient"})
    pres = pd.read_csv(D / "gt_region_presence.csv").rename(columns={"case": "patient"})
    pres = pres[pres["gt_vox"] > 0][["patient", "region"]]
    df = pr.merge(sig[["train", "eval", "region"]], on=["train", "eval", "region"]).merge(pres, on=["patient", "region"])
    df["delta"] = (df["dice_realfill"] - df["dice_voronoi"]) * 100.0
    df["cell"] = df["train"] + ">" + df["eval"] + ":" + df["region"]
    Fe = F[["patient", "contrast", "region"] + fcols].rename(columns={c: "E_" + c for c in fcols}).rename(columns={"contrast": "eval"})
    Ft = F[["patient", "contrast", "region"] + fcols].rename(columns={c: "T_" + c for c in fcols}).rename(columns={"contrast": "train"})
    df = df.merge(Fe, on=["patient", "eval", "region"], how="inner").merge(Ft, on=["patient", "train", "region"], how="inner")
    Dm = pd.DataFrame({"D_" + c: df["E_" + c] - df["T_" + c] for c in fcols})
    df = pd.concat([df, Dm], axis=1).reset_index(drop=True)
    names = [p + c for p in ("E_", "T_", "D_") for c in fcols]
    X = df[names].to_numpy(float)
    keep = np.nanstd(X, axis=0) > 1e-9
    names = [n for n, k in zip(names, keep) if k]
    X = X[:, keep]
    cells = list(sig["train"] + ">" + sig["eval"] + ":" + sig["region"])
    df["cell_idx"] = df["cell"].map({c: i for i, c in enumerate(cells)})
    return df, X, names, sig, cells, len(fcols)


# ---------------------------------------------------------------- models
def pipe(model, score):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("sel", SelectKBest(score, k=K)), ("m", model)])


REG = {
    "HGB": lambda: pipe(HistGradientBoostingRegressor(max_depth=3, max_iter=60, learning_rate=0.05, min_samples_leaf=40,
                                                      l2_regularization=5.0, early_stopping=False, random_state=0), f_regression),
    "RF": lambda: pipe(RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=20, max_features=0.3,
                                             random_state=0), f_regression),
    "ENet": lambda: pipe(ElasticNetCV(l1_ratio=[0.3, 0.7, 1.0], n_alphas=15, cv=3, max_iter=3000), f_regression),
}
CLF = {
    "L1-logit": lambda: pipe(LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced"), f_classif),
    "HGB": lambda: pipe(HistGradientBoostingClassifier(max_depth=3, max_iter=60, learning_rate=0.05, min_samples_leaf=40,
                                                       l2_regularization=5.0, early_stopping=False,
                                                       class_weight="balanced", random_state=0), f_classif),
    "RF": lambda: pipe(RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=20, max_features=0.3,
                                              class_weight="balanced", random_state=0), f_classif),
}


def splits(groups):
    g = np.asarray(groups)
    return [(np.where(g != u)[0], np.where(g == u)[0]) for u in pd.unique(g)]


def cv_pred(mk, X, y, sp, proba=False):
    p = np.full(len(y), np.nan)
    for tr, te in sp:
        if proba and len(np.unique(y[tr])) < 2:
            p[te] = float(y[tr].mean())
            continue
        m = mk()
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1] if proba else m.predict(X[te])
    return p


def sp_(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(spearmanr(a, b)[0])


def auc_(score, lab):
    return float(roc_auc_score(lab, score)) if 0 < lab.sum() < len(lab) else 0.5


# ---------------------------------------------------------------- regression experiment
def reg_stats(p, y, cidx, ncell, sig_idx, sig_sign):
    pc = pd.Series(p).groupby(cidx).mean().reindex(range(ncell)).to_numpy()
    yc = pd.Series(y).groupby(cidx).mean().reindex(range(ncell)).to_numpy()
    s = np.sign(pc[sig_idx])
    return dict(row_rho=sp_(p, y), rho=sp_(pc, yc), acc=float((s == sig_sign).mean()),
                fc=int(((s < 0) & (sig_sign < 0)).sum()), auc=auc_(pc[sig_idx], (sig_sign > 0).astype(int)))


def run_reg(X, y, cidx, ncell, sig_idx, sig_sign, schemes):
    return {(mn, sn): reg_stats(cv_pred(mk, X, y, sps), y, cidx, ncell, sig_idx, sig_sign)
            for mn, mk in REG.items() for sn, sps in schemes.items()}


def reg_perm(seed, X, y, cidx, cmean, ncell, sig_idx, sig_sign, schemes):
    pm = np.random.default_rng(seed).permutation(cmean)
    yp = y - cmean[cidx] + pm[cidx]
    return run_reg(X, yp, cidx, ncell, sig_idx, sig_sign, schemes)


# ---------------------------------------------------------------- classification experiment
def clf_stats(p, lab_row, cidx_s, cells_s, lab_cell):
    pc = pd.Series(p).groupby(cidx_s).mean().reindex(cells_s).to_numpy()
    pred = (pc > 0.5).astype(int)
    return dict(acc=float((pred == lab_cell).mean()), fc=int(((pred == 0) & (lab_cell == 0)).sum()),
                auc=auc_(pc, lab_cell), bacc=float(0.5 * ((pred[lab_cell == 1] == 1).mean() + (pred[lab_cell == 0] == 0).mean())))


def run_clf(Xs, lab_row, cidx_s, cells_s, lab_cell, schemes):
    return {(mn, sn): clf_stats(cv_pred(mk, Xs, lab_row, sps, proba=True), lab_row, cidx_s, cells_s, lab_cell)
            for mn, mk in CLF.items() for sn, sps in schemes.items()}


def clf_perm(seed, Xs, cidx_s, cells_s, lab_cell, train_s, schemes):
    pl = np.random.default_rng(seed).permutation(lab_cell)
    mp = dict(zip(cells_s, pl))
    lab_row = np.array([mp[c] for c in cidx_s])
    return run_clf(Xs, lab_row, cidx_s, cells_s, pl, schemes)


def pval(null, obs):
    null = np.asarray(null)
    return (1 + int((null >= obs - 1e-12).sum())) / (len(null) + 1)


# ---------------------------------------------------------------- importance
def perm_importance(mk, X, y, names, groups, rng):
    imp, nsel = {}, {}
    for tr, te in splits(groups):
        m = mk().fit(X[tr], y[tr])
        pre = Pipeline(m.steps[:3])
        sel = m.named_steps["sel"].get_support()
        Xte = pre.transform(X[te])
        Xs = Xte.copy()
        base = np.mean((m.named_steps["m"].predict(Xs) - y[te]) ** 2)
        idxs = np.where(sel)[0]
        for j, jj in enumerate(range(Xte.shape[1])):
            pass
        sel_names = [names[i] for i in idxs]
        for j, nm in enumerate(sel_names):
            incr = []
            for _ in range(5):
                Xp = Xte.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                incr.append(np.mean((m.named_steps["m"].predict(Xp) - y[te]) ** 2) - base)
            imp.setdefault(nm, []).append(np.mean(incr))
    return imp


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    df, X, names, sig, cells, nbase = build()
    ncell = len(cells)
    cidx = df["cell_idx"].to_numpy(int)
    y = df["delta"].to_numpy(float)
    cmean = pd.Series(y).groupby(cidx).mean().reindex(range(ncell)).to_numpy()
    sig_idx = np.where(sig["significant"].to_numpy())[0]
    sig_sign = np.sign(sig["mean_delta_pts"].to_numpy(float)[sig_idx])
    print(f"rows={len(df)} patients={df['patient'].nunique()} cells={ncell} (with rows {len(np.unique(cidx))}) base_feats={nbase} "
          f"total_feats={len(names)} sig={len(sig_idx)} fails={(sig_sign<0).sum()} "
          f"max|cell mean - csv mean|={np.nanmax(np.abs(cmean - sig['mean_delta_pts'].to_numpy())):.3f}", flush=True)
    schemes = {"LOCO(36 cells)": splits(cidx), "LO-train-contrast(3)": splits(df["train"]),
               "GroupKFold-patient(5) [ref]": [(a, b) for a, b in GroupKFold(5).split(X, y, df["patient"])]}
    L = ["# BraTS fill-swap: big-feature classifier/regressor on unseen cells", "",
         f"Run {time.strftime('%Y-%m-%d')}. {len(df)} rows (patient x OOD cell, GT-present), {df['patient'].nunique()} patients, "
         f"{ncell} cells, {len(names)} features ({nbase} base per contrast x eval/train/diff, constants dropped). "
         f"Selection (SelectKBest k={K}), imputation, scaling inside every fold. {NPERM} permutations of the cell-level delta "
         f"(whole pipeline re-run). 14 Holm-significant cells = {int((sig_sign>0).sum())} helps / {int((sig_sign<0).sum())} fails; "
         f"trivial 'always helps' sign accuracy = {np.mean(sig_sign>0):.2f}, failures caught = 0/4.", ""]

    # ---- regression
    obs = run_reg(X, y, cidx, ncell, sig_idx, sig_sign, schemes)
    print(f"reg observed done {time.time()-t0:.0f}s", flush=True)
    nulls = Parallel(n_jobs=NJ)(delayed(reg_perm)(SEED + i, X, y, cidx, cmean, ncell, sig_idx, sig_sign, schemes) for i in range(NPERM))
    print(f"reg perms done {time.time()-t0:.0f}s", flush=True)
    L += ["## A. Regression (target = patient delta, Dice pts); metrics on cell means of held-out predictions", "",
          "| model | CV | row Spearman | cell Spearman | p | sign acc (14) | p | failures caught (/4) | p | AUC fail-vs-help | p |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for (mn, sn), s in obs.items():
        pp = {k: pval([n[(mn, sn)][k] for n in nulls], s[k]) for k in ("rho", "acc", "fc", "auc")}
        L.append(f"| {mn} | {sn} | {s['row_rho']:+.2f} | {s['rho']:+.2f} | {pp['rho']:.3f} | {s['acc']:.2f} | {pp['acc']:.3f} | "
                 f"{s['fc']} | {pp['fc']:.3f} | {s['auc']:.2f} | {pp['auc']:.3f} |")
    L += ["", "Best-of-3-models cell Spearman (null = same max over models in each permutation):", ""]
    for sn in schemes:
        o = max(obs[(m, sn)]["rho"] for m in REG)
        pn = pval([max(n[(m, sn)]["rho"] for m in REG) for n in nulls], o)
        L.append(f"- {sn}: max cell Spearman {o:+.2f}, permutation p = {pn:.3f}")
    L.append("")

    # ---- classification on the 14 sig cells
    rmask = np.isin(cidx, sig_idx)
    Xs, cs = X[rmask], cidx[rmask]
    lab_cell = (sig_sign > 0).astype(int)
    cmap = dict(zip(sig_idx, lab_cell))
    lab_row = np.array([cmap[c] for c in cs])
    trs = df["train"].to_numpy()[rmask]
    cschemes = {"LOCO(14 cells)": splits(cs), "LO-train-contrast(3)": splits(trs)}
    cells_s = list(sig_idx)
    cobs = run_clf(Xs, lab_row, cs, cells_s, lab_cell, cschemes)
    cnulls = Parallel(n_jobs=NJ)(delayed(clf_perm)(SEED + i, Xs, cs, cells_s, lab_cell, trs, cschemes) for i in range(NPERM))
    print(f"clf done {time.time()-t0:.0f}s ({rmask.sum()} rows)", flush=True)
    L += [f"## B. Classification: sign of cell outcome on the 14 significant cells ({int(rmask.sum())} rows; permutation = cell labels permuted)", "",
          "| model | CV | cell acc (thr .5) | p | balanced acc | p | failures caught (/4) | p | AUC | p |", "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for (mn, sn), s in cobs.items():
        pp = {k: pval([n[(mn, sn)][k] for n in cnulls], s[k]) for k in ("acc", "bacc", "fc", "auc")}
        L.append(f"| {mn} | {sn} | {s['acc']:.2f} | {pp['acc']:.3f} | {s['bacc']:.2f} | {pp['bacc']:.3f} | {s['fc']} | {pp['fc']:.3f} | "
                 f"{s['auc']:.2f} | {pp['auc']:.3f} |")
    L.append("")

    # ---- importance (best regression model by LOCO cell Spearman)
    best = max(REG, key=lambda m: obs[(m, "LOCO(36 cells)")]["rho"])
    rng = np.random.default_rng(SEED)
    perm_cells = rng.permutation(ncell)
    g6 = perm_cells[cidx] % 6
    imp = perm_importance(REG[best], X, y, names, g6, rng)
    imp_df = pd.DataFrame([(k, np.mean(v), len(v)) for k, v in imp.items()], columns=["feature", "mean_mse_increase", "folds_selected"])
    imp_df = imp_df.sort_values("mean_mse_increase", ascending=False)
    L += [f"## C. Top features (permutation importance, held-out 6-fold-by-cell, best model by LOCO Spearman = {best}; "
          "importance = MSE increase when the feature is permuted among held-out rows, over folds in which it was selected)", "",
          "| feature | MSE increase | folds selected (/6) |", "|---|--:|--:|"]
    for r in imp_df.head(15).itertuples():
        L.append(f"| {r.feature} | {r.mean_mse_increase:+.3f} | {r.folds_selected} |")
    # LOCO selection frequency
    cnt = {}
    for tr, te in schemes["LOCO(36 cells)"]:
        m = REG[best]().fit(X[tr], y[tr])
        for i in np.where(m.named_steps["sel"].get_support())[0]:
            cnt[names[i]] = cnt.get(names[i], 0) + 1
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[:10]
    L += ["", "Most frequently selected in LOCO folds (of 36): " + ", ".join(f"{k} ({v})" for k, v in top), ""]

    # ---- plot
    pl = {sn: cv_pred(REG[best], X, y, schemes[sn]) for sn in ("LOCO(36 cells)", "LO-train-contrast(3)")}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    col = np.array(["#bbbbbb"] * ncell, dtype=object)
    col[sig_idx[sig_sign > 0]] = "#2a9d8f"
    col[sig_idx[sig_sign < 0]] = "#d62828"
    for a, sn in zip(ax[:2], pl):
        pc = pd.Series(pl[sn]).groupby(cidx).mean().reindex(range(ncell)).to_numpy()
        a.scatter(pc, cmean, c=list(col), s=45, edgecolor="k", linewidth=0.4)
        a.axhline(0, color="k", lw=0.5)
        a.axvline(0, color="k", lw=0.5)
        a.set_xlabel("held-out predicted cell delta (Dice pts)")
        a.set_ylabel("observed cell delta (Dice pts)")
        a.set_title(f"{best}, {sn}\ncell Spearman {obs[(best, sn)]['rho']:+.2f}")
    a = ax[2]
    nl = [n[(best, "LOCO(36 cells)")]["rho"] for n in nulls]
    a.hist(nl, bins=25, color="#999999")
    a.axvline(obs[(best, "LOCO(36 cells)")]["rho"], color="r")
    a.set_xlabel("cell Spearman (LOCO)")
    a.set_title(f"permutation null ({NPERM}), observed in red")
    ax[0].scatter([], [], c="#2a9d8f", label="sig. helps")
    ax[0].scatter([], [], c="#d62828", label="sig. fails")
    ax[0].scatter([], [], c="#bbbbbb", label="n.s.")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    (OUT / "plots").mkdir(exist_ok=True)
    plt.savefig(OUT / "plots" / "bigfeat_classifier.png", dpi=130)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "tables" / "bigfeat_classifier.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
