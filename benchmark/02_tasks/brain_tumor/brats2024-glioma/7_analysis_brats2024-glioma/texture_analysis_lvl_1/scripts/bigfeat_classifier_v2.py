#!/usr/bin/env python
"""
bigfeat v2: anti-leakage follow-up to bigfeat_classifier.py (reuses its data builder / metrics).

New CV scheme: leave-one-EVAL-contrast-out (4 groups) next to LOCO(36), leave-one-training-contrast-out(3), and
GroupKFold-by-patient (reference).  Two feature sets, every scheme x model run on both:
  raw    : E_/T_/D_ features as in v1 (311).
  deid   : every base feature minus its population mean for that contrast (mean over TRAINING-fold rows only, pooling
           eval-side rows of that contrast and train-side rows of that contrast); differences recomputed from the
           centred values; whole-brain features dropped.  A contrast with no training rows (e.g. t1c under
           leave-eval-t1c-out) is centred with the label-free mean of the held-out rows' own features.
Regression models HGB / RF / ElasticNet, selection (SelectKBest k=K) + imputation + scaling inside the fold.
Null: cell-level permutation (row delta -> delta - own cell mean + permuted cell mean), whole pipeline re-run.
Interpretation: best model (best LOEO cell Spearman, preferring ones with permutation p<0.05): grouped permutation
importance by feature family and by source on held-out LOEO folds, PDPs of the top 6 features.
Env: BF_NPERM (150) BF_NJOBS (8) BF_K (30).
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigfeat_classifier import build, cv_pred, pval, reg_stats, splits  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs"
NPERM = int(os.environ.get("BF_NPERM", 150))
NJ = int(os.environ.get("BF_NJOBS", 8))
K = int(os.environ.get("BF_K", 30))
SEED = 0
CODE = {"t1n": 0, "t1c": 1, "t2w": 2, "t2f": 3}


class DeID(BaseEstimator, TransformerMixin):
    """Input X = [E base (nb) | T base (nb) | eval code | train code]; output [E' | T' | E'-T'] without brain_ cols."""

    def __init__(self, nb=0, keep=None):
        self.nb = nb
        self.keep = keep

    @staticmethod
    def _mu(E, T, ec, tc, c):
        parts = [E[ec == c], T[tc == c]]
        parts = np.vstack([p for p in parts if len(p)]) if any(len(p) for p in parts) else None
        return None if parts is None else np.nanmean(parts, axis=0)

    def fit(self, X, y=None):
        nb = self.nb
        E, T, ec, tc = X[:, :nb], X[:, nb:2 * nb], X[:, -2], X[:, -1]
        self.mu_ = {c: m for c in range(4) if (m := self._mu(E, T, ec, tc, c)) is not None}
        return self

    def transform(self, X):
        nb = self.nb
        E, T, ec, tc = X[:, :nb].copy(), X[:, nb:2 * nb].copy(), X[:, -2], X[:, -1]
        for c in np.unique(np.concatenate([ec, tc])):
            mu = self.mu_.get(int(c))
            if mu is None:
                mu = self._mu(E, T, ec, tc, c)
            E[ec == c] -= mu
            T[tc == c] -= mu
        E, T = E[:, self.keep], T[:, self.keep]
        return np.hstack([E, T, E - T])


def pipe(model, deid_args=None):
    steps = []
    if deid_args is not None:
        steps.append(("deid", DeID(**deid_args)))
    steps += [("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sc", StandardScaler()),
              ("sel", SelectKBest(f_regression, k=K)), ("m", model)]
    return Pipeline(steps)


def make_models(deid_args):
    return {
        "HGB": lambda: pipe(HistGradientBoostingRegressor(max_depth=3, max_iter=60, learning_rate=0.05, min_samples_leaf=40,
                                                          l2_regularization=5.0, early_stopping=False, random_state=0), deid_args),
        "RF": lambda: pipe(RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=20, max_features=0.3,
                                                 random_state=0), deid_args),
        "ENet": lambda: pipe(ElasticNetCV(l1_ratio=[0.3, 0.7, 1.0], n_alphas=15, cv=3, max_iter=3000), deid_args),
    }


def run_all(sets, y, cidx, ncell, sig_idx, sig_sign, schemes):
    res = {}
    for fs, (X, models) in sets.items():
        for mn, mk in models.items():
            for sn, sps in schemes.items():
                res[(fs, mn, sn)] = reg_stats(cv_pred(mk, X, y, sps), y, cidx, ncell, sig_idx, sig_sign)
    return res


def one_perm(seed, sets, y, cidx, cmean, ncell, sig_idx, sig_sign, schemes):
    pm = np.random.default_rng(seed).permutation(cmean)
    yp = y - cmean[cidx] + pm[cidx]
    return run_all(sets, yp, cidx, ncell, sig_idx, sig_sign, schemes)


def family(n):
    b = n[2:]
    if b.startswith("shape_"):
        return "shape/size"
    if b.startswith("con_") or b.startswith("bnd_"):
        return "region-vs-ring contrast/boundary"
    if "acf_ax2" in b:
        return "through-plane acf (ax2)"
    if any(t in b for t in ("acf", "hp_std", "gm_", "lap_", "glcm", "lbp")):
        return "texture (acf ax0/1, highpass, gradient, GLCM, LBP)"
    return "first-order intensity"


def importance(mk, X, y, names, sps, rng, nrep=5):
    ind, grp = {}, {}
    fam = np.array([family(n) for n in names])
    src = np.array([{"E": "eval", "T": "train", "D": "diff"}[n[0]] for n in names])
    for tr, te in sps:
        m = mk().fit(X[tr], y[tr])
        Z = Pipeline(m.steps[:-2]).transform(X[te]) if len(m.steps) > 4 else Pipeline(m.steps[:-2]).transform(X[te])
        sel = m.named_steps["sel"].get_support()
        idx = np.where(sel)[0]
        Zs = m.named_steps["sel"].transform(Z)
        est = m.named_steps["m"]
        base = np.mean((est.predict(Zs) - y[te]) ** 2)

        def inc(cols):
            out = []
            for _ in range(nrep):
                Zp = Zs.copy()
                perm = rng.permutation(len(Zp))
                Zp[:, cols] = Zp[perm][:, cols]
                out.append(np.mean((est.predict(Zp) - y[te]) ** 2) - base)
            return float(np.mean(out))
        for j, i in enumerate(idx):
            ind.setdefault(names[i], []).append(inc([j]))
        for lab, arr in (("family", fam), ("source", src)):
            for g in np.unique(arr[idx]):
                cols = [j for j, i in enumerate(idx) if arr[i] == g]
                grp.setdefault((lab, g), []).append(inc(cols))
            for g in np.unique(arr):
                if g not in arr[idx]:
                    grp.setdefault((lab, g), []).append(0.0)
    return ind, grp


def main():
    t0 = time.time()
    df, Xraw, names, sig, cells, _ = build()
    ncell = len(cells)
    cidx = df["cell_idx"].to_numpy(int)
    y = df["delta"].to_numpy(float)
    cmean = pd.Series(y).groupby(cidx).mean().reindex(range(ncell)).to_numpy()
    sig_idx = np.where(sig["significant"].to_numpy())[0]
    sig_sign = np.sign(sig["mean_delta_pts"].to_numpy(float)[sig_idx])
    base = [n[2:] for n in names if n.startswith("E_") and ("T_" + n[2:]) in names]
    nb = len(base)
    XA = np.hstack([df[["E_" + b for b in base]].to_numpy(float), df[["T_" + b for b in base]].to_numpy(float),
                    df["eval"].map(CODE).to_numpy(float)[:, None], df["train"].map(CODE).to_numpy(float)[:, None]])
    keep = np.array([not b.startswith("brain_") for b in base])
    kb = [b for b, k in zip(base, keep) if k]
    names_d = ["E_" + b for b in kb] + ["T_" + b for b in kb] + ["D_" + b for b in kb]
    deid_args = dict(nb=nb, keep=keep)
    sets = {"raw": (Xraw, make_models(None)), "deid": (XA, make_models(deid_args))}
    schemes = {"LOCO(36 cells)": splits(cidx), "LO-train-contrast(3)": splits(df["train"]),
               "LO-EVAL-contrast(4)": splits(df["eval"]),
               "GroupKFold-patient(5) [ref]": [(a, b) for a, b in GroupKFold(5).split(Xraw, y, df["patient"])]}
    print(f"rows={len(df)} raw feats={Xraw.shape[1]} deid feats={len(names_d)} perms={NPERM}", flush=True)

    obs = run_all(sets, y, cidx, ncell, sig_idx, sig_sign, schemes)
    print(f"observed done {time.time()-t0:.0f}s", flush=True)
    nulls = Parallel(n_jobs=NJ)(delayed(one_perm)(SEED + i, sets, y, cidx, cmean, ncell, sig_idx, sig_sign, schemes)
                                for i in range(NPERM))
    print(f"perms done {time.time()-t0:.0f}s", flush=True)

    L = ["# BraTS fill-swap big-feature regression v2: leave-eval-contrast-out + de-identified features", "",
         f"Run {time.strftime('%Y-%m-%d')}. {len(df)} rows, {ncell} cells, 14 Holm-significant (10 helps / 4 fails; trivial "
         f"'always helps' sign acc 0.71, 0 failures, AUC 0.5). raw = {Xraw.shape[1]} features; deid = {len(names_d)} features "
         f"(per-contrast centring fitted on training-fold rows; whole-brain dropped). {NPERM} cell-level permutations, whole "
         f"pipeline (centring, selection k={K}, model) re-run. p = (1+#null>=obs)/(N+1), one-sided.", "",
         "| features | model | CV | row Spearman | cell Spearman | p | sign acc (14) | p | failures caught (/4) | p | AUC fail-vs-help | p |",
         "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    pres = {}
    for (fs, mn, sn), s in obs.items():
        pp = {k: pval([n[(fs, mn, sn)][k] for n in nulls], s[k]) for k in ("rho", "acc", "fc", "auc")}
        pres[(fs, mn, sn)] = pp
        L.append(f"| {fs} | {mn} | {sn} | {s['row_rho']:+.2f} | {s['rho']:+.2f} | {pp['rho']:.3f} | {s['acc']:.2f} | {pp['acc']:.3f} | "
                 f"{s['fc']} | {pp['fc']:.3f} | {s['auc']:.2f} | {pp['auc']:.3f} |")
    L += ["", "Best-of-3-models cell Spearman per feature set and scheme (null = same max over models):", ""]
    for fs in sets:
        for sn in schemes:
            o = max(obs[(fs, m, sn)]["rho"] for m in sets[fs][1])
            pn = pval([max(n[(fs, m, sn)]["rho"] for m in sets[fs][1]) for n in nulls], o)
            L.append(f"- {fs} / {sn}: max cell Spearman {o:+.2f}, p = {pn:.3f}")
    L.append("")

    # best model: LOEO, prefer p<0.05
    cands = [(fs, mn) for fs in sets for mn in sets[fs][1]]
    surv = [c for c in cands if pres[(*c, "LO-EVAL-contrast(4)")]["rho"] < 0.05]
    pool = surv if surv else cands
    best = max(pool, key=lambda c: obs[(*c, "LO-EVAL-contrast(4)")]["rho"])
    fsb, mnb = best
    L += [f"## Interpretation model: {fsb} / {mnb} ({'survives' if surv else 'NO model survives'} LOEO permutation null at p<0.05; "
          f"LOEO cell Spearman {obs[(fsb, mnb, 'LO-EVAL-contrast(4)')]['rho']:+.2f}, p={pres[(fsb, mnb, 'LO-EVAL-contrast(4)')]['rho']:.3f})", ""]
    Xb, mkb = sets[fsb][0], sets[fsb][1][mnb]
    nm = names if fsb == "raw" else names_d
    rng = np.random.default_rng(SEED)
    ind, grp = importance(mkb, Xb, y, nm, schemes["LO-EVAL-contrast(4)"], rng)
    L += ["Grouped permutation importance on held-out leave-eval-contrast-out folds (MSE increase when all selected columns of the "
          "group are jointly permuted among held-out rows; mean over 4 folds, 0 when group not selected):", "",
          "| group type | group | MSE increase |", "|---|---|--:|"]
    for (lab, g), v in sorted(grp.items(), key=lambda kv: (kv[0][0], -np.mean(kv[1]))):
        L.append(f"| {lab} | {g} | {np.mean(v):+.3f} |")
    top = sorted(((k, np.mean(v), len(v)) for k, v in ind.items()), key=lambda t: -t[1])
    L += ["", "Top individual features (same scheme):", "", "| feature | MSE increase | folds selected (/4) |", "|---|--:|--:|"]
    for k, v, c in top[:12]:
        L.append(f"| {k} | {v:+.3f} | {c} |")
    # PDP on full fit
    m = mkb().fit(Xb, y)
    sel = np.where(m.named_steps["sel"].get_support())[0]
    Z = m.named_steps["sel"].transform(Pipeline(m.steps[:-2]).transform(Xb))
    est = m.named_steps["m"]
    selnames = [nm[i] for i in sel]
    top6 = [k for k, _, _ in top if k in selnames][:6]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for a, k in zip(ax.ravel(), top6):
        j = [nm[i] for i in sel].index(k)
        grid = np.quantile(Z[:, j], np.linspace(0.02, 0.98, 25))
        pdp, lo, hi = [], [], []
        for g in grid:
            Zg = Z.copy()
            Zg[:, j] = g
            pr = est.predict(Zg)
            pc = pd.Series(pr).groupby(cidx).mean().to_numpy()
            pdp.append(pr.mean())
            lo.append(np.percentile(pc, 25))
            hi.append(np.percentile(pc, 75))
        a.plot(grid, pdp, "k")
        a.fill_between(grid, lo, hi, color="grey", alpha=0.3)
        a.plot(Z[:, j], np.full(len(Z), min(lo)), "|", color="b", alpha=0.05)
        a.axhline(0, color="r", lw=0.5)
        a.set_title(k, fontsize=9)
        a.set_xlabel("standardised feature value")
        a.set_ylabel("predicted delta (pts)")
    fig.suptitle(f"PDP, {fsb}/{mnb} fitted on all rows (line = mean, band = IQR across cells' mean predictions)")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "bigfeat_pdp.png", dpi=120)
    (OUT / "tables" / "bigfeat_classifier_v2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
