#!/usr/bin/env python
"""
Experiment A: IBSI-radiomics version of bigfeat_classifier_v2 (within-BraTS fill-swap regression), head-to-head with the
hand-made bigfeat features on IDENTICAL rows / CV splits / permutation seeds.
Sets: hand_raw, hand_deid (bigfeat), rad_tex_{raw,deid} (R, ring, R-ring texture; label-free |Spearman|>0.95 pruning),
      rad_ts_{raw,deid} (texture + R shape).   raw = [E,T,E-T]; deid = per-contrast-centred (fit on training-fold rows).
Models HGB/RF/ENet; selection (top-K |corr|) + mean-imputation + scaling inside each fold (FastSel; same machinery on all
sets).  CV: LOCO, leave-train-contrast-out, leave-EVAL-contrast-out (+ patient GroupKFold reference, observed only).
Null: cell-level permutation (whole pipeline re-run).  Stages: obs | perm --start S --end E | merge.
Env: BF_K (30) BF_NJOBS (8) RAD_PRUNE (0.95)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))
import bigfeat_classifier as bc  # noqa: E402
import bigfeat_classifier_v2 as bv  # noqa: E402
from radiomics_common import DATA, prune_corr  # noqa: E402

K = int(os.environ.get("BF_K", 30))
NJ = int(os.environ.get("BF_NJOBS", 8))
PRUNE = float(os.environ.get("RAD_PRUNE", 0.95))
SEED = 0
CODE = bv.CODE
SCHEMES_OBS = ["LOCO(36 cells)", "LO-train-contrast(3)", "LO-EVAL-contrast(4)", "GroupKFold-patient(5) [ref]"]
SCHEMES_NULL = SCHEMES_OBS[:3]


class FastSel:
    """[deid] -> mean-impute -> standardise -> top-K |corr(y)| -> model. All fitted on the training fold only."""

    def __init__(self, model_fn, deid=None, k=K):
        self.model_fn, self.deid, self.k = model_fn, deid, k

    def _pre(self, X, fit):
        if self.deid is not None:
            if fit:
                self.d_ = bv.DeID(**self.deid).fit(X)
            X = self.d_.transform(X)
        if fit:
            mu = np.nanmean(X, 0)
            self.mu_ = np.where(np.isnan(mu), 0.0, mu)
        X = np.where(np.isnan(X), self.mu_, X)
        if fit:
            sd = X.std(0)
            self.sd_ = np.where(sd < 1e-9, 1.0, sd)
        return (X - self.mu_) / self.sd_

    def fit(self, X, y):
        Z = self._pre(X, True)
        yc = (y - y.mean())
        r = np.abs(Z.T @ yc) / (len(y) * (y.std() + 1e-12))
        self.sel_ = np.argsort(-r)[: self.k]
        self.m_ = self.model_fn().fit(Z[:, self.sel_], y)
        return self

    def Zsel(self, X):
        return self._pre(X, False)[:, self.sel_]

    def predict(self, X):
        return self.m_.predict(self.Zsel(X))


MODELS = {
    "HGB": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=60, learning_rate=0.05, min_samples_leaf=40,
                                                 l2_regularization=5.0, early_stopping=False, random_state=0),
    "RF": lambda: RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=20, max_features=0.3, random_state=0),
    "ENet": lambda: ElasticNetCV(l1_ratio=[0.3, 0.7, 1.0], n_alphas=15, cv=3, max_iter=3000),
}


def make_models(deid):
    return {n: (lambda f=f: FastSel(f, deid)) for n, f in MODELS.items()}


# ------------------------------------------------------------------ data
def load_rad():
    sh = sorted(glob.glob(str(DATA / "radiomics_brats_shard*.csv")))
    F = pd.concat([pd.read_csv(s) for s in sh], ignore_index=True).drop_duplicates(["patient", "contrast", "region"], keep="last")
    R = [c for c in F.columns if c.startswith("R__")]
    G = [c for c in F.columns if c.startswith("ring__")]
    S = [c for c in F.columns if c.startswith("S__")]
    D = pd.DataFrame({"RmR__" + c[3:]: F[c].to_numpy() - F["ring__" + c[3:]].to_numpy() for c in R})
    F = pd.concat([F[["patient", "contrast", "region"]], F[R + G + S], D], axis=1)
    return F, R + G + list(D.columns), S


def build():
    df, Xh, names_h, sig, cells, _ = bc.build()
    F, tex, shp = load_rad()
    ix = F.set_index(["patient", "contrast", "region"])
    ke = pd.MultiIndex.from_arrays([df["patient"], df["eval"], df["region"]])
    kt = pd.MultiIndex.from_arrays([df["patient"], df["train"], df["region"]])
    ok = ke.isin(ix.index) & kt.isin(ix.index)
    keep = np.where(ok)[0]
    df, Xh = df.iloc[keep].reset_index(drop=True), Xh[keep]
    ke, kt = ke[keep], kt[keep]
    out = {"df": df, "Xh": Xh, "names_h": names_h, "sig": sig, "cells": cells}
    for nm, cols in (("tex", tex), ("ts", tex + shp)):
        E, T = ix.loc[ke, cols].to_numpy(float), ix.loc[kt, cols].to_numpy(float)
        out[nm] = (E, T, cols)
    return out


def prep_sets(d):
    df = d["df"]
    ec = df["eval"].map(CODE).to_numpy(float)[:, None]
    tc = df["train"].map(CODE).to_numpy(float)[:, None]
    sets, names, info = {}, {}, {}
    # hand-made (identical to bv.main)
    names_h = d["names_h"]
    base = [n[2:] for n in names_h if n.startswith("E_") and ("T_" + n[2:]) in names_h]
    nb = len(base)
    XA = np.hstack([df[["E_" + b for b in base]].to_numpy(float), df[["T_" + b for b in base]].to_numpy(float), ec, tc])
    keep = np.array([not b.startswith("brain_") for b in base])
    kb = [b for b, k in zip(base, keep) if k]
    sets["hand_raw"] = (d["Xh"], make_models(None))
    names["hand_raw"] = names_h
    sets["hand_deid"] = (XA, make_models(dict(nb=nb, keep=keep)))
    names["hand_deid"] = ["E_" + b for b in kb] + ["T_" + b for b in kb] + ["D_" + b for b in kb]
    for nm, label in (("tex", "rad_tex"), ("ts", "rad_ts")):
        E, T, cols = d[nm]
        kidx = prune_corr(np.vstack([E, T]), PRUNE)
        E, T = E[:, kidx], T[:, kidx]
        cs = [cols[i] for i in kidx]
        info[label] = (len(cols), len(cs))
        sets[label + "_raw"] = (np.hstack([E, T, E - T]), make_models(None))
        names[label + "_raw"] = ["E_" + c for c in cs] + ["T_" + c for c in cs] + ["D_" + c for c in cs]
        sets[label + "_deid"] = (np.hstack([E, T, ec, tc]), make_models(dict(nb=len(cs), keep=np.ones(len(cs), bool))))
        names[label + "_deid"] = names[label + "_raw"]
    return sets, names, info


def common(d):
    df, sig = d["df"], d["sig"]
    ncell = len(d["cells"])
    cidx = df["cell_idx"].to_numpy(int)
    y = df["delta"].to_numpy(float)
    cmean = pd.Series(y).groupby(cidx).mean().reindex(range(ncell)).to_numpy()
    sig_idx = np.where(sig["significant"].to_numpy())[0]
    sig_sign = np.sign(sig["mean_delta_pts"].to_numpy(float)[sig_idx])
    allsch = {"LOCO(36 cells)": bc.splits(cidx), "LO-train-contrast(3)": bc.splits(df["train"]),
              "LO-EVAL-contrast(4)": bc.splits(df["eval"]),
              "GroupKFold-patient(5) [ref]": [(a, b) for a, b in GroupKFold(5).split(df, y, df["patient"])]}
    return y, cidx, cmean, ncell, sig_idx, sig_sign, allsch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["obs", "perm", "merge"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    d = build()
    sets, names, info = prep_sets(d)
    y, cidx, cmean, ncell, sig_idx, sig_sign, allsch = common(d)
    print(f"rows={len(d['df'])} patients={d['df']['patient'].nunique()} prune(base kept)={info} "
          f"dims={{{', '.join(f'{k}: {v[0].shape[1]}' for k, v in sets.items())}}} K={K}", flush=True)
    if a.stage == "obs":
        sch = {k: allsch[k] for k in SCHEMES_OBS}
        def one(fs, mn, sn):
            tt = time.time()
            r = bc.reg_stats(bc.cv_pred(sets[fs][1][mn], sets[fs][0], y, sch[sn]), y, cidx, ncell, sig_idx, sig_sign)
            print(f"  {fs}/{mn}/{sn} {time.time()-tt:.0f}s rho={r['rho']:+.2f}", flush=True)
            return (fs, mn, sn), r
        obs = dict(Parallel(n_jobs=NJ)(delayed(one)(fs, mn, sn) for fs in sets for mn in sets[fs][1] for sn in sch))
        pickle.dump(dict(obs=obs, info=info, nrows=len(d["df"]), npat=d["df"]["patient"].nunique()),
                    open(DATA / "radiomics_A_obs.pkl", "wb"))
        print("obs done", time.time() - t0, flush=True)
    elif a.stage == "perm":
        sch = {k: allsch[k] for k in SCHEMES_NULL}
        nulls = Parallel(n_jobs=NJ)(delayed(bv.one_perm)(SEED + i, sets, y, cidx, cmean, ncell, sig_idx, sig_sign, sch)
                                    for i in range(a.start, a.end))
        pickle.dump(nulls, open(DATA / f"radiomics_A_null_{a.start}_{a.end}.pkl", "wb"))
        print("perm done", a.start, a.end, time.time() - t0, flush=True)
    else:
        merge(d, sets, names, y, cidx, ncell, sig_idx, sig_sign, allsch)


# ------------------------------------------------------------------ merge / interpretation
def fam_parts(n):
    src = {"E": "eval", "T": "train", "D": "diff"}[n[0]]
    b = n[2:]
    if "__" not in b:
        return dict(src=src, part="hand", itype="hand", cls="hand")
    part, rest = b.split("__", 1)
    part = {"R": "region", "ring": "ring", "RmR": "region-ring", "S": "shape"}[part]
    itype, cls = rest.split("_")[0], rest.split("_")[1]
    itype = "LoG" if itype.startswith("log") else ("wavelet" if itype.startswith("wavelet") else itype)
    return dict(src=src, part=part, itype=itype, cls=cls)


def grouped_importance(mk, X, y, names, sps, rng, nrep=5):
    ind, grp = {}, {}
    P = [fam_parts(n) for n in names]
    for tr, te in sps:
        m = mk().fit(X[tr], y[tr])
        Zs = m.Zsel(X[te])
        est = m.m_
        base = np.mean((est.predict(Zs) - y[te]) ** 2)
        idx = m.sel_

        def inc(cols):
            o = []
            for _ in range(nrep):
                Zp = Zs.copy()
                Zp[:, cols] = Zs[rng.permutation(len(Zs))][:, cols]
                o.append(np.mean((est.predict(Zp) - y[te]) ** 2) - base)
            return float(np.mean(o))
        for j, i in enumerate(idx):
            ind.setdefault(names[i], []).append(inc([j]))
        for lab in ("src", "part", "itype", "cls"):
            vals = np.array([p[lab] for p in P])
            for g in np.unique(vals):
                cols = [j for j, i in enumerate(idx) if vals[i] == g]
                grp.setdefault((lab, g), []).append(inc(cols) if cols else 0.0)
    return ind, grp


def merge(d, sets, names, y, cidx, ncell, sig_idx, sig_sign, allsch):
    O = pickle.load(open(DATA / "radiomics_A_obs.pkl", "rb"))
    obs = O["obs"]
    nulls = []
    for f in sorted(glob.glob(str(DATA / "radiomics_A_null_*.pkl"))):
        nulls += pickle.load(open(f, "rb"))
    N = len(nulls)
    rows = []
    for (fs, mn, sn), s in obs.items():
        r = dict(set=fs, model=mn, scheme=sn, row_rho=s["row_rho"], rho=s["rho"], acc=s["acc"], fc=s["fc"], auc=s["auc"])
        if sn in SCHEMES_NULL:
            for k in ("rho", "acc", "fc", "auc"):
                r["p_" + k] = bc.pval([n[(fs, mn, sn)][k] for n in nulls], s[k])
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(DATA / "radiomics_A_summary.csv", index=False)
    best, nullmax = [], {}
    for fs in sets:
        for sn in SCHEMES_NULL:
            o = max(obs[(fs, m, sn)]["rho"] for m in sets[fs][1])
            nl = np.array([max(n[(fs, m, sn)]["rho"] for m in sets[fs][1]) for n in nulls])
            best.append(dict(set=fs, scheme=sn, max_rho=o, p=bc.pval(nl, o), null95=float(np.percentile(nl, 95)), N=N))
            nullmax[(fs, sn)] = (o, nl)
    pickle.dump(nullmax, open(DATA / "radiomics_A_nullmax.pkl", "wb"))
    B = pd.DataFrame(best)
    B.to_csv(DATA / "radiomics_A_best.csv", index=False)
    # interpretation: best radiomics set/model by LOEO (prefer permutation-surviving), else best anyway
    loeo = S[(S.scheme == "LO-EVAL-contrast(4)") & S.set.str.startswith("rad")]
    surv = loeo[loeo.p_rho < 0.05]
    pick = (surv if len(surv) else loeo).sort_values("rho", ascending=False).iloc[0]
    fs, mn = pick["set"], pick["model"]
    print("interpretation model", fs, mn, "survives" if len(surv) else "no survivor", flush=True)
    rng = np.random.default_rng(SEED)
    ind, grp = grouped_importance(sets[fs][1][mn], sets[fs][0], y, names[fs], allsch["LO-EVAL-contrast(4)"], rng)
    G = pd.DataFrame([dict(type=k[0], group=k[1], mse_inc=float(np.mean(v))) for k, v in grp.items()])
    I = pd.DataFrame([dict(feature=k, mse_inc=float(np.mean(v)), folds=len(v)) for k, v in ind.items()]).sort_values("mse_inc", ascending=False)
    G.to_csv(DATA / "radiomics_A_importance_groups.csv", index=False)
    I.to_csv(DATA / "radiomics_A_importance_features.csv", index=False)
    pickle.dump(dict(interp=(fs, mn, bool(len(surv))), N=N, nrows=O["nrows"], npat=O["npat"], info=O["info"]),
                open(DATA / "radiomics_A_meta.pkl", "wb"))
    print(B.to_string(), flush=True)


if __name__ == "__main__":
    main()
