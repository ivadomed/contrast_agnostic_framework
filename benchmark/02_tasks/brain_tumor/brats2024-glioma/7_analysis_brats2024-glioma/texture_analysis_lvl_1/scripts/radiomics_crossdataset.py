#!/usr/bin/env python
"""
Experiment B: IBSI-radiomics version of cross_dataset_fillswap_model.py.  Cells = xd.dev_cells() (69 dev cells, same cells),
feature of a cell = eval-key mean minus train-key mean of radiomics features (R, ring, R-ring [, R shape]); labels pooled
voxel-weighted per case, then mean over cases (BraTS: mean over patients, per contrast x region).
Simple models, selection INSIDE each leave-one-dataset-out fold: sign1 (best single feature by |Spearman|, score = sign*x/sd),
enet10 (top-10 |Spearman| -> ElasticNetCV), logit10 (top-10 -> L1-logistic).  Feature sets tex / ts(tex+shape).
Selection-aware permutation null: (delta,p) shuffled jointly over cells, whole search (set x model) re-run, max statistic.
Same-cell/same-permutation baseline: the hand-made 7-feature xd model family.
Rule (frozen before Open-MS features exist): best pooled LODO Spearman over the 6 (set x model) candidates is refit on all dev
cells -> outputs/data/radiomics_xds_frozen.json.  Stage `openms` (separate job, after open-ms extraction) scores it once.
Stages: fit | openms.   Env: RAD_NPERM (200) RAD_NJOBS (8) RAD_PRUNE (0.95)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob
import json
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import ElasticNetCV, LogisticRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radiomics_common import DATA, prune_corr  # noqa: E402
import cross_dataset_fillswap_model as xd  # noqa: E402

NPERM = int(os.environ.get("RAD_NPERM", 200))
NJ = int(os.environ.get("RAD_NJOBS", 8))
PRUNE = float(os.environ.get("RAD_PRUNE", 0.95))
KSEL = 10


# ------------------------------------------------------------------ features
def key_means(task):
    sh = sorted(glob.glob(str(DATA / f"radiomics_{task}_shard*.csv")))
    sh = [s for s in sh if "_TEST" not in s]
    F = pd.concat([pd.read_csv(s) for s in sh], ignore_index=True)
    R = [c for c in F.columns if c.startswith("R__")]
    G = [c for c in F.columns if c.startswith("ring__")]
    S = [c for c in F.columns if c.startswith("S__")]
    D = pd.DataFrame({"RmR__" + c[3:]: F[c].to_numpy() - F["ring__" + c[3:]].to_numpy() for c in R})
    F = pd.concat([F.drop(columns=[c for c in F.columns if c.startswith("diag")]), D], axis=1)
    cols = R + G + list(D.columns) + S
    if task == "brats":
        F["key"] = "brats:" + F["contrast"] + ":" + F["region"]
        pc = F.groupby("key")[cols].mean()
        n = F.groupby("key")["patient"].nunique()
    else:
        F = F.drop_duplicates(["key", "case", "label"], keep="last")
        w = F["n_R"].to_numpy(float)
        Fw = F[cols].mul(w, axis=0)
        num = Fw.groupby([F["key"], F["case"]]).sum(min_count=1)
        den = pd.Series(w, index=F.index).groupby([F["key"], F["case"]]).sum()
        percase = num.div(den, axis=0)
        pc = percase.groupby(level=0).mean()
        n = percase.groupby(level=0).size()
    return pc, n, cols


def group_of(c):
    return "shape" if c.startswith("S__") else "texture"


# ------------------------------------------------------------------ models (selection inside fold)
def _spearman_cols(Xtr, ytr):
    mu = np.nanmean(Xtr, 0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    X = np.where(np.isnan(Xtr), mu, Xtr)
    Rx = rankdata(X, axis=0)
    Rx = Rx - Rx.mean(0)
    Ry = rankdata(ytr)
    Ry = Ry - Ry.mean()
    den = np.sqrt((Rx ** 2).sum(0)) * np.sqrt((Ry ** 2).sum()) + 1e-12
    return (Rx.T @ Ry) / den, mu


def fit_model(kind, Xtr, ytr):
    r, mu = _spearman_cols(Xtr, ytr)
    X = np.where(np.isnan(Xtr), mu, Xtr)
    if kind == "sign1":
        j = int(np.argmax(np.abs(r)))
        sd = X[:, j].std() + 1e-12
        return dict(kind=kind, idx=[j], mu=mu[[j]], sd=[sd], coef=[float(np.sign(r[j]) or 1.0)], b=0.0, scaleonly=True)
    idx = np.argsort(-np.abs(r))[:KSEL]
    sd = X[:, idx].std(0) + 1e-12
    Z = (X[:, idx] - mu[idx]) / sd
    if kind == "enet10":
        m = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0], n_alphas=20, cv=5, max_iter=5000).fit(Z, ytr)
        return dict(kind=kind, idx=list(idx), mu=mu[idx], sd=sd, coef=list(m.coef_), b=float(m.intercept_), scaleonly=False)
    pos = ytr > 0
    if pos.all() or (~pos).all():
        return dict(kind=kind, idx=list(idx), mu=mu[idx], sd=sd, coef=[0.0] * len(idx), b=1.0 if pos.all() else -1.0, scaleonly=False)
    m = LogisticRegression(penalty="l1", solver="liblinear", C=0.3, class_weight="balanced").fit(Z, pos.astype(int))
    return dict(kind=kind, idx=list(idx), mu=mu[idx], sd=sd, coef=list(m.coef_[0]), b=float(m.intercept_[0]), scaleonly=False)


def score_model(M, X):
    idx = M["idx"]
    Xs = np.where(np.isnan(X[:, idx]), np.asarray(M["mu"]), X[:, idx])
    if M["scaleonly"]:  # zero-centred relative feature: only scale, no re-centring
        return M["coef"][0] * Xs[:, 0] / M["sd"][0]
    return ((Xs - np.asarray(M["mu"])) / np.asarray(M["sd"])) @ np.asarray(M["coef"]) + M["b"]


CANDS = [(s, k) for s in ("tex", "ts") for k in ("sign1", "enet10", "logit10")]


def lodo(Xs, y, grp):
    """Xs: {set: matrix}. returns {(set,kind): (oof, [selected idx per fold])}"""
    out = {}
    for s, k in CANDS:
        oof = np.full(len(y), np.nan)
        sel = []
        for g in np.unique(grp):
            te = grp == g
            M = fit_model(k, Xs[s][~te], y[~te])
            oof[te] = score_model(M, Xs[s][te])
            sel.append(M["idx"])
        out[(s, k)] = (oof, sel)
    return out


def stats(oof, y, p):
    sig = p < 0.05
    rho = spearmanr(oof, y)[0]
    sa = float(np.mean((oof[sig] > 0) == (y[sig] > 0))) if sig.sum() else np.nan
    return rho, sa


def one_perm(i, Xs, y, p, grp, Xh):
    perm = np.random.RandomState(1000 + i).permutation(len(y))
    yp, pp = y[perm], p[perm]
    res = lodo(Xs, yp, grp)
    r = {c: stats(v[0], yp, pp) for c, v in res.items()}
    best = max(r, key=lambda c: np.nan_to_num(r[c][0], nan=-9))
    hb, hrho, hres = xd.best_of(Xh, yp, grp, xd.model_specs())
    return dict(rho=r[best][0], sa=r[best][1], cand_rho={c: r[c][0] for c in r}, h_rho=hrho[hb], h_sa=xd.sig_acc(hres[hb], yp, pp))


# ------------------------------------------------------------------ stage: fit
def fit_stage():
    t0 = time.time()
    KB, nb, cols = key_means("brats")
    KX, nx, cols2 = key_means("xds")
    assert cols == cols2
    KF = pd.concat([KB, KX])
    nkey = pd.concat([nb, nx])
    cells = xd.dev_cells()
    have = set(KF.index)
    miss = cells[~(cells.train.isin(have) & cells["eval"].isin(have))]
    cells = cells[cells.train.isin(have) & cells["eval"].isin(have)].reset_index(drop=True)
    Xh_all = xd.add_feats(cells)
    ok = np.isfinite(Xh_all).all(1) & np.isfinite(cells.delta.values)
    dropped = cells[~ok]
    cells, Xh = cells[ok].reset_index(drop=True), Xh_all[ok]
    y, p, grp = cells.delta.values, cells.p.fillna(1).values, cells.fam.values
    devkeys = sorted(set(cells.train) | set(cells["eval"]))
    Kd = KF.loc[devkeys]
    tex = [c for c in cols if group_of(c) == "texture"]
    sh = [c for c in cols if group_of(c) == "shape"]
    keep_tex = [tex[i] for i in prune_corr(Kd[tex].to_numpy(float), PRUNE)]
    keep_ts = keep_tex + [sh[i] for i in prune_corr(Kd[sh].to_numpy(float), PRUNE)]
    def cellX(c): return KF.loc[cells["eval"], c].to_numpy(float) - KF.loc[cells.train, c].to_numpy(float)
    Xs = {"tex": cellX(keep_tex), "ts": cellX(keep_ts)}
    names = {"tex": keep_tex, "ts": keep_ts}
    print(f"cells={len(cells)} (dropped {len(dropped)} NaN-hand, {len(miss)} missing rad keys) keys={len(devkeys)} "
          f"base={len(cols)} tex kept={len(keep_tex)} ts kept={len(keep_ts)}", flush=True)
    res = lodo(Xs, y, grp)
    st = {c: stats(v[0], y, p) for c, v in res.items()}
    best = max(st, key=lambda c: np.nan_to_num(st[c][0], nan=-9))
    hb, hrho, hres = xd.best_of(Xh, y, grp, xd.model_specs())
    h_sa = xd.sig_acc(hres[hb], y, p)
    # FREEZE: refit best on ALL dev cells, write before anything open-ms exists
    s, k = best
    M = fit_model(k, Xs[s], y)
    fz = dict(set=s, kind=k, features=[names[s][i] for i in M["idx"]], mu=[float(x) for x in M["mu"]], sd=[float(x) for x in M["sd"]],
              coef=[float(x) for x in M["coef"]], b=M["b"], scaleonly=M["scaleonly"], lodo_rho=float(st[best][0]),
              frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(fz, open(DATA / "radiomics_xds_frozen.json", "w"), indent=1)
    print("FROZEN", fz["set"], fz["kind"], fz["features"][:3], flush=True)
    nulls = Parallel(n_jobs=NJ)(delayed(one_perm)(i, Xs, y, p, grp, Xh) for i in range(NPERM))
    nrho = np.array([n["rho"] for n in nulls])
    nsa = np.array([n["sa"] for n in nulls], float)
    hrho_n = np.array([n["h_rho"] for n in nulls])
    hsa_n = np.array([n["h_sa"] for n in nulls], float)
    sig = p < 0.05
    pv = lambda nl, o: (1 + np.sum(nl[~np.isnan(nl)] >= o - 1e-12)) / (1 + np.sum(~np.isnan(nl)))  # noqa: E731
    sc = res[best][0]
    pergrp = {g: float(spearmanr(sc[grp == g], y[grp == g])[0]) for g in np.unique(grp) if (grp == g).sum() > 3}
    # selection frequency over LODO folds (selected idx per fold) for the frozen candidate, and per candidate best-feature tallies
    freq = pd.Series([names[s][i] for fold in res[best][1] for i in fold]).value_counts()
    out = dict(n_cells=len(cells), n_sig=int(sig.sum()), always_helps=float(np.mean(y[sig] > 0)), cand={c: dict(rho=float(v[0]), sa=v[1]) for c, v in st.items()},
               best=best, rho=float(st[best][0]), sa=st[best][1], p_rho=float(pv(nrho, st[best][0])), p_sa=float(pv(nsa, st[best][1])),
               null_rho=nrho, null95=float(np.percentile(nrho, 95)), pergrp=pergrp, freq=freq, frozen=fz,
               hand=dict(best=hb, rho=float(hrho[hb]), sa=h_sa, p_rho=float(pv(hrho_n, hrho[hb])), p_sa=float(pv(hsa_n, h_sa)), null_rho=hrho_n),
               oof=sc, y=y, p=p, grp=grp, cellnames=list(cells.name), N=NPERM, dropped=list(dropped.name),
               nkeys=nkey.loc[devkeys].to_dict(), sizes=dict(tex=len(keep_tex), ts=len(keep_ts), base=len(cols)))
    pickle.dump(out, open(DATA / "radiomics_B_fit.pkl", "wb"))
    print(f"fit done {time.time()-t0:.0f}s best={best} rho={out['rho']:+.3f} p={out['p_rho']:.3f}; hand {hb} rho={out['hand']['rho']:+.3f} p={out['hand']['p_rho']:.3f}", flush=True)


# ------------------------------------------------------------------ stage: openms (blind second look)
def openms_stage():
    fz = json.load(open(DATA / "radiomics_xds_frozen.json"))
    KO, _, cols = key_means("openms")
    oc = xd.openms_cells()
    f = fz["features"]
    X = KO.loc[oc["eval"], f].to_numpy(float) - KO.loc[oc.train, f].to_numpy(float)
    M = dict(idx=list(range(len(f))), mu=fz["mu"], sd=fz["sd"], coef=fz["coef"], b=fz["b"], scaleonly=fz["scaleonly"])
    so = score_model(M, X)
    oc = oc.assign(score=so, pred=np.where(so > 0, "helps", "hurts"), actual=np.where(oc.delta > 0, "helps", "hurts"))
    oc["hit"] = oc.pred == oc.actual
    rho = spearmanr(so, oc.delta)[0] if len(oc) > 2 else np.nan
    sg = oc[oc.p < 0.05]
    out = dict(cells=oc[["name", "score", "pred", "delta", "p", "actual", "hit"]], hits=int(oc.hit.sum()), n=len(oc), rho=float(rho),
               sig_hits=int(sg.hit.sum()), n_sig=len(sg), always=int((sg.delta > 0).sum()), frozen=fz)
    pickle.dump(out, open(DATA / "radiomics_B_openms.pkl", "wb"))
    print(oc.to_string(), f"\nhits {out['hits']}/{out['n']} sig {out['sig_hits']}/{out['n_sig']} always-helps {out['always']}/{out['n_sig']} rho {rho:+.2f}", flush=True)


if __name__ == "__main__":
    {"fit": fit_stage, "openms": openms_stage}[sys.argv[1]]()
