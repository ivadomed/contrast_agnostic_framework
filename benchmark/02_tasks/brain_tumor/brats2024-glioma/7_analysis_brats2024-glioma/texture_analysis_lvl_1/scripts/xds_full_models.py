"""Models / CV helpers for xds_full_* (importable so the frozen pickle can be loaded by the test script)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet

K_PREFILTER = 60

CFGS = {
    "hgb_d3": ("hgb", dict(max_depth=3, learning_rate=0.05, max_iter=100, min_samples_leaf=30, l2_regularization=1.0)),
    "hgb_d2": ("hgb", dict(max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=30, l2_regularization=1.0)),
    "rf_d6": ("rf", dict(n_estimators=150, max_depth=6, min_samples_leaf=15, max_features=0.3)),
    "rf_d3": ("rf", dict(n_estimators=150, max_depth=3, min_samples_leaf=30, max_features=0.3)),
    "enet_a0.05": ("enet", dict(alpha=0.05, l1_ratio=0.5)),
    "enet_a0.3": ("enet", dict(alpha=0.3, l1_ratio=0.5)),
}
SETS = ("hand", "rad", "both")
CANDS = [(s, c) for s in SETS for c in CFGS]  # listing order = tie-break order


def wnorm_ranks(X, w):
    """weighted-centered, weighted-norm-scaled column ranks (so corr_j = sum_i w_i Rn_ij ryn_i)."""
    R = rankdata(X, axis=0)
    m = (w[:, None] * R).sum(0) / w.sum()
    R = R - m
    n = np.sqrt((w[:, None] * R ** 2).sum(0)) + 1e-12
    return R / n


def prep_fold(Xtr, Xte, w):
    """median-impute (train medians), drop constant cols; returns dict used by every candidate on this fold."""
    med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    A = np.where(np.isfinite(Xtr), Xtr, med)
    B = np.where(np.isfinite(Xte), Xte, med) if Xte is not None else None
    keep = A.std(0) > 1e-9
    A = A[:, keep]
    out = dict(med=med, keep_idx=np.where(keep)[0], Xtr=A, Xte=None if B is None else B[:, keep], w=w)
    out["Rn"] = wnorm_ranks(A, w)
    return out


class Pipe:
    """impute (median) -> weighted-Spearman top-K prefilter -> {HGB | RF | ElasticNet}. All selection uses train rows only."""

    def __init__(self, cfg_name):
        self.cfg_name = cfg_name
        self.kind, self.kw = CFGS[cfg_name]

    def fit(self, cache, y, ycentered_rank=None):
        w = cache["w"]
        ry = rankdata(y)
        ry = ry - (w * ry).sum() / w.sum()
        ry = ry / (np.sqrt((w * ry ** 2).sum()) + 1e-12)
        r = cache["Rn"].T @ (w * ry)
        k = min(K_PREFILTER, len(r))
        self.sel = np.argsort(-np.abs(r))[:k]  # indices into cache's kept columns
        self.sel_global = cache["keep_idx"][self.sel]
        self.med = cache["med"]
        Xs = cache["Xtr"][:, self.sel]
        wn = w / w.mean()
        if self.kind == "enet":
            self.mu = (wn[:, None] * Xs).sum(0) / wn.sum()
            self.sd = Xs.std(0) + 1e-9
            self.ym = float((wn * y).sum() / wn.sum())
            self.ys = float(y.std() + 1e-9)
            self.m = ElasticNet(max_iter=5000, **self.kw).fit((Xs - self.mu) / self.sd, (y - self.ym) / self.ys, sample_weight=wn)
        elif self.kind == "hgb":
            self.m = HistGradientBoostingRegressor(random_state=0, **self.kw).fit(Xs, y, sample_weight=wn)
        else:
            self.m = RandomForestRegressor(random_state=0, n_jobs=1, **self.kw).fit(Xs, y, sample_weight=wn)
        return self

    def _pred(self, Xs):
        if self.kind == "enet":
            return self.m.predict((Xs - self.mu) / self.sd) * self.ys + self.ym
        return self.m.predict(Xs)

    def predict_cache(self, cache):
        return self._pred(cache["Xte"][:, self.sel])

    def predict_raw(self, X):
        Xs = X[:, self.sel_global]
        Xs = np.where(np.isfinite(Xs), Xs, self.med[self.sel_global])
        return self._pred(Xs)


def cell_agg(pred, cell_idx, ncell):
    s = np.bincount(cell_idx, weights=pred, minlength=ncell)
    n = np.bincount(cell_idx, minlength=ncell)
    out = np.full(ncell, np.nan)
    m = n > 0
    out[m] = s[m] / n[m]
    return out


def lodo_candidate(caches, grp, y, cell_idx, ncell, cfg_name):
    """caches: {group: fold cache}. returns oof row predictions."""
    oof = np.full(len(y), np.nan)
    for g, ca in caches.items():
        te = grp == g
        p = Pipe(cfg_name).fit(ca, y[~te])
        oof[te] = p.predict_cache(ca)
    return oof


def cell_rho(oof, cell_idx, ncell, cell_truth):
    pc = cell_agg(oof, cell_idx, ncell)
    m = np.isfinite(pc) & np.isfinite(cell_truth)
    return float(spearmanr(pc[m], cell_truth[m])[0]), pc
