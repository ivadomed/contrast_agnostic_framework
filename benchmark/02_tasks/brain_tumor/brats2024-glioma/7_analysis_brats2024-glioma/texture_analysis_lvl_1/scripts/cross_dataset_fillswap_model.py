#!/usr/bin/env python
"""
cross_dataset_fillswap_model.py -- predict sign/size of the ladder fill-swap effect
(rung 4 Voronoi noise-fill -> rung 5 real-fill, per-contrast delta Dice, in points) from
RELATIVE (eval minus train) contrast-pair features, Open-MS held out BLIND.

PRE-REGISTERED (written before any feature or outcome was computed):
  Cells (dev): every OOD (train,eval) pair in active ladder_series.json files, open-ms excluded.
    - contrasts iterated from the ladder's ood list (toothfairy2: fill_swap_significance keys), never
      the "(in-domain)" duplicates; dwi_ap/ablations_restricted skipped (duplicate of ablations).
    - breast: per-cohort ladders (duke/ispy1/acrin) + the ispy2 t1wce-trained -> t2w group (single-source).
      Same-contrast cross-dataset cells (t1wce-trained -> duke t1wce_uniap / ispy1 t1wce) are NOT
      OOD-contrast and are excluded. The ispy2-own t2w->t1wce cell exists only pooled with duke/ispy1,
      so it is dropped.
    - BraTS: per-region OOD cells (region_fill_swap_significance.csv, delta=mean_delta_pts, p=p_raw);
      pooled BraTS ladders not used. No t1c-trained region rows exist -> t1c-trained dropped.
    - delta = real-fill rung minus previous rung (Dice points); p = fill_swap_significance (raw).
  Features per (dataset,contrast,label), <=40 cases, labels pooled voxel-weighted: ring (1-5 vox) visibility
    AUC max(AUC,1-AUC); step_d; ramp R (r_adj_for_label); log high-pass(sigma=2) std ratio target/ring;
    in-plane high-pass autocorr lag1, lag2 of target; texture distance (L2 of acf1,acf2 target vs ring).
    Cell feature = eval - train. No identity/absolute values enter any model.
  Models: per feature a sign rule (score = s*x, s = train-fold Spearman sign) and OLS linear;
    plus ridge(alpha=1) and L2-logistic (C=1) on all 7 (standardised on training folds).
  CV: leave-one-dataset-out over {brats, chaos, onharmony, breast, toothfairy2}. Metrics: pooled
    out-of-fold Spearman(pred, delta); sign accuracy on p<0.05 cells vs "always helps".
    Permutation null: 200 shuffles of (delta,p) jointly across cells, WHOLE pipeline incl. selection.
  SELECTION RULE (frozen now): the model with the highest pooled leave-one-dataset-out Spearman
    (ties -> first listed) is refit on ALL dev cells and frozen; open-ms is touched only afterwards,
    scored once. Stale-fold3 mixing in some ladders is known and ignored.
Stages in one run: extract-dev -> fit(freeze) -> openms.  SMOKE=1 env: 2 cases/key, 5 perms, separate dirs.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings("ignore")
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
import nibabel as nib  # noqa: E402
import ramp_out_of_sample as ro  # noqa: E402

REPO = ro.REPO
SMOKE = os.environ.get("SMOKE") == "1"
OUT = THIS.parent / "outputs"
DATA = OUT / ("smoke_data" if SMOKE else "data")
TABLES = OUT / "tables"
PLOTS = OUT / "plots"
for d in (DATA, TABLES, PLOTS):
    d.mkdir(parents=True, exist_ok=True)
MAXC = 2 if SMOKE else 40
NPERM = 5 if SMOKE else 200
MINVOX = 100
FEATS = ["vis_auc", "step_d", "ramp_R", "log_hp_ratio", "acf1", "acf2", "tex_dist"]
RS = np.random.RandomState(0)

# ---------------- feature source keys ----------------
DS = REPO / "benchmark/02_tasks"
BR = DS / "breast_cancer"


def _nn(base, i, l):
    return ro._list_cases(Path(base) / i, Path(base) / l)


def extra_manifest():
    m = {}
    d = BR / "duke-breast-mri/2_nnUNet_duke-breast-mri/raw"
    m["duke:t1wce_uniap"] = (False, lambda d=d: _nn(d, "imagesTs_t1wce_uniap", "labelsTs_t1wce_uniap"))
    m["duke:precontrast_uniap"] = (False, lambda d=d: _nn(d, "imagesTs_precontrast_uniap", "labelsTs_precontrast_uniap"))
    d = BR / "ispy1/2_nnUNet_ispy1/raw"
    m["ispy1:t1wce"] = (False, lambda d=d: _nn(d, "imagesTs_t1wce", "labelsTs_t1wce"))
    m["ispy1:precontrast"] = (False, lambda d=d: _nn(d, "imagesTs_precontrast", "labelsTs_precontrast"))
    d = BR / "acrin6698/2_nnUNet_acrin6698/raw"
    m["acrin:dwi_uniap"] = (False, lambda d=d: _nn(d, "imagesTs_dwi_uniap", "labelsTs_dwi_uniap"))
    return m


MAN = {**ro.build_manifest(), **extra_manifest()}
BRATS_ROOT = DS / "brain_tumor/brats2024-glioma"
BIDS = BRATS_ROOT / "1_BIDS_brats2024-glioma/glioma-brain-brats2024"
LABDIR = BRATS_ROOT / "2_nnUNet_brats2024-glioma/raw/Dataset051_BraTS2024GliomaT1n/labelsTr"
BSUF = {"t1n": "T1w.nii", "t1c": "ce-gadolinium_T1w.nii", "t2w": "T2w.nii", "t2f": "FLAIR.nii"}
BREG = {"NCR": 1, "SNFH": 2, "ET": 3, "RC": 4}
W = BRATS_ROOT / "7_analysis_brats2024-glioma/texture_analysis_lvl_1"


def brats_patients():
    df = pd.read_csv(W / "outputs/data/internal_ramp_patient.csv")
    return sorted(df["patient"].unique())[:MAXC]


# ---------------- per-image features ----------------
def _acf(hp, mask, axes, lag):
    vals = []
    for ax in axes:
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax] = slice(0, -lag)
        b[ax] = slice(lag, None)
        a, b = tuple(a), tuple(b)
        ok = mask[a] & mask[b]
        if ok.sum() < 30:
            continue
        x, y = hp[a][ok].astype(np.float64), hp[b][ok].astype(np.float64)
        den = np.sqrt((x * x).sum() * (y * y).sum())
        if den > 0:
            vals.append((x * y).sum() / den)
    return float(np.mean(vals)) if vals else np.nan


def _sub(x, n=20000):
    return x if x.size <= n else x[RS.choice(x.size, n, replace=False)]


def label_feats(img, hp, fg, mask, axes, pad=12):
    if mask.sum() < MINVOX:
        return None
    idx = np.where(mask)
    sl = tuple(slice(max(0, i.min() - pad), i.max() + pad + 1) for i in idx)
    m = mask[sl]
    z, h, f = img[sl], hp[sl], fg[sl]
    ring = (distance_transform_edt(~m) <= 5) & ~m & f
    if ring.sum() < 50:
        return None
    t, r = z[m], z[ring]
    ts, rs_ = _sub(t), _sub(r)
    rk = rankdata(np.concatenate([ts, rs_]))
    n1, n2 = ts.size, rs_.size
    auc = (rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
    vis = max(auc, 1 - auc)
    sd = np.sqrt((t.var() + r.var()) / 2)
    step = abs(t.mean() - r.mean()) / sd if sd > 0 else np.nan
    R, _ = ro.r_adj_for_label(z, m)
    ht, hr = h[m].std(), h[ring].std()
    lhp = np.log(ht / hr) if ht > 0 and hr > 0 else np.nan
    a1t, a2t = _acf(h, m, axes, 1), _acf(h, m, axes, 2)
    a1r, a2r = _acf(h, ring, axes, 1), _acf(h, ring, axes, 2)
    td = np.sqrt((a1t - a1r) ** 2 + (a2t - a2r) ** 2)
    return [vis, step, R, lhp, a1t, a2t, td], int(m.sum())


def prep(img, zooms):
    img = np.squeeze(img).astype(np.float32)
    while img.ndim > 3:
        img = img[..., 0]
    lo, hi = float(img.min()), float(np.percentile(img, 99.5))
    fg = img > lo + 0.02 * (hi - lo)
    if fg.sum() < 1000:
        fg = np.ones_like(fg)
    mu, sd = img[fg].mean(), img[fg].std() + 1e-9
    z = (img - mu) / sd
    hp = z - gaussian_filter(z, 2.0)
    zz = np.asarray(zooms[:3])
    axes = tuple(int(a) for a in np.argsort(zz)[:2])
    return z, hp, fg, axes


def case_task(task):
    kind, key, cid, ip, lp, mand = task
    rows = []
    try:
        if kind == "brats":
            lbl = np.squeeze(np.asarray(nib.load(str(lp)).dataobj)).round().astype(int)
            for c, suf in BSUF.items():
                im = nib.load(str(Path(ip) / f"sub-{cid}_{suf}"))
                z, hp, fg, ax = prep(np.asarray(im.dataobj), im.header.get_zooms())
                for rn, rid in BREG.items():
                    o = label_feats(z, hp, fg, lbl == rid, ax)
                    if o:
                        rows.append((f"brats:{c}:{rn}", cid, rn, o[1], *o[0]))
        else:
            im = nib.load(str(ip))
            z, hp, fg, ax = prep(np.asarray(im.dataobj), im.header.get_zooms())
            lbl = np.squeeze(np.asarray(nib.load(str(lp)).dataobj)).round().astype(int)
            labs = [1] if mand else [int(v) for v in np.unique(lbl) if v > 0]
            for lab in labs:
                o = label_feats(z, hp, fg, lbl == lab, ax)
                if o:
                    rows.append((key, cid, "pooled", o[1], *o[0]))
    except Exception as e:  # noqa: BLE001
        print(f"SKIP {key} {cid}: {e}", flush=True)
    return rows


def extract(keys, brats=False):
    tasks = []
    todo = []
    for k in keys:
        if (DATA / f"xds_feat_{k.replace(':', '_')}.csv").exists():
            continue
        mand, fn = MAN[k]
        cs = fn()[:MAXC]
        print(f"{k}: {len(cs)} cases", flush=True)
        tasks += [("gen", k, c, i, l, mand) for c, i, l in cs]
        todo.append(k)
    if brats and not (DATA / "xds_feat_brats.csv").exists():
        tasks += [("brats", "brats", p, BIDS / f"sub-{p}" / "anat", LABDIR / f"{p}.nii.gz", False)
                  for p in brats_patients()]
        todo.append("brats")
    if not tasks:
        return
    with Pool(4) as pool:
        res = [r for rs in pool.imap_unordered(case_task, tasks, chunksize=1) for r in rs]
    cols = ["key", "case", "label", "n_vox"] + FEATS
    df = pd.DataFrame(res, columns=cols)
    for k in todo:
        if k == "brats":
            df[df.key.str.startswith("brats:")].to_csv(DATA / "xds_feat_brats.csv", index=False)
        else:
            df[df.key == k].to_csv(DATA / f"xds_feat_{k.replace(':', '_')}.csv", index=False)


def key_feat(key):
    """Mean over cases of the voxel-weighted label-pooled feature vector for a source key."""
    if key.startswith("brats:"):
        df = pd.read_csv(DATA / "xds_feat_brats.csv")
        df = df[df.key == key]
        pc = df.groupby("case")[FEATS].mean()  # single label per row-group
    else:
        df = pd.read_csv(DATA / f"xds_feat_{key.replace(':', '_')}.csv")
        pc = df.groupby("case").apply(
            lambda g: pd.Series({f: np.average(g[f].fillna(g[f].mean()), weights=g.n_vox) if g[f].notna().any() else np.nan
                                 for f in FEATS}))
    return pc.mean().values.astype(float), len(pc)


# ---------------- cells ----------------
def jload(rel):
    return json.loads((REPO / rel).read_text())


def ladder_cells(rel, train_key, fam, evmap, skip=()):
    d = jload(rel)
    ir = next(i for i, l in enumerate(d["labels"]) if "real fill" in l)
    out = []
    pcs = d.get("fill_swap_significance", {}).get("dice", {}).get("per_contrast", {})
    for ek, fk in evmap.items():
        if ek in skip or ek not in d["per_contrast"]["dice"]:
            continue
        a = d["per_contrast"]["dice"][ek]
        if np.isnan(a[ir]) or np.isnan(a[ir - 1]):
            continue
        out.append(dict(fam=fam, train=train_key, eval=fk, delta=a[ir] - a[ir - 1], p=pcs.get(ek, np.nan),
                        name=rel.split("8_results_")[1].split("/")[0] + ":" + train_key + "->" + fk))
    return out


def M(ds, tr):
    return f"benchmark/02_tasks/{ds}/8_results_{ds.split('/')[-1]}/02_metrics/{tr}/ablations/ladder_series.json"


def dev_cells():
    c = []
    ch = "abdomen_healthy/chaos/8_results_chaos/02_metrics/chaos_model"
    for t in ("t1in", "t2spir"):
        d = jload(f"benchmark/02_tasks/{ch}/{t}/ablations/ladder_series.json")
        c += ladder_cells(f"benchmark/02_tasks/{ch}/{t}/ablations/ladder_series.json", f"chaos:{t}", "chaos",
                          {e: f"chaos:{e}" for e in d["ood_contrasts"]})
    oh = "brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model"
    for t in ("T1w", "T2w", "dwi_ap"):
        rel = f"benchmark/02_tasks/{oh}/{t}/ablations/ladder_series.json"
        c += ladder_cells(rel, f"onharmony:{t}", "onharmony", {e: f"onharmony:{e}" for e in jload(rel)["ood_contrasts"]})
    rel = "benchmark/02_tasks/mandible_healthy/toothfairy2/8_results_toothfairy2/02_metrics/toothfairy2_model/cbct/ablations/ladder_series.json"
    c += ladder_cells(rel, "toothfairy2:cbct", "toothfairy2",
                      {k: k.replace("/", ":") for k in jload(rel)["fill_swap_significance"]["dice"]["per_contrast"]})
    # breast
    i2 = "benchmark/02_tasks/breast_cancer/ispy2/8_results_ispy2/02_metrics/ispy2_model"
    c += ladder_cells(f"{i2}/t1wce/ablations/ladder_series.json", "ispy2:t1wce", "breast", {"t2w": "ispy2:t2w"})
    sp = "benchmark/02_tasks/breast_cancer/{c}/8_results_{c}/02_metrics/ispy2_model/{t}/ablations/{s}/ladder_series.json"
    for t, tk in (("t2w", "ispy2:t2w"), ("t1wce", "ispy2:t1wce")):
        for cohort, sub, fk in (("ispy1", "precontrast", "ispy1:precontrast"), ("ispy1", "t1wce", "ispy1:t1wce"),
                                ("duke-breast-mri", "precontrast_uniap", "duke:precontrast_uniap"),
                                ("duke-breast-mri", "t1wce_uniap", "duke:t1wce_uniap"),
                                ("acrin6698", "dwi_uniap", "acrin:dwi_uniap")):
            if t == "t1wce" and sub in ("t1wce", "t1wce_uniap"):
                continue  # same-contrast cross-dataset only: not OOD-contrast
            c += ladder_cells(sp.format(c=cohort, t=t, s=sub), tk, "breast", {sub: fk})
    # brats region cells
    rg = pd.read_csv(W / "outputs/data/region_fill_swap_significance.csv")
    rg = rg[rg.family == "OOD"]
    for r in rg.itertuples():
        c.append(dict(fam="brats", train=f"brats:{r.train}:{r.region}", eval=f"brats:{r.eval}:{r.region}",
                      delta=r.mean_delta_pts, p=r.p_raw, name=f"brats:{r.train}->{r.eval}:{r.region}"))
    return pd.DataFrame(c)


def openms_cells():
    base = "benchmark/02_tasks/brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model"
    c = []
    for t, ev in (("flair", ("t1w", "t2w")), ("t1w", ("flair", "t2w"))):
        c += ladder_cells(f"{base}/{t}/ablations/ladder_series.json", f"open-ms:{t}", "openms",
                          {e: f"open-ms:{e}" for e in ev})
    return pd.DataFrame(c)


def add_feats(cells):
    cache = {}

    def g(k):
        if k not in cache:
            cache[k] = key_feat(k)[0]
        return cache[k]

    X = np.array([g(e) - g(t) for t, e in zip(cells.train, cells["eval"])])
    return X


# ---------------- models ----------------
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402


class Mdl:
    def __init__(self, kind, j=None):
        self.kind, self.j = kind, j
        self.name = f"{kind}:{FEATS[j]}" if j is not None else kind

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        if self.kind == "sign":
            self.s = np.sign(spearmanr(Z[:, self.j], y)[0]) or 1.0
        elif self.kind == "linear":
            b = np.polyfit(Z[:, self.j], y, 1)
            self.b = b
        elif self.kind == "ridge":
            self.m = Ridge(alpha=1.0).fit(Z, y)
        elif self.kind == "logit":
            self.m = LogisticRegression(C=1.0, max_iter=500).fit(Z, (y > 0).astype(int)) if len(set(y > 0)) > 1 else None
            self.const = float(np.mean(y > 0) * 2 - 1)
        return self

    def score(self, X):
        Z = (X - self.mu) / self.sd
        if self.kind == "sign":
            return self.s * X[:, self.j]  # zero-centred relative feature, no re-centring
        if self.kind == "linear":
            return np.polyval(self.b, Z[:, self.j])
        if self.kind == "ridge":
            return self.m.predict(Z)
        return self.m.decision_function(Z) if self.m is not None else np.full(len(Z), self.const)


def model_specs():
    s = []
    for j in range(len(FEATS)):
        s += [("sign", j), ("linear", j)]
    return s + [("ridge", None), ("logit", None)]


def lodo(X, y, grp, specs):
    """returns {name: oof score array}"""
    out = {}
    for kind, j in specs:
        oof = np.full(len(y), np.nan)
        for g in np.unique(grp):
            te = grp == g
            m = Mdl(kind, j).fit(X[~te], y[~te])
            oof[te] = m.score(X[te])
        out[Mdl(kind, j).name] = oof
    return out


def pred_sign(name, score):
    # sign rules: sign of score (zero-centred relative feature); regressions: sign of predicted delta/logit
    return np.where(score > 0, 1, -1)


def sig_acc(score, y, p):
    s = p < 0.05
    return float(np.mean(pred_sign("", score[s]) == np.sign(y[s]))) if s.sum() else np.nan


def best_of(X, y, grp, specs):
    res = lodo(X, y, grp, specs)
    rho = {k: spearmanr(v, y)[0] for k, v in res.items()}
    best = max(rho, key=lambda k: (np.nan_to_num(rho[k], nan=-9)))
    return best, rho, res


# ---------------- main ----------------
def main():
    dev_keys = sorted({k for k in MAN if not k.startswith("open-ms") and not k.startswith("brats")})
    cells = dev_cells()
    need = sorted(set(cells.train) | set(cells["eval"]))
    gen_keys = [k for k in need if not k.startswith("brats:")]
    print("dev cells", len(cells), "source keys", len(gen_keys), flush=True)
    extract(gen_keys, brats=True)
    X = add_feats(cells)
    ok = np.isfinite(X).all(1) & np.isfinite(cells.delta.values)
    dropped = cells[~ok]
    cells, X = cells[ok].reset_index(drop=True), X[ok]
    y, p, grp = cells.delta.values, cells.p.fillna(1).values, cells.fam.values
    specs = model_specs()
    best, rho, oof = best_of(X, y, grp, specs)
    # freeze: refit on all dev cells, then stop touching dev
    frozen = Mdl(*next(s for s in specs if Mdl(*s).name == best)).fit(X, y)
    json.dump(dict(model=best, kind=frozen.kind, j=frozen.j, mu=frozen.mu.tolist(), sd=frozen.sd.tolist(),
                   rho_lodo=rho[best]), open(DATA / "xds_frozen_model.json", "w"), indent=1)
    sc = oof[best]
    sig = p < 0.05
    # permutation null (whole pipeline incl. selection)
    nb, ns = [], []
    for _ in range(NPERM):
        perm = RS.permutation(len(y))
        yp, pp = y[perm], p[perm]
        b, r, o = best_of(X, yp, grp, specs)
        nb.append(r[b])
        ns.append(sig_acc(o[b], yp, pp))
    nb, ns = np.array(nb), np.array(ns, dtype=float)
    real_sa = sig_acc(sc, y, p)
    base_sa = float(np.mean(y[sig] > 0))
    pr = (1 + np.sum(nb >= rho[best])) / (1 + NPERM)
    ps = (1 + np.sum(ns[~np.isnan(ns)] >= real_sa)) / (1 + np.sum(~np.isnan(ns)))
    pergrp = {g: spearmanr(sc[grp == g], y[grp == g])[0] if (grp == g).sum() > 3 else np.nan for g in np.unique(grp)}
    print("best", best, rho[best], flush=True)

    # ---- stage 3: open-ms, blind, scored once ----
    oc = openms_cells()
    extract(sorted(set(oc.train) | set(oc["eval"])))
    Xo = add_feats(oc)
    fz = Mdl(frozen.kind, frozen.j)
    fz.__dict__.update(frozen.__dict__)
    so = fz.score(Xo)
    oc["score"] = so
    oc["pred"] = np.where(so > 0, "helps", "hurts")
    oc["actual"] = np.where(oc.delta > 0, "helps", "hurts")
    oc["hit"] = oc.pred == oc.actual
    rho_o = spearmanr(so, oc.delta)[0] if len(oc) > 2 else np.nan

    # ---- report ----
    L = ["# Cross-dataset fill-swap model (relative features, Open-MS blind)", "",
         f"Dev cells: {len(cells)} (dropped for NaN features: {len(dropped)}); per group: "
         + ", ".join(f"{g} {int((grp == g).sum())}" for g in np.unique(grp))
         + f"; significant (p<0.05): {int(sig.sum())}. BraTS dominates the pooled Spearman.", "",
         "Excluded by design: t1wce-trained -> duke t1wce_uniap / ispy1 t1wce (same-contrast, not OOD); ispy2 "
         "t2w->t1wce (only available pooled); brats t1c-trained (no region rows); open-ms (blind).", "",
         "## Leave-one-dataset-out (pooled out-of-fold Spearman, sorted)", "", "| model | rho |", "|---|--:|"]
    for k, v in sorted(rho.items(), key=lambda kv: -np.nan_to_num(kv[1], nan=-9)):
        L.append(f"| {k} | {v:+.3f} |")
    L += ["", f"**Selected/frozen: {best}**, LODO rho = {rho[best]:+.3f}; permutation null (selection inside, "
          f"{NPERM} shuffles) mean {nb.mean():+.3f}, 95th pct {np.percentile(nb, 95):+.3f}, p = {pr:.3f}.",
          f"Sign accuracy on significant cells: model {real_sa:.2f} vs always-helps {base_sa:.2f} "
          f"(n={int(sig.sum())}); perm p(sign-acc >= model) = {ps:.3f}.", "",
          "Per held-out group Spearman: " + ", ".join(f"{g} {v:+.2f}" for g, v in pergrp.items()), "",
          "## Open-ms (frozen model scored once)", "",
          "| cell | score | pred | delta | p | actual | hit |", "|---|--:|:-:|--:|--:|:-:|:-:|"]
    for r in oc.itertuples():
        L.append(f"| {r.name} | {r.score:+.2f} | {r.pred} | {r.delta:+.2f} | {r.p:.3g} | {r.actual} | "
                 f"{'HIT' if r.hit else 'miss'} |")
    sg = oc[oc.p < 0.05]
    L += ["", f"Hits: {int(oc.hit.sum())}/{len(oc)} (significant cells {int(sg.hit.sum())}/{len(sg)}; "
          f"always-helps {int((sg.delta > 0).sum())}/{len(sg)}); Spearman(score, delta) = {rho_o:+.2f} (n={len(oc)})."]
    if len(dropped):
        L += ["", "Dropped cells: " + ", ".join(dropped.name)]
    (TABLES / ("cross_dataset_fillswap_model_SMOKE.md" if SMOKE else "cross_dataset_fillswap_model.md")).write_text("\n".join(L))
    print("\n".join(L), flush=True)

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    cols = dict(zip(np.unique(grp), plt.cm.tab10.colors))
    for g in np.unique(grp):
        s = grp == g
        ax[0].scatter(sc[s], y[s], c=[cols[g]], label=g, s=np.where(sig[s], 45, 15), alpha=.8)
    ax[0].scatter(so, oc.delta, marker="*", s=220, c="k", label="open-ms (blind)")
    ax[0].axhline(0, c="grey", lw=.5)
    ax[0].axvline(0, c="grey", lw=.5)
    ax[0].set_xlabel(f"out-of-fold score ({best}); open-ms = frozen")
    ax[0].set_ylabel("real-fill minus noise-fill, Dice pts")
    ax[0].legend(fontsize=7)
    ax[0].set_title(f"LODO rho={rho[best]:+.2f} (perm p={pr:.3f}); big markers p<0.05")
    ax[1].hist(nb, bins=20, color="lightgrey")
    ax[1].axvline(rho[best], c="r")
    ax[1].set_xlabel("selected-model LODO Spearman under null")
    ax[1].set_title("permutation null (red = observed)")
    fig.tight_layout()
    fig.savefig(PLOTS / ("cross_dataset_fillswap_model_SMOKE.png" if SMOKE else "cross_dataset_fillswap_model.png"), dpi=130)


if __name__ == "__main__":
    main()
