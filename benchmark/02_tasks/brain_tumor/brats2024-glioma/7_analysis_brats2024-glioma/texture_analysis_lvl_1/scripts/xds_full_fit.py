#!/usr/bin/env python
"""
xds_full_fit.py -- STAGE 1 (dev only) of the full-feature cross-dataset fill-swap model.

Question: with ALL image-extractable features (contrast-identifying ones allowed), can a model trained on dev datasets
(BraTS per-region OOD cells, breast family, on-harmony, toothfairy2[+hanseg/pddca]) predict the per-patient fill-swap
effect (rung 4 Voronoi noise-fill -> rung 5 real-fill, delta Dice, points) well enough to score two held-out datasets
(Open-MS, CHAOS) that never enter any fit, selection or null here?  This script never imports or opens chaos / open-ms
ladder JSON or eval_all.csv (xds_full_common builds DEV specs only).  Stage 2 (xds_full_test.py) is a separate job run
afterok and asserts the frozen pickle written here.

PRE-REGISTERED (written before any model was run)
  Rows: patient(case) x OOD cell, target = per-case delta Dice points (ladder's own run_keys, folds 0-2, labels pooled per
    case as in the ladder; BraTS per patient x region, GT-present).  <=20 cases per non-BraTS key (what the feature
    extraction covers); all BraTS patients.  The cell's truth for modelling = mean of its rows' deltas.
  Features per row (target region; hand-made = the 105 bigfeat features, radiomics = PyRadiomics R/ring/R-minus-ring/shape):
    E = eval-contrast features, T = train-contrast features (same patient if co-registered: same patient key, equal shape,
    allclose affine; else the train key's dataset mean), D = E - T.  Label-free correlation pruning (|Spearman|>0.95) of the
    base features is done once on the dev EVAL rows (no outcome used; transductive over dev groups, stated).
  Feature sets: hand | rad | both.  Models (6 configs, fixed grid): HistGradientBoosting x2, RandomForest x2,
    ElasticNet x2.  Inside every training fold: median impute -> weighted-Spearman top-60 prefilter -> model; standardisation
    (elastic-net) inside the fold.  Row weights = 1/n_cell(normalised) so each cell counts equally (BraTS ~2000 rows
    would otherwise dominate).  18 candidates = 3 sets x 6 configs.
  SELECTION RULE (frozen now): leave-one-DATASET-out over the 4 dev groups {brats, breast, onharmony, toothfairy2};
    for each candidate the held-out row predictions are averaged per cell, pooled over all dev cells, and scored by
    Spearman against the cell truth.  The candidate with the highest pooled cell-level Spearman (ties: listed order) is
    refit on ALL dev rows and frozen to outputs/data/xds_full_frozen.pkl BEFORE any test outcome is read.
  NULL: selection-aware permutation (NPERM>=100): the cell means are permuted across all dev cells and every row is
    shifted by (new cell mean - old cell mean) (within-cell residuals kept); the WHOLE pipeline (all 18 candidates x LODO,
    best chosen) is rerun; p = (1+#{null best >= observed best})/(1+NPERM).
  Importance: permutation importance of the selected candidate on its LODO held-out folds (each selected feature permuted
    over held-out rows, 5 repeats; metric = increase of cell-level MSE of aggregated predictions), averaged over folds
    (0 where not selected), summed by source (E/T/D) and family.
SMOKE=1: few perms, separate output names; never writes the real frozen pickle.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
import xds_full_common as C  # noqa: E402
import xds_full_models as M  # noqa: E402
from radiomics_common import prune_corr  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
NPERM = int(os.environ.get("XF_NPERM", 3 if SMOKE else 100))
NJ = int(os.environ.get("XF_NJOBS", 8))
PRUNE = 0.95
SUF = "_SMOKE" if SMOKE else ""
FROZEN = C.DATA / f"xds_full_frozen{SUF}.pkl"
FITRES = C.DATA / f"xds_full_fit_results{SUF}.pkl"


def prune_base(E, names, thr=PRUNE, max_nan=0.4):
    ok = [j for j in range(E.shape[1]) if np.isfinite(E[:, j]).mean() > 1 - max_nan and np.nanstd(E[:, j]) > 1e-9]
    keep = prune_corr(E[:, ok], thr)
    return [names[ok[i]] for i in keep]


def one_perm(i, cache_sets, grp, y, cell_idx, ncell, cell_mean):
    rs = np.random.RandomState(1000 + i)
    perm = rs.permutation(ncell)
    newm = cell_mean[perm]
    yp = y - cell_mean[cell_idx] + newm[cell_idx]
    best, bc = -9, None
    for s, cfg in M.CANDS:
        oof = M.lodo_candidate(cache_sets[s], grp, yp, cell_idx, ncell, cfg)
        r, _ = M.cell_rho(oof, cell_idx, ncell, newm)
        if np.nan_to_num(r, nan=-9) > best:
            best, bc = r, (s, cfg)
    return best, bc


def main():
    t0 = time.time()
    MAN = C.manifest()
    specs = C.dev_specs()
    H, R = C.load_hand(), C.load_rad()
    # chaos / open-ms feature units may exist in H,R; they are excluded here by construction (dev keys only)
    keep_keys = {k for s in specs for k in (s["train_key"], s["eval_key"])}
    H = H[H.index.get_level_values(0).isin(keep_keys)]
    R = R[R.index.get_level_values(0).isin(keep_keys)]
    meta, cells, Eh, Th, Er, Tr, hcols, rcols = C.build_rows(specs, C.targets, H, R, MAN)
    print(f"rows {len(meta)} cells {len(cells)} (with rows {int((cells.n_rows > 0).sum())}) hand {len(hcols)} rad {len(rcols)}  {time.time()-t0:.0f}s", flush=True)
    cells = cells.copy()
    ci = meta["cell"].to_numpy()
    uc = sorted(set(ci))
    remap = {c: i for i, c in enumerate(uc)}
    cell_idx = np.array([remap[c] for c in ci])
    ncell = len(uc)
    cells_used = cells[cells.cell.isin(uc)].set_index("cell").loc[uc].reset_index()
    y = meta["y"].to_numpy()
    grp = meta["fam"].to_numpy()
    n_cell = np.bincount(cell_idx, minlength=ncell)
    cell_mean = np.bincount(cell_idx, weights=y, minlength=ncell) / n_cell
    w = 1.0 / n_cell[cell_idx]
    w = w / w.mean()
    cells_used["sub_mean"] = cell_mean
    cells_used["group"] = cells_used["fam"]

    # label-free pruning on dev eval rows
    kh = prune_base(Eh, hcols)
    kr = prune_base(Er, rcols)
    ih, ir = [hcols.index(c) for c in kh], [rcols.index(c) for c in kr]
    print(f"pruned base feats: hand {len(hcols)}->{len(kh)}  rad {len(rcols)}->{len(kr)}", flush=True)

    def blocks(E, T, idx, kind, names):
        e, t = E[:, idx], T[:, idx]
        nm = [f"E|{kind}|{names[j]}" for j in idx] + [f"T|{kind}|{names[j]}" for j in idx] + [f"D|{kind}|{names[j]}" for j in idx]
        return np.hstack([e, t, e - t]), nm

    Xh, nh = blocks(Eh, Th, ih, "hand", hcols)
    Xr, nr = blocks(Er, Tr, ir, "rad", rcols)
    sets = {"hand": (Xh, nh), "rad": (Xr, nr), "both": (np.hstack([Xh, Xr]), nh + nr)}
    groups = sorted(set(grp))
    print("groups", {g: int((grp == g).sum()) for g in groups}, {g: int(len(set(cell_idx[grp == g]))) for g in groups}, flush=True)

    # fold caches per set
    cache_sets = {}
    for s, (X, nm) in sets.items():
        cache_sets[s] = {g: M.prep_fold(X[grp != g], X[grp == g], w[grp != g]) for g in groups}
    print(f"caches built {time.time()-t0:.0f}s", flush=True)

    # ---- LODO for all candidates
    res = {}
    for s, cfg in M.CANDS:
        ta = time.time()
        oof = M.lodo_candidate(cache_sets[s], grp, y, cell_idx, ncell, cfg)
        r, pc = M.cell_rho(oof, cell_idx, ncell, cell_mean)
        res[(s, cfg)] = dict(oof=oof, rho=r, pc=pc)
        print(f"  LODO {s}/{cfg}: cell rho {r:+.3f}  ({time.time()-ta:.1f}s)", flush=True)
    best = max(M.CANDS, key=lambda c: np.nan_to_num(res[c]["rho"], nan=-9))
    print("SELECTED", best, res[best]["rho"], flush=True)

    # ---- FREEZE: refit on ALL dev rows, write before any test code can run
    X, nm = sets[best[0]]
    full_cache = M.prep_fold(X, None, w)
    pipe = M.Pipe(best[1]).fit(full_cache, y)
    frozen = dict(best=best, pipe=pipe, names=nm, set=best[0], hand_kept=kh, rad_kept=kr, hand_cols=hcols, rad_cols=rcols,
                  lodo_rho=res[best]["rho"], frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), n_rows=len(y), n_cells=ncell,
                  selected_features=[nm[j] for j in pipe.sel_global])
    with open(FROZEN, "wb") as f:
        pickle.dump(frozen, f)
    print("FROZEN written", FROZEN, flush=True)

    # ---- observed stats
    oof = res[best]["oof"]
    pc = res[best]["pc"]
    pcell = cells_used["p"].fillna(1).to_numpy()
    sig = pcell < 0.05
    sa = float(np.mean((pc[sig] > 0) == (cell_mean[sig] > 0))) if sig.sum() else np.nan
    always = float(np.mean(cell_mean[sig] > 0)) if sig.sum() else np.nan
    pergrp = {g: float(spearmanr(pc[(cells_used.fam == g).to_numpy()], cell_mean[(cells_used.fam == g).to_numpy()])[0])
              if (cells_used.fam == g).sum() > 3 else np.nan for g in groups}
    rowrho = float(spearmanr(oof, y)[0])
    table = pd.DataFrame([dict(set=s, cfg=c, rho=res[(s, c)]["rho"]) for s, c in M.CANDS]).sort_values("rho", ascending=False)

    # ---- permutation importance on LODO held-out folds (selected candidate)
    X, nm = sets[best[0]]
    imp = np.zeros(X.shape[1])
    rs = np.random.RandomState(0)
    for g in groups:
        te = grp == g
        ca = cache_sets[best[0]][g]
        p = M.Pipe(best[1]).fit(ca, y[~te])
        cidx = cell_idx[te]
        ucg = sorted(set(cidx))
        rm = {c: i for i, c in enumerate(ucg)}
        ci2 = np.array([rm[c] for c in cidx])
        truth = cell_mean[ucg]
        base = np.mean((M.cell_agg(p.predict_cache(ca), ci2, len(ucg)) - truth) ** 2)
        Xte = ca["Xte"].copy()
        for jj, j in enumerate(p.sel):
            col = Xte[:, j].copy()
            d = []
            for _ in range(5):
                Xte[:, j] = rs.permutation(col)
                pr = p._pred(Xte[:, p.sel])
                d.append(np.mean((M.cell_agg(pr, ci2, len(ucg)) - truth) ** 2) - base)
            Xte[:, j] = col
            imp[p.sel_global[jj]] += np.mean(d) / len(groups)
    impdf = pd.DataFrame(dict(feature=nm, importance=imp))
    parts = impdf.feature.str.split("|", n=2, expand=True)
    impdf["src"], impdf["kind"], impdf["base"] = parts[0], parts[1], parts[2]
    fam = [C.family_of(b, k) for b, k in zip(impdf.base, impdf.kind)]
    impdf["family"], impdf["subfamily"] = [f[0] for f in fam], [f[1] for f in fam]

    # ---- selection-aware permutation null (parallel over perms)
    print(f"null: {NPERM} perms x {len(M.CANDS)} candidates, {NJ} jobs  ({time.time()-t0:.0f}s so far)", flush=True)
    tn = time.time()
    out = Parallel(n_jobs=NJ, verbose=0)(delayed(one_perm)(i, cache_sets, grp, y, cell_idx, ncell, cell_mean) for i in range(NPERM))
    nb = np.array([o[0] for o in out])
    pval = float((1 + np.sum(nb >= res[best]["rho"] - 1e-12)) / (1 + NPERM))
    print(f"null done {time.time()-tn:.0f}s ({(time.time()-tn)/max(NPERM,1):.1f}s/perm wall): mean {nb.mean():+.3f} p95 {np.percentile(nb,95):+.3f} p={pval:.3f}", flush=True)
    outd = dict(best=best, table=table, rho=res[best]["rho"], null=nb, pval=pval, pergrp=pergrp, sign_acc=sa, always=always,
                n_sig=int(sig.sum()), cells=cells_used, cell_pred=pc, cell_truth=cell_mean, row_rho=rowrho, importance=impdf,
                meta_summary=dict(n_rows=len(y), n_cells=ncell, groups={g: int((grp == g).sum()) for g in groups},
                                  n_hand_base=(len(hcols), len(kh)), n_rad_base=(len(rcols), len(kr)), NPERM=NPERM),
                cells_all=cells, matched_frac=float(meta.matched.mean()), frozen_at=frozen["frozen_at"])
    with open(FITRES, "wb") as f:
        pickle.dump(outd, f)
    print(table.to_string(), flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
