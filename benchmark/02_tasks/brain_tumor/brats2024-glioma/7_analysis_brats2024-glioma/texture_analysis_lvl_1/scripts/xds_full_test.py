#!/usr/bin/env python
"""
xds_full_test.py -- STAGE 2: score the FROZEN full-feature model on the held-out Open-MS (4 OOD cells) and CHAOS (6 OOD
cells) once, and write the report + plot.  Run only after xds_full_fit.py (afterok); asserts the frozen pickle exists and
records its mtime.  This is the first code that opens chaos / open-ms ladder JSON or eval_all.csv.

Scoring rule (fixed before looking): row score = frozen model prediction of the per-case delta Dice (points); cell score =
mean of its rows; predicted sign = sign of the cell score (> 0 "helps").  Observed cell delta = the ladder's own value
(ladder_series.json per_contrast, all cases) and p = its fill_swap_significance (raw Wilcoxon).  Reported: sign hits
overall and on p<0.05 cells vs the "always helps" baseline (= predict helps everywhere), Spearman(cell score, observed)
over the 10 cells (and per dataset), row-level Spearman, per-cell n.
Honesty note (also in the report): both test sets' ladder outcomes were seen before.  Open-MS was scored twice by earlier
models (FINDINGS 15, 16).  CHAOS was a DEV group in FINDINGS 15 and 16 (6 cells, selection and null) so its outcomes shaped
those earlier model choices; only THIS pipeline (selection rule, grid, features) never touched chaos/open-ms rows.
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
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
import xds_full_common as C  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
SUF = "_SMOKE" if SMOKE else ""
FROZEN = C.DATA / f"xds_full_frozen{SUF}.pkl"
FITRES = C.DATA / f"xds_full_fit_results{SUF}.pkl"


def test_specs():
    sp = []
    if os.environ.get("XF_DRYRUN") == "1":  # debug the report code on DEV non-brats cells only (never opens chaos/open-ms)
        return [x for x in C.dev_specs() if x["fam"] in ("breast", "onharmony")][::3]
    for t in ("t1in", "t2spir"):
        rel = f"abdomen_healthy/chaos/8_results_chaos/02_metrics/chaos_model/{t}/ablations/ladder_series.json"
        sp += C._ladder_specs(rel, f"chaos:{t}", "chaos", {e: f"chaos:{e}" for e in C.jload(C.TS / rel)["ood_contrasts"]})
    for t, evs in (("flair", ("t1w", "t2w")), ("t1w", ("flair", "t2w"))):
        rel = f"brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model/{t}/ablations/ladder_series.json"
        sp += C._ladder_specs(rel, f"open-ms:{t}", "openms", {e: f"open-ms:{e}" for e in evs})
    return sp


def main():
    assert FROZEN.exists(), "frozen model missing: run xds_full_fit.py first"
    fmt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(FROZEN.stat().st_mtime))
    fz = pickle.load(open(FROZEN, "rb"))
    fit = pickle.load(open(FITRES, "rb"))
    pipe, names = fz["pipe"], fz["names"]
    MAN = C.manifest()
    specs = test_specs()
    print("test cells", len(specs), [s["name"] for s in specs], flush=True)
    H, R = C.load_hand(), C.load_rad()
    keys = {k for s in specs for k in (s["train_key"], s["eval_key"])}
    H = H[H.index.get_level_values(0).isin(keys)]
    R = R[R.index.get_level_values(0).isin(keys)]
    meta, cells, Eh, Th, Er, Tr, hcols, rcols = C.build_rows(specs, C.cell_targets, H, R, MAN)
    assert hcols == fz["hand_cols"] and rcols == fz["rad_cols"], "feature column order differs from the frozen model"

    def blocks(E, T, kept, cols):
        idx = [cols.index(c) for c in kept]
        e, t = E[:, idx], T[:, idx]
        return np.hstack([e, t, e - t])

    parts = []
    if fz["set"] in ("hand", "both"):
        parts.append(blocks(Eh, Th, fz["hand_kept"], hcols))
    if fz["set"] in ("rad", "both"):
        parts.append(blocks(Er, Tr, fz["rad_kept"], rcols))
    X = np.hstack(parts)
    assert X.shape[1] == len(names)
    meta["pred"] = pipe.predict_raw(X)
    cells = cells.copy()
    g = meta.groupby("cell")
    cells["score"] = cells.cell.map(g["pred"].mean())
    cells["sub_mean"] = cells.cell.map(g["y"].mean())
    cells["observed"] = cells["json_delta"]
    cells["pred_sign"] = np.where(cells.score > 0, "helps", "hurts")
    cells["actual"] = np.where(cells.observed > 0, "helps", "hurts")
    cells["hit"] = cells.pred_sign == cells.actual
    cells["always_hit"] = cells.observed > 0
    cells["sig"] = cells.p < 0.05
    ok = cells.score.notna()
    c = cells[ok]
    rho_all = spearmanr(c.score, c.observed)[0]
    per = {d: (spearmanr(x.score, x.observed)[0] if len(x) > 2 else np.nan) for d, x in c.groupby("fam")}
    rowrho = spearmanr(meta.pred, meta.y)[0]
    sg = c[c.sig]
    res = dict(hits=int(c.hit.sum()), n=len(c), sig_hits=int(sg.hit.sum()), n_sig=len(sg), always=int(c.always_hit.sum()),
               always_sig=int(sg.always_hit.sum()), rho=float(rho_all), per=per, rowrho=float(rowrho),
               hits_by={d: (int(x.hit.sum()), len(x)) for d, x in c.groupby("fam")}, cells=cells, meta=meta)
    pickle.dump(res, open(C.DATA / f"xds_full_test_results{SUF}.pkl", "wb"))

    # ------------------------------------------------------------------ report
    cu = fit["cells"]
    imp = fit["importance"]
    ms = fit["meta_summary"]
    L = [f"# Full-feature cross-dataset fill-swap model: frozen, then scored on Open-MS and CHAOS{' (SMOKE)' if SMOKE else ''}", "",
         "## Setup (pre-registered in `scripts/xds_full_fit.py` docstring)", "",
         f"- Dev rows {ms['n_rows']} (patient/case x OOD cell), {ms['n_cells']} dev cells; rows per group {ms['groups']}. "
         "Dev groups: BraTS (per-region OOD cells, 3 training contrasts), breast family (ispy2 + duke/ispy1/acrin items), on-harmony, "
         "toothfairy2 (+hanseg/pddca). **No Open-MS or CHAOS row in any fit, selection, pruning or null.**",
         "- Target: per-case rung-5 minus rung-4 delta Dice (points), ladder run_keys, folds 0-2, labels pooled per case; "
         "<=20 cases per non-BraTS key (feature coverage), all GT-present BraTS patients. Cell weights 1/n_cell.",
         f"- Features per row: eval-contrast (E), train-contrast (T; same patient if co-registered, else train-key dataset mean; "
         f"{100*fit['matched_frac']:.0f}% of dev rows matched), difference (D); hand-made (105 base -> {ms['n_hand_base'][1]} after label-free "
         f"pruning) and PyRadiomics (R/ring/R-ring/shape, {ms['n_rad_base'][0]} base -> {ms['n_rad_base'][1]}); incl. contrast-identifying whole-brain features.",
         "- Models: HGB x2, RF x2, elastic-net x2 configs x feature sets {hand, rad, both} = 18 candidates; weighted-Spearman top-60 prefilter, "
         "imputation and scaling inside folds. **Rule:** highest pooled leave-one-dataset-out cell-level Spearman (4 groups) -> refit on all dev -> frozen.",
         f"- Frozen at {fz['frozen_at']} (`outputs/data/xds_full_frozen.pkl` mtime {fmt}); test stage opened chaos/open-ms outcomes only after that.", "",
         "## Dev: leave-one-dataset-out selection", "", "| feature set | config | LODO cell Spearman |", "|---|---|--:|"]
    for r in fit["table"].itertuples():
        L.append(f"| {r.set} | {r.cfg} | {r.rho:+.3f} |")
    nb = fit["null"]
    L += ["", f"**Selected: {fit['best'][0]} / {fit['best'][1]}**, pooled LODO cell Spearman {fit['rho']:+.3f} "
          f"(row-level {fit['row_rho']:+.3f}). Selection-aware permutation null ({len(nb)} perms, whole 18-candidate search, cell means permuted "
          f"across cells, within-cell residuals kept): null mean {nb.mean():+.3f}, 95th pct {np.percentile(nb, 95):+.3f}, **p = {fit['pval']:.3f}**.",
          "Per held-out dataset (cell-level Spearman): " + ", ".join(f"{k} {v:+.2f}" for k, v in fit["pergrp"].items()) + ".",
          f"Sign accuracy of cell scores on the {fit['n_sig']} dev cells with fill-swap p<0.05: {fit['sign_acc']:.2f} vs always-helps {fit['always']:.2f}.", "",
          "## Held-out test (frozen model, scored once)", "",
          "| cell | rows used / cases | pred (pts) | observed (pts) | ladder p | pred | actual | hit |", "|---|--:|--:|--:|--:|:-:|:-:|:-:|"]
    for r in cells.itertuples():
        L.append(f"| {r.name} | {r.n_rows}/{r.n_full} ({r.n_matched} same-patient train feats) | {r.score:+.2f} | {r.observed:+.2f} | {r.p:.3g} | "
                 f"{r.pred_sign} | {r.actual} | {'HIT' if r.hit else 'miss'} |")
    hb = ", ".join(f"{d} {a}/{b}" for d, (a, b) in res["hits_by"].items())
    L += ["", f"- Sign hits: **{res['hits']}/{res['n']}** ({hb}); on p<0.05 cells **{res['sig_hits']}/{res['n_sig']}**.",
          f"- 'Always helps' baseline: {res['always']}/{res['n']} overall, {res['always_sig']}/{res['n_sig']} on p<0.05 cells.",
          f"- Spearman(cell score, observed) over {res['n']} cells: **{res['rho']:+.2f}**; per dataset: " + ", ".join(f"{k} {v:+.2f}" for k, v in per.items()) +
          f". Row-level Spearman (all test rows, n={len(meta)}): {res['rowrho']:+.2f}.",
          "- CHAOS MR cells have only 4 patients (MR05, MR19, ...); the CT cells 20; Open-MS 8 per cell.",
          "- **Not blind in the strict sense:** both test sets' ladder outcomes were seen before. Open-MS was scored twice by earlier models "
          "(FINDINGS 15, 16: 2/4 hits both times). CHAOS was a DEV group in FINDINGS 15 and 16 (its 6 cells informed those models' selection), "
          "so earlier chaos conclusions are in-sample; this pipeline's grid, rule and features were fixed without reading either set's outcomes, "
          "but the researcher knew them.", "", "## Top features (LODO permutation importance, selected model)", ""]
    top = imp.sort_values("importance", ascending=False).head(15)
    L += ["| feature | source | importance (cell-MSE increase) |", "|---|---|--:|"]
    for r in top.itertuples():
        L.append(f"| {r.kind}: {r.base} | {r.src} | {r.importance:.4f} |")
    gsrc = imp.groupby("src")["importance"].sum().sort_values(ascending=False)
    gfam = imp.groupby(["kind", "family"])["importance"].sum().sort_values(ascending=False)
    gsub = imp.groupby(["kind", "family", "subfamily"])["importance"].sum().sort_values(ascending=False).head(10)
    tot = max(imp.importance.clip(lower=0).sum(), 1e-12)
    L += ["", "Importance summed by source: " + ", ".join(f"{k} {v:.4f}" for k, v in gsrc.items()) +
          " (E = eval contrast, T = train contrast, D = difference).", "",
          "By feature family: " + ", ".join(f"{k[0]}/{k[1]} {v:.4f}" for k, v in gfam.items()) + ".", "",
          "Top sub-families: " + ", ".join(f"{k[0]}/{k[1]}/{k[2]} {v:.4f}" for k, v in gsub.items()) + ".", "",
          f"Selected-feature count in the frozen refit: {len(fz['selected_features'])} of {len(names)}.", "",
          "## Cell coverage (dev)", "", "| cell | group | rows / cases | observed Δ (ladder) | subset-mean Δ | LODO cell pred |", "|---|---|--:|--:|--:|--:|"]
    for i, r in enumerate(cu.itertuples()):
        L.append(f"| {r.name} | {r.fam} | {r.n_rows}/{r.n_full} | {r.json_delta:+.2f} | {r.sub_mean:+.2f} | {fit['cell_pred'][i]:+.2f} |")
    (C.TABLES / f"xds_full_classifier{SUF}.md").write_text("\n".join(L))
    print("\n".join(L[:60]), flush=True)

    # ------------------------------------------------------------------ plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    colors = {"brats": "#4c72b0", "breast": "#dd8452", "onharmony": "#55a868", "toothfairy2": "#8172b3", "chaos": "#c44e52", "openms": "#222222"}
    for gname in sorted(set(cu.fam)):
        m = (cu.fam == gname).to_numpy()
        ax[0].scatter(fit["cell_pred"][m], fit["cell_truth"][m], s=26, c=colors[gname], alpha=.75, label=f"dev {gname}")
    for gname, x in c.groupby("fam"):
        ax[0].scatter(x.score, x.observed, marker="*", s=230, c=colors[gname], edgecolors="k", label=f"TEST {gname}")
    ax[0].axhline(0, c="grey", lw=.6)
    ax[0].axvline(0, c="grey", lw=.6)
    ax[0].set_xlabel("predicted cell delta Dice (pts)  [dev: leave-one-dataset-out]")
    ax[0].set_ylabel("observed cell delta Dice (pts)")
    ax[0].set_title(f"dev LODO rho={fit['rho']:+.2f} (perm p={fit['pval']:.2f}); TEST rho={res['rho']:+.2f}, sign {res['hits']}/{res['n']}", fontsize=9)
    ax[0].legend(fontsize=6.5, ncol=2)
    ax[1].hist(nb, bins=18, color="lightgrey")
    ax[1].axvline(fit["rho"], c="r", label=f"observed {fit['rho']:+.2f}")
    ax[1].set_xlabel("best-candidate LODO cell Spearman under permuted cell means")
    ax[1].set_title("selection-aware null", fontsize=9)
    ax[1].legend(fontsize=8)
    pv = imp.copy()
    pv["fam"] = pv["kind"] + "/" + pv["family"]
    pt = pv.pivot_table(index="fam", columns="src", values="importance", aggfunc="sum", fill_value=0)
    pt = pt.loc[pt.sum(1).sort_values().index]
    left = np.zeros(len(pt))
    for s, col in zip(("E", "T", "D"), ("#4c72b0", "#dd8452", "#55a868")):
        if s in pt:
            ax[2].barh(pt.index, pt[s], left=left, color=col, label=f"{s}")
            left += pt[s].to_numpy()
    ax[2].set_xlabel("permutation importance (cell-MSE increase)")
    ax[2].set_title("importance by family / source (E eval, T train, D diff)", fontsize=9)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.PLOTS / f"xds_full_classifier{SUF}.png", dpi=130)


if __name__ == "__main__":
    main()
