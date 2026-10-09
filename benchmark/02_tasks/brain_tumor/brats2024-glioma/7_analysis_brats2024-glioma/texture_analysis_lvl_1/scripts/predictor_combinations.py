#!/usr/bin/env python
"""
Can a COMBINATION of image predictors explain the noise->real fill-swap change (BraTS)?
Linear models of delta on all 1-3-predictor subsets of the 7 scorecard predictors, fitted on the 36
OOD cells, scored by leave-one-out (LOO) predictions: Spearman(LOO pred, delta) over 36 cells, sign
accuracy on the 14 Holm-significant cells, failures caught. The whole subset search is re-run on 1000
permutations of delta (shuffled across cells) to calibrate the best-of-search score (selection bias).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explanation_scorecard import load  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs"
PRED = {"NGF": "ngf_region_z", "eta2": "eta2_train_given_eval_z", "visibility": "vis_gap", "ramp": "R_gap",
        "margin": "margin", "energy": "E_mean_eval_minus_train", "anatomySI": "SI_gap"}
N_PERM, SEED = 1000, 0


def loo_pred(X, y):
    n = len(y)
    Xa = np.column_stack([np.ones(n), X])
    p = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        b, *_ = np.linalg.lstsq(Xa[m], y[m], rcond=None)
        p[i] = Xa[i] @ b
    return p


def search(X, y, sig_idx, sig_sign):
    rows = []
    for k in (1, 2, 3):
        for combo in itertools.combinations(range(X.shape[1]), k):
            p = loo_pred(X[:, combo], y)
            rho = spearmanr(p, y)[0]
            s = np.sign(p[sig_idx])
            rows.append((combo, rho, float((s == sig_sign).mean()), int(((s < 0) & (sig_sign < 0)).sum())))
    return rows


def main():
    t, _ = load()
    for c in ("ngf_region", "eta2_train_given_eval"):
        t[c + "_z"] = t.groupby("region")[c].transform(lambda s: (s - s.mean()) / s.std())
    names = list(PRED)
    X = t[[PRED[n] for n in names]].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    y = t["mean_delta_pts"].to_numpy(float)
    sig_idx = np.where(t["significant"].to_numpy())[0]
    sig_sign = np.sign(y[sig_idx])

    res = search(X, y, sig_idx, sig_sign)
    res.sort(key=lambda r: -r[1])
    best_rho, best_acc = res[0][1], max(r[2] for r in res)

    rng = np.random.default_rng(SEED)
    null_rho, null_acc = [], []
    for _ in range(N_PERM):
        yp = rng.permutation(y)
        rp = search(X, yp, sig_idx, np.sign(yp[sig_idx]))
        null_rho.append(max(r[1] for r in rp))
        null_acc.append(max(r[2] for r in rp))
    p_rho = (1 + sum(v >= best_rho for v in null_rho)) / (N_PERM + 1)
    p_acc = (1 + sum(v >= best_acc for v in null_acc)) / (N_PERM + 1)

    L = ["# Predictor combinations vs fill-swap Δ (BraTS, 36 OOD cells, leave-one-out)", "",
         f"Best LOO Spearman over all {len(res)} subsets = {best_rho:+.2f}; best-of-search permutation p = {p_rho:.3f} "
         f"(null 95th pct {np.percentile(null_rho, 95):+.2f}).",
         f"Best LOO sign accuracy on the 14 significant cells = {best_acc:.2f}; permutation p = {p_acc:.3f} "
         f"(null 95th pct {np.percentile(null_acc, 95):.2f}; 'always helps' = {np.mean(sig_sign > 0):.2f}).", "",
         "| predictors | LOO Spearman | sign acc (14 sig) | failures caught (of 4) |", "|---|--:|--:|--:|"]
    for combo, rho, acc, fc in res[:12]:
        L.append(f"| {' + '.join(names[i] for i in combo)} | {rho:+.2f} | {acc:.2f} | {fc} |")
    L += ["", "Single predictors:", "", "| predictor | LOO Spearman | sign acc | failures caught |", "|---|--:|--:|--:|"]
    for combo, rho, acc, fc in sorted([r for r in res if len(r[0]) == 1], key=lambda r: -r[1]):
        L.append(f"| {names[combo[0]]} | {rho:+.2f} | {acc:.2f} | {fc} |")
    (OUT / "tables" / "predictor_combinations.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
