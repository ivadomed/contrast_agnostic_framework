#!/usr/bin/env python
"""Shared pieces of the threshold-rule study (tune_rule_brats_breast.py, 2026-10-02; refreshed 2026-10-07):
per-key image features, percentile features per cell, rule application. Features per eval/train key:
vis = target-vs-ring AUC (con_auc), tex = target high-pass std (R_hp_std), mism = |R_mean eval - R_mean train|;
percentiles are ranks within (dataset family, region) over the cells handed in."""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

W = Path(__file__).resolve().parent.parent / "outputs" / "data"
FEATS = ("R_mean", "R_hp_std", "con_auc")


def key_features() -> pd.DataFrame:
    h = pd.concat([pd.read_csv(f, usecols=["key", "case", *FEATS]) for f in glob.glob(str(W / "xds_full_hand_shard*.csv"))])
    kf = h.groupby("key")[list(FEATS)].mean()
    b = pd.concat([pd.read_csv(f, usecols=["patient", "contrast", "region", *FEATS])
                   for f in glob.glob(str(W / "bigfeat_shard*.csv"))]).drop_duplicates(["patient", "contrast", "region"])
    for (c, r), g in b.groupby(["contrast", "region"]):
        kf.loc[f"brats:{c}:{r}"] = g[list(FEATS)].mean()
    return kf


def feats(cells: pd.DataFrame, kf: pd.DataFrame | None = None) -> pd.DataFrame:
    kf = key_features() if kf is None else kf
    cells = cells.copy()
    cells["region"] = [e.split(":")[2] if e.startswith("brats") else "target" for e in cells["eval"]]
    cells["vis"] = [kf.loc[e, "con_auc"] for e in cells["eval"]]
    cells["tex"] = [kf.loc[e, "R_hp_std"] for e in cells["eval"]]
    cells["mism"] = [abs(kf.loc[e, "R_mean"] - kf.loc[t, "R_mean"]) for t, e in zip(cells["train"], cells["eval"])]
    for c in ("vis", "tex", "mism"):
        cells[c + "_q"] = cells.groupby(["fam", "region"])[c].rank(pct=True)
    return cells


def cond(df, f, d, q):
    return ((df[f + "_q"] <= q) if d == "lo" else (df[f + "_q"] > q)).values


def apply_rule(df, combo, ops):
    """combo = [[feat, 'lo'|'hi', q], ...], ops = ['&'|'|', ...] applied left to right. Returns +1 (real-fill
    wins) / -1 (noise-fill wins) per row."""
    m = cond(df, *combo[0])
    for o, c in zip(ops, combo[1:]):
        m = m & cond(df, *c) if o == "&" else m | cond(df, *c)
    return np.where(m, -1, 1)


def rule_text(combo, ops):
    def txt(c):
        return f"{c[0]}_q {'<=' if c[1] == 'lo' else '>'} {c[2]}"
    return " ".join([txt(combo[0])] + [f"{o} {txt(c)}" for o, c in zip(ops, combo[1:])])


def score(df, pred):
    """Hits on all cells / significant cells, failures caught, false alarms, 'always helps' on the significant cells."""
    y = np.sign(df["delta"].values)
    sig = df["sig"].values.astype(bool)
    fail = (y < 0) & sig
    return dict(n=len(df), hits=int((pred == y).sum()), n_sig=int(sig.sum()), hits_sig=int(((pred == y) & sig).sum()),
                n_fail=int(fail.sum()), caught=int(((pred < 0) & fail).sum()), false_alarms=int(((pred < 0) & sig & ~fail).sum()),
                always_helps_sig=int(((y > 0) & sig).sum()))
