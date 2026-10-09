#!/usr/bin/env python
"""Tune a threshold rule (noise-fill vs real-fill winner) on BraTS + breast ladder cells, freeze it, then score
chaos and open-ms once. Features per eval/train key: vis = target-vs-ring AUC (con_auc), tex = target high-pass
std (R_hp_std), mism = |R_mean eval - R_mean train|; percentiles within (dataset family, region)."""
import glob, itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset_fillswap_model as X
W = Path(__file__).resolve().parent.parent / "outputs" / "data"

h = pd.concat([pd.read_csv(f, usecols=["key", "case", "R_mean", "R_hp_std", "con_auc"]) for f in glob.glob(str(W / "xds_full_hand_shard*.csv"))])
kf = h.groupby("key")[["R_mean", "R_hp_std", "con_auc"]].mean()
b = pd.concat([pd.read_csv(f, usecols=["patient", "contrast", "region", "R_mean", "R_hp_std", "con_auc"]) for f in glob.glob(str(W / "bigfeat_shard*.csv"))]).drop_duplicates(["patient", "contrast", "region"])
for (c, r), g in b.groupby(["contrast", "region"]):
    kf.loc[f"brats:{c}:{r}"] = g[["R_mean", "R_hp_std", "con_auc"]].mean()

def feats(cells):
    cells = cells.copy()
    cells["region"] = [e.split(":")[2] if e.startswith("brats") else "target" for e in cells["eval"]]
    cells["vis"] = [kf.loc[e, "con_auc"] for e in cells["eval"]]
    cells["tex"] = [kf.loc[e, "R_hp_std"] for e in cells["eval"]]
    cells["mism"] = [abs(kf.loc[e, "R_mean"] - kf.loc[t, "R_mean"]) for t, e in zip(cells["train"], cells["eval"])]
    for c in ("vis", "tex", "mism"):
        cells[c + "_q"] = cells.groupby(["fam", "region"])[c].rank(pct=True)
    return cells

dev = X.dev_cells(); dev = feats(dev[dev.fam.isin(["brats", "breast"])])
hol = pd.read_csv(W / "region_fill_swap_significance.csv"); hol = hol[hol.family == "OOD"]
sigb = {f"brats:{r.train}->{r.eval}:{r.region}": bool(r.significant) for r in hol.itertuples()}
dev["sig"] = [sigb[n] if f == "brats" else p < 0.05 for n, f, p in zip(dev["name"], dev["fam"], dev["p"])]
S = dev[dev.sig]; y = np.sign(S.delta.values); fail = y < 0
print(f"tuning cells: {len(dev)} (brats {int((dev.fam=='brats').sum())}, breast {int((dev.fam=='breast').sum())}); significant {len(S)}, failures {int(fail.sum())}")

Q = [0.25, 0.5, 0.75]
def cond(df, f, d, q): return ((df[f + "_q"] <= q) if d == "lo" else (df[f + "_q"] > q)).values
conds = [(f, d, q) for f in ("mism", "tex", "vis") for d in ("lo", "hi") for q in Q]
res = []
for k in (1, 2, 3):
    for combo in itertools.combinations(conds, k):
        if len({c[0] for c in combo}) < k: continue
        for ops in itertools.product("&|", repeat=k - 1):
            m = cond(S, *combo[0])
            for o, c in zip(ops, combo[1:]):
                m = m & cond(S, *c) if o == "&" else m | cond(S, *c)
            pred = np.where(m, -1, 1)
            res.append(dict(combo=combo, ops=ops, k=k, acc=(pred == y).mean(), caught=int((m & fail).sum()), fa=int((m & ~fail).sum())))
r = pd.DataFrame(res); r["score"] = r.caught - r.fa
best = r.sort_values(["acc", "score", "k"], ascending=[False, False, True]).iloc[0]
def txt(c): return f"{c[0]}_q {'<=' if c[1]=='lo' else '>'} {c[2]}"
rule_txt = " ".join([txt(best.combo[0])] + [f"{o} {txt(c)}" for o, c in zip(best.ops, best.combo[1:])])
print(f"rules tried {len(r)}; rules tied at best acc {int((r.acc==best.acc).sum())}")
print(f"FROZEN: noise wins iff {rule_txt} (left-to-right); dev sig acc {best.acc:.2f}, failures caught {best.caught}/{int(fail.sum())}, false alarms {best.fa}; always-helps {np.mean(y>0):.2f}")
json.dump(dict(rule=f"noise wins iff {rule_txt}", combo=[list(c) for c in best.combo], ops=list(best.ops)), open(W / "frozen_rule_brats_breast_20261002.json", "w"))

def apply(df):
    m = cond(df, *best.combo[0])
    for o, c in zip(best.ops, best.combo[1:]):
        m = m & cond(df, *c) if o == "&" else m | cond(df, *c)
    return np.where(m, -1, 1)
for f in ("brats", "breast"):
    d = S[S.fam == f]; print(f"  dev {f}: sig acc {(apply(d)==np.sign(d.delta.values)).mean():.2f} (n={len(d)})")

print("\n=== TEST (scored once) ===")
allc = X.dev_cells()
test = pd.concat([allc[allc.fam == "chaos"], X.openms_cells()])
test = feats(test); test["pred"] = apply(test); test["hit"] = test.pred == np.sign(test.delta)
print(test[["name", "delta", "p", "pred", "hit"]].round(3).to_string(index=False))
for f in ("chaos", "openms"):
    d = test[test.fam == f]; s = d[d.p < 0.05]
    print(f"{f}: hits {int(d.hit.sum())}/{len(d)}; significant {int(s.hit.sum())}/{len(s)}; always-helps sig {int((s.delta>0).sum())}/{len(s)}")
