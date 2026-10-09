#!/usr/bin/env python
"""2026-10-07 re-run of the threshold-rule search (tune_rule_brats_breast.py protocol, unchanged) on the refreshed
outcomes (rung 5 = val000 retrains). Two tuning sets, each frozen before its single test on chaos + open-ms:
  A. BraTS only (48 OOD region cells, t1c arm included)      -> frozen_rule_brats_20261007.json
  B. BraTS + breast (48 + 9)                                 -> frozen_rule_brats_breast_20261007.json
Search: 1-3 conditions on mism/tex/vis within-(family,region) percentiles, thresholds .25/.5/.75, & / | left to right;
objective = sign accuracy on the significant tuning cells, then (failures caught - false alarms), then fewer conditions.
Output: outputs/tables/tune_rule_v2_20261007.md"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset_fillswap_model as X
import rule_common as R

W = R.W; TAB = W.parent / "tables"
Q = [0.25, 0.5, 0.75]
CONDS = [(f, d, q) for f in ("mism", "tex", "vis") for d in ("lo", "hi") for q in Q]

hol = pd.read_csv(W / "region_fill_swap_significance.csv"); hol = hol[hol.family == "OOD"]
sigb = {f"brats:{r.train}->{r.eval}:{r.region}": bool(r.significant) for r in hol.itertuples()}
def with_sig(df):
    df = df.copy(); df["sig"] = [sigb[n] if f == "brats" else p < 0.05 for n, f, p in zip(df["name"], df["fam"], df["p"])]
    return df
kf = R.key_features()
allc = X.dev_cells(); om = X.openms_cells()
test = R.feats(with_sig(pd.concat([allc[allc.fam == "chaos"], om])), kf)

L = ["# Threshold rule re-tuned on the val000 outcomes (2026-10-07), tested once on chaos + open-ms", "",
     "Protocol identical to tune_rule_brats_breast.py (2026-10-02). Outcomes: rung 5 val000 retrains; BraTS 48 OOD cells "
     "(t1c arm included); breast = 9 companion/own cells (ladders regenerated 2026-10-07); chaos = 6 grouped cross-dataset cells; "
     "open-ms = 4 cells.", ""]
for tag, fams in (("brats", ["brats"]), ("brats_breast", ["brats", "breast"])):
    dev = R.feats(with_sig(allc[allc.fam.isin(fams)]), kf)
    S = dev[dev.sig]; y = np.sign(S.delta.values); fail = y < 0
    res = []
    for k in (1, 2, 3):
        for combo in itertools.combinations(CONDS, k):
            if len({c[0] for c in combo}) < k:
                continue
            for ops in itertools.product("&|", repeat=k - 1):
                pred = R.apply_rule(S, [list(c) for c in combo], list(ops))
                m = pred < 0
                res.append(dict(combo=[list(c) for c in combo], ops=list(ops), k=k, acc=(pred == y).mean(),
                                caught=int((m & fail).sum()), fa=int((m & ~fail).sum())))
    r = pd.DataFrame(res); r["score"] = r.caught - r.fa
    best = r.sort_values(["acc", "score", "k"], ascending=[False, False, True]).iloc[0]
    rule = dict(rule=f"noise wins iff {R.rule_text(best.combo, best.ops)}", combo=best.combo, ops=best.ops,
                tuned_on=f"{tag}: {len(dev)} cells, {len(S)} significant, {int(fail.sum())} failures", frozen="2026-10-07, before the chaos/open-ms scoring below")
    json.dump(rule, open(W / f"frozen_rule_{tag}_20261007.json", "w"), indent=1)
    ties = int((r.acc == best.acc).sum())
    L += [f"## Tuning set {tag}: {len(dev)} cells ({', '.join(f'{f} {int((dev.fam == f).sum())}' for f in fams)}); significant {len(S)}, failures {int(fail.sum())}", "",
          f"FROZEN: `noise wins iff {R.rule_text(best.combo, best.ops)}` -- dev sig acc {best.acc:.2f}, failures caught {best.caught}/{int(fail.sum())}, "
          f"false alarms {best.fa}; always-helps {np.mean(y > 0):.2f}; {len(r)} rules tried, {ties} tied at the best accuracy.", ""]
    for f in fams:
        d = S[S.fam == f]; L.append(f"- dev {f}: sig acc {(R.apply_rule(d, best.combo, best.ops) == np.sign(d.delta.values)).mean():.2f} (n={len(d)})")
    L += ["", "### TEST (scored once)", "", "| cell | delta | p | pred | hit |", "|---|--:|--:|:--:|:--:|"]
    pred = R.apply_rule(test, best.combo, best.ops)
    for (_, c), p in zip(test.iterrows(), pred):
        L.append(f"| {c['name']} | {c['delta']:+.2f} | {c['p']:.3g} | {'noise' if p < 0 else 'real'} | {'YES' if p == np.sign(c['delta']) else 'no'} |")
    L.append("")
    for f in ("chaos", "openms"):
        d = test[test.fam == f]; s = R.score(d, R.apply_rule(d, best.combo, best.ops))
        L.append(f"- {f}: hits {s['hits']}/{s['n']}; significant {s['hits_sig']}/{s['n_sig']}; always-helps sig {s['always_helps_sig']}/{s['n_sig']}")
    s = R.score(test, pred)
    L += [f"- combined: {s['hits']}/{s['n']} (always-helps {int((np.sign(test.delta) > 0).sum())}/{s['n']}); significant {s['hits_sig']}/{s['n_sig']} (always-helps {s['always_helps_sig']}/{s['n_sig']})", ""]
out = TAB / "tune_rule_v2_20261007.md"; out.write_text("\n".join(L)); print("\n".join(L)); print("wrote", out)
