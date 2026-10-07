#!/usr/bin/env python
"""Re-score the two rules frozen on 2026-10-02 (before the rung-5 retrain) on the REFRESHED fill-swap outcomes
(rung 5 = val000 retrains, 2026-10-06 ladders; BraTS per-region table rebuilt 2026-10-07 incl. the t1c arm).
The rules are untouched; only the outcomes changed, so this is an honest test of each rule on its own former
tuning cells (brats; brats+breast) and on the never-tuned sets (chaos, open-ms).
Also prints, for BraTS, every cell whose Holm call changed between the val100 and val000 tables.
Output: outputs/tables/rescore_frozen_rules_20261007.md"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset_fillswap_model as X
import rule_common as R

W = R.W
TAB = W.parent / "tables"
RULES = {
    "brats-only (frozen_rule_20261002)": dict(combo=[["tex", "lo", 0.75], ["vis", "lo", 0.5]], ops=["&"]),
    "brats+breast (frozen_rule_brats_breast_20261002)": json.load(open(W / "frozen_rule_brats_breast_20261002.json")),
}
L = ["# Frozen rules (2026-10-02) re-scored on the val000-retrain outcomes (2026-10-07)", "",
     "Rules unchanged; outcomes = rung 5 val000 retrains everywhere (BraTS per-region table rebuilt with the t1c arm; "
     "breast companion ladders regenerated; chaos ladders are the cross-dataset grouped ones since 2026-10-03; "
     "on-harmony T1w/T2w still val100 and not used here). 'sig' = Holm (BraTS) or raw p<0.05 (ladder per-contrast p)."
     " BraTS is reported both on all 48 OOD cells (percentiles ranked over 48) and on the original 36 (t1n/t2w/t2f-trained,"
     " percentiles ranked over 36, as when the rules were tuned).", ""]

# --- BraTS old vs new
new = pd.read_csv(W / "region_fill_swap_significance.csv"); new = new[new.family == "OOD"]
old = pd.read_csv(W / "region_fill_swap_significance.csv.bak_20261007_pre_val000"); old = old[old.family == "OOD"]
m = old.merge(new, on=["train", "eval", "region"], suffixes=("_old", "_new"))
chg = m[m.direction_old != m.direction_new]
L += ["## BraTS cells whose Holm call changed (val100 -> val000 rung 5)", "",
      "| train | eval | region | delta old | delta new | call old | call new |", "|---|---|---|--:|--:|---|---|"]
for r in chg.itertuples():
    L.append(f"| {r.train} | {r.eval} | {r.region} | {r.mean_delta_pts_old:+.1f} | {r.mean_delta_pts_new:+.1f} | {r.direction_old} | {r.direction_new} |")
L += ["", f"Unchanged calls: {len(m) - len(chg)}/{len(m)} of the 36 original cells. New t1c-trained cells: "
      + ", ".join(f"{r.eval}:{r.region} {r.direction} ({r.mean_delta_pts:+.1f})" for r in new[new.train == 't1c'].itertuples() if r.direction != 'n.s.')
      + " (significant ones)", ""]

# --- cells
dev = X.dev_cells(); om = X.openms_cells()
sigb = {f"brats:{r.train}->{r.eval}:{r.region}": bool(r.significant) for r in new.itertuples()}
def with_sig(df):
    df = df.copy(); df["sig"] = [sigb[n] if f == "brats" else p < 0.05 for n, f, p in zip(df["name"], df["fam"], df["p"])]
    return df
kf = R.key_features()
sets = {
    "brats (48 cells, 4 train arms)": R.feats(with_sig(dev[dev.fam == "brats"]), kf),
    "brats (36 cells, t1n/t2w/t2f arms)": R.feats(with_sig(dev[(dev.fam == "brats") & ~dev["name"].str.startswith("brats:t1c")]), kf),
    "breast (9)": R.feats(with_sig(dev[dev.fam == "breast"]), kf),
    "chaos (6)": R.feats(with_sig(dev[dev.fam == "chaos"]), kf),
    "open-ms (4)": R.feats(with_sig(om), kf),
}
for rname, rule in RULES.items():
    L += [f"## Rule: {rname}", "", f"`noise wins iff {R.rule_text(rule['combo'], rule['ops'])}`", "",
          "| cell set | hits all | hits sig | failures caught | false alarms | always-helps sig |", "|---|--:|--:|--:|--:|--:|"]
    for sname, df in sets.items():
        s = R.score(df, R.apply_rule(df, rule["combo"], rule["ops"]))
        L.append(f"| {sname} | {s['hits']}/{s['n']} | {s['hits_sig']}/{s['n_sig']} | {s['caught']}/{s['n_fail']} | {s['false_alarms']} | {s['always_helps_sig']}/{s['n_sig']} |")
    L.append("")
    # per-cell detail for the test sets + brats significant
    L += ["<details><summary>per-cell predictions</summary>", "", "| set | cell | delta | p | sig | pred | hit |", "|---|---|--:|--:|:--:|:--:|:--:|"]
    for sname, df in sets.items():
        if sname.startswith("brats (36"):
            continue
        pred = R.apply_rule(df, rule["combo"], rule["ops"])
        for (_, r), p in zip(df.iterrows(), pred):
            if sname.startswith("brats") and not r.sig:
                continue
            L.append(f"| {sname.split(' ')[0]} | {r['name']} | {r['delta']:+.2f} | {r['p']:.3g} | {'Y' if r.sig else ''} | {'noise' if p < 0 else 'real'} | {'YES' if p == np.sign(r['delta']) else 'no'} |")
    L += ["", "</details>", ""]
out = TAB / "rescore_frozen_rules_20261007.md"
out.write_text("\n".join(L)); print("\n".join(L)); print("wrote", out)
