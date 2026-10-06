"""DIAGNOSTIC (isles2022, 2026-10-06): case-level failure modes behind the headline table. For every method and every (training contrast -> test item) cell: median Dice, share of
cases with Dice < 0.05 (lesion missed / empty), and mean Dice by lesion-volume tertile (volumes from 4_splits partition.json). Reads the 66 eval_all.csv; writes a markdown table
to 8_results_isles2022/02_metrics/isles2022_model/diag_case_level.md. Not a benchmark table."""
import csv, json, sys
from pathlib import Path
import numpy as np

DS = Path(__file__).resolve().parents[2]
M = DS / "8_results_isles2022/02_metrics/isles2022_model"
vol = json.loads((DS / "4_splits_isles2022/partition.json").read_text())["lesion_ml"]
t1, t2 = np.percentile(list(vol.values()), [33.3, 66.7])
rows = {}
for tc in ("dwi", "flair"):
    for rd in sorted((M / tc).glob("*_isles2022_*")):
        if rd.name.startswith("ablations") or not rd.is_dir():
            continue
        meth = rd.name.split(f"isles2022_{tc}_")[1].rsplit("_2026", 1)[0]
        for item in ("dwi", "flair"):
            d = {}
            for f in range(3):
                p = rd / f"fold{f}" / "eval_all.csv"
                for r in csv.DictReader(open(p)):
                    if r["group"] == item:
                        d.setdefault(r["case"], []).append(float(r["dice"]))
            rows[(tc, item, meth)] = {c: float(np.mean(v)) for c, v in d.items()}
out = [f"# isles2022 case-level diagnostics (lesion-volume tertile edges: {t1:.1f} / {t2:.1f} ml)\n",
       "| train -> test | method | n | mean Dice | median | % Dice<0.05 | small | mid | large |", "|---|---|---|---|---|---|---|---|---|"]
order = ["baseline", "auglab_default", "srcsm", "synthseg_EM", "synthseg_noEM", "auglabAug_v26_6_2_train050_val000"]
for tc in ("dwi", "flair"):
    for item in ("dwi", "flair"):
        for m in order:
            c = rows.get((tc, item, m))
            if not c: continue
            v = np.array(list(c.values())); vv = np.array([vol[k] for k in c])
            tert = [v[(vv <= t1)].mean(), v[(vv > t1) & (vv <= t2)].mean(), v[vv > t2].mean()]
            out.append(f"| {tc}->{item}{' (in-domain)' if tc == item else ''} | {m} | {len(v)} | {100*v.mean():.1f} | {100*np.median(v):.1f} | {100*(v < 0.05).mean():.0f}% | "
                       + " | ".join(f"{100*t:.1f}" for t in tert) + " |")
(M / "diag_case_level.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
