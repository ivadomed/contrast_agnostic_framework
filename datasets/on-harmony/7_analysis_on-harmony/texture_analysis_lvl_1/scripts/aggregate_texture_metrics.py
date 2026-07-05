#!/usr/bin/env python
"""
Aggregate Level-1 texture metrics + paired significance tests, split by set (blur / noblur / ref).

Reads the long CSV shard(s) from compute_texture_metrics.py (accepts a glob), then:
  1. averages variants within (set, method, subject, session, roi),
  2. per-(method × set) summary  (summary.md / .csv) — blur vs no-blur side by side,
  3. per-(roi × method × set) tables (per_roi_<metric>.csv),
  4. paired Wilcoxon PALETTE vs synthseg_em / synthseg_noem, within each set, per metric,
     with rank-biserial effect size (stats.md / .csv).

Metrics: census_r1 (PRIMARY texture, floor 0), census_r2 (robustness radius), nmi (content).

Usage:
  python aggregate_texture_metrics.py --input "outputs/data/metrics_*_rank*.csv" \
      --output-dir outputs/tables
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS   = ["census_r1", "census_r2", "nmi"]
REFERENCE = "palette"
COMPARE_AGAINST = ["synthseg_em", "synthseg_noem"]


def load(input_glob: str) -> pd.DataFrame:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise SystemExit(f"No CSVs match {input_glob}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "set" not in df.columns:
        df["set"] = "blur"
    print(f"Loaded {len(df)} rows from {len(files)} file(s); sets={sorted(df['set'].unique())}")
    return df


def rank_biserial(a, b) -> float:
    d = a - b
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().values
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.input)

    # 1. average variants → per (set, method, subject, session, roi)
    keys = ["set", "method", "subject", "session", "roi_id"]
    per_vol = df.groupby(keys, as_index=False)[METRICS].mean()

    # 2. per (method × set) summary
    summ = per_vol.groupby(["method", "set"])[METRICS].agg(["mean", "median"])
    summ.to_csv(args.output_dir / "summary.csv")
    with open(args.output_dir / "summary.md", "w") as fh:
        fh.write("# Level-1 texture metrics — per method × set\n\n")
        fh.write("Mean over all (volume × ROI) units, variants averaged. "
                 "**census_r1** = PRIMARY texture (rank/census correlation; **0 = no-texture "
                 "floor**, →1 = preserved; read relatively). census_r2 = radius-2 robustness. "
                 "NMI ∈ [1,2] = content.\n\n")
        cols = " | ".join(f"{m} mean" for m in METRICS)
        fh.write(f"| method | set | {cols} |\n")
        fh.write("|---|---|" + "---|" * len(METRICS) + "\n")
        for (method, s) in summ.index:
            vals = " | ".join(f"{summ.loc[(method, s), (m, 'mean')]:.3f}" for m in METRICS)
            fh.write(f"| {method} | {s} | {vals} |\n")

    # 3. per (roi × method × set) tables
    per_roi = per_vol.groupby(["roi_id", "method", "set"], as_index=False)[METRICS].mean()
    for m in METRICS:
        per_roi.pivot_table(index="roi_id", columns=["set", "method"], values=m).to_csv(
            args.output_dir / f"per_roi_{m}.csv")

    # 4. paired Wilcoxon within each set: per-volume mean over ROIs, PALETTE vs competitors
    vol_mean = per_vol.groupby(["set", "method", "subject", "session"], as_index=False)[METRICS].mean()
    rows = []
    for s in sorted(vol_mean["set"].unique()):
        sub = vol_mean[vol_mean["set"] == s]
        for metric in METRICS:
            ref = sub[sub.method == REFERENCE].set_index(["subject", "session"])[metric]
            for comp in COMPARE_AGAINST:
                oth = sub[sub.method == comp].set_index(["subject", "session"])[metric]
                common = ref.index.intersection(oth.index)
                if len(common) < 5:
                    continue
                a, b = ref.loc[common].values, oth.loc[common].values
                _, pv = wilcoxon(a, b)
                rows.append(dict(set=s, metric=metric, comparison=f"{REFERENCE} vs {comp}",
                                 n=len(common), median_ref=float(np.median(a)),
                                 median_comp=float(np.median(b)),
                                 effect=rank_biserial(a, b), p_value=pv))
    stats = pd.DataFrame(rows)
    stats.to_csv(args.output_dir / "stats.csv", index=False)
    with open(args.output_dir / "stats.md", "w") as fh:
        fh.write("# Level-1 texture metrics — paired Wilcoxon (per-volume mean over ROIs)\n\n")
        fh.write("Effect = paired rank-biserial (positive → PALETTE higher). "
                 "p = Wilcoxon signed-rank.\n\n")
        fh.write("| set | metric | comparison | n | median PALETTE | median comp | effect | p |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in stats.iterrows():
            fh.write(f"| {r.set} | {r.metric} | {r.comparison} | {int(r.n)} | {r.median_ref:.3f} "
                     f"| {r.median_comp:.3f} | {r.effect:+.3f} | {r.p_value:.2e} |\n")

    print(f"Wrote summary.md, per_roi_*.csv, stats.md → {args.output_dir}")


if __name__ == "__main__":
    main()
