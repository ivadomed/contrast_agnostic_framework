#!/usr/bin/env python
"""
Aggregate Level-1 texture metrics + paired significance tests.

Reads the long CSV(s) from compute_texture_metrics.py (accepts a glob so the 4 rank
shards can be combined), then:
  1. averages variants within (method, subject, session, roi)  → per-volume-ROI values
     (avoids pseudo-replication before stats),
  2. writes a per-method pooled summary (summary.md / .csv),
  3. writes a per-method × ROI table (per_roi.csv),
  4. runs paired Wilcoxon signed-rank: PALETTE vs each of synthseg_em / synthseg_noem,
     pooled over volumes (per-volume mean across ROIs), per metric, with rank-biserial
     effect size (stats.md / .csv).

Usage:
  python aggregate_texture_metrics.py \
      --input "outputs/data/metrics_rank*.csv" \
      --output-dir outputs/tables
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# All numeric metric columns (ngf_e0p5/ngf/ngf_e2p0 = NGF η-grid; nmi secondary; lncc robustness).
METRICS      = ["ngf_e0p5", "ngf", "ngf_e2p0", "nmi", "lncc"]
STAT_METRICS = ["ngf", "nmi", "lncc"]    # headline NGF (η×1.0) + secondary + robustness
REFERENCE = "palette"
COMPARE_AGAINST = ["synthseg_em", "synthseg_noem"]


def load(input_glob: str) -> pd.DataFrame:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise SystemExit(f"No CSVs match {input_glob}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(files)} file(s)")
    return df


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Paired rank-biserial effect size for a>b (a,b same length)."""
    d = a - b
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().values
    rpos = ranks[d > 0].sum()
    rneg = ranks[d < 0].sum()
    return float((rpos - rneg) / ranks.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="glob for metric CSV shard(s)")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.input)

    # 1. average variants → per (method, subject, session, roi)
    keys = ["method", "subject", "session", "roi_id"]
    per_vol = df.groupby(keys, as_index=False)[METRICS].mean()

    # 2. pooled per-method summary (mean over all volume-ROI units)
    summ = per_vol.groupby("method")[METRICS].agg(["mean", "std", "median"])
    summ.to_csv(args.output_dir / "summary.csv")
    with open(args.output_dir / "summary.md", "w") as fh:
        fh.write("# Level-1 texture metrics — per-method summary\n\n")
        fh.write("Mean over all (volume × ROI) units, after averaging variants. "
                 "**NGF** = canonical Normalized Gradient Fields (PRIMARY texture; **1/3 = "
                 "random/no-texture floor**, →1 = preserved; read relatively). `ngf`=headline "
                 "(η×1.0); `ngf_e0p5`/`ngf_e2p0`=η×0.5/×2 robustness grid. NMI ∈ [1,2] "
                 "(secondary content); |LNCC| ∈ [0,1] (robustness appendix).\n\n")
        cols = " | ".join(f"{m.upper()} mean | {m.upper()} median" for m in METRICS)
        fh.write(f"| method | {cols} |\n")
        fh.write("|---|" + "---|" * (2 * len(METRICS)) + "\n")
        for m in summ.index:
            vals = " | ".join(f"{summ.loc[m,(mt,'mean')]:.3f} | {summ.loc[m,(mt,'median')]:.3f}"
                              for mt in METRICS)
            fh.write(f"| {m} | {vals} |\n")

    # 3. per-method × ROI table (mean over volumes)
    per_roi = per_vol.groupby(["roi_id", "method"])[METRICS].mean().reset_index()
    for mt in METRICS:
        per_roi.pivot(index="roi_id", columns="method", values=mt).to_csv(
            args.output_dir / f"per_roi_{mt}.csv")

    # 4. paired Wilcoxon: per-volume mean over ROIs, PALETTE vs competitors
    vol_mean = per_vol.groupby(["method", "subject", "session"], as_index=False)[METRICS].mean()
    stats_rows = []
    for metric in STAT_METRICS:
        ref = vol_mean[vol_mean.method == REFERENCE].set_index(["subject", "session"])[metric]
        for comp in COMPARE_AGAINST:
            other = vol_mean[vol_mean.method == comp].set_index(["subject", "session"])[metric]
            common = ref.index.intersection(other.index)
            a, b = ref.loc[common].values, other.loc[common].values
            if len(common) < 5:
                stats_rows.append(dict(metric=metric, comparison=f"{REFERENCE} vs {comp}",
                                       n=len(common), p_value=np.nan, effect=np.nan,
                                       median_ref=np.nan, median_comp=np.nan))
                continue
            stat, pval = wilcoxon(a, b)
            stats_rows.append(dict(metric=metric, comparison=f"{REFERENCE} vs {comp}",
                                   n=len(common), p_value=pval, effect=rank_biserial(a, b),
                                   median_ref=float(np.median(a)), median_comp=float(np.median(b))))
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(args.output_dir / "stats.csv", index=False)
    with open(args.output_dir / "stats.md", "w") as fh:
        fh.write("# Level-1 texture metrics — paired Wilcoxon (per-volume mean over ROIs)\n\n")
        fh.write("Effect = paired rank-biserial (positive → PALETTE higher). "
                 "p from Wilcoxon signed-rank.\n\n")
        fh.write("| metric | comparison | n | median PALETTE | median comp | effect | p |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for _, r in stats.iterrows():
            fh.write(f"| {r.metric} | {r.comparison} | {int(r.n)} | {r.median_ref:.3f} | "
                     f"{r.median_comp:.3f} | {r.effect:+.3f} | {r.p_value:.2e} |\n")

    print(f"Wrote summary.md, per_roi_*.csv, stats.md → {args.output_dir}")


if __name__ == "__main__":
    main()
