#!/usr/bin/env python
"""
Plot Level-1 texture metrics (census + NMI), split by set (blur / noblur).

Reads the long CSV shard(s) from compute_texture_metrics.py and produces:
  1. headline bar — census_r1 mean per method, grouped by set (blur vs noblur), with the 0 floor;
  2. per-set violins of census_r1 per method;
  3. per-set heatmaps census_r1 per ROI × method;
  4. radius robustness — census_r1 vs census_r2 per method (blur set).

The 2×2 (texture × contrast-coverage) figure is assembled separately once the manifold re-run
provides the coverage axis.

Usage:
  python plot_texture_metrics.py --input "outputs/data/metrics_*_rank*.csv" \
      --output-dir outputs/plots --labels-json <Dataset031 dataset.json>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALL_METRICS = ["census_r1", "census_r2", "census_local8", "nmi"]   # any subset may be present
PRIMARY_CANDIDATES = ["census_r1", "census_local8"]                 # headline/violin per present one
METHOD_ORDER = ["palette", "synthseg_em", "synthseg_noem", "auglab_default", "gamma", "histeq"]
SET_ORDER = ["blur", "noblur", "ref"]


def present_metrics(df):
    return [m for m in ALL_METRICS if m in df.columns]


def load(g):
    files = sorted(glob.glob(g))
    if not files:
        raise SystemExit(f"No CSVs match {g}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "set" not in df.columns:
        df["set"] = "blur"
    return df


def roi_names(j):
    if j is None or not Path(j).exists():
        return {}
    return {int(v): k for k, v in json.load(open(j)).get("labels", {}).items() if int(v) != 0}


def mpresent(df):
    return [m for m in METHOD_ORDER if m in set(df.method)]


def spresent(df):
    """SET_ORDER first (on-harmony's blur/noblur/ref), then any other sets found (e.g. open-ms's
    flair/t1w, which repurpose the "set" column for source contrast rather than blur ablation) —
    sorted so unlisted sets still render instead of being silently dropped."""
    present = set(df["set"])
    return [s for s in SET_ORDER if s in present] + sorted(present - set(SET_ORDER))


def color_for(m):
    return "#2166ac" if m == "palette" else ("#b2182b" if m.startswith("synthseg") else "#7a7a7a")


def per_vol(df):
    return df.groupby(["set", "method", "subject", "session", "roi_id"], as_index=False)[
        present_metrics(df)].mean()


def plot_headline(df, out, primary="census_r1"):
    pv = per_vol(df)
    # exclude the monotone controls from the bars — they define the ceiling reference line
    methods = [m for m in mpresent(df) if m not in ("gamma", "histeq")]
    sets = [s for s in spresent(df) if s != "ref"]
    x = np.arange(len(methods)); w = 0.8 / max(1, len(sets))
    fig, ax = plt.subplots(figsize=(1.4 * len(methods) + 2, 4.8))
    for i, s in enumerate(sets):
        ys = [pv[(pv["set"] == s) & (pv.method == m)][primary].mean() for m in methods]
        ax.bar(x + i * w, ys, w, label=f"set={s}", edgecolor="k", linewidth=0.4,
               color=[color_for(m) for m in methods], alpha=0.65 if s != "blur" else 1.0)
    # ceiling = monotone-remap controls (gamma/histeq); floor = SynthSeg (or 0)
    ctrl = pv[pv.method.isin(["gamma", "histeq"])][primary]
    ss   = pv[pv.method.str.startswith("synthseg")][primary]
    if len(ctrl):
        c = ctrl.mean(); ax.axhline(c, color="green", ls="--", lw=1.2,
                                    label=f"monotone ceiling ≈ {c:.2f}")
    floor = ss.mean() if len(ss) else 0.0
    ax.axhline(floor, color="red", ls=":", lw=1.2, label=f"SynthSeg floor ≈ {floor:.2f}")
    ax.set_ylim(0, 1)
    ax.set_xticks(x + w * (len(sets) - 1) / 2); ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(f"{primary}  (texture, ↑)")
    ax.set_title(f"Texture preservation ({primary} |corr|) — read vs floor & ceiling")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(out / f"headline_{primary}.png", dpi=150); plt.close(fig)


def plot_violins(df, out, primary="census_r1"):
    pv = per_vol(df)
    for s in spresent(df):
        methods = [m for m in mpresent(df)
                   if len(pv[(pv["set"] == s) & (pv.method == m)][primary].dropna())]
        if not methods:
            continue
        data = [pv[(pv["set"] == s) & (pv.method == m)][primary].dropna().values for m in methods]
        fig, ax = plt.subplots(figsize=(1.3 * len(methods) + 2, 4.5))
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for i, m in enumerate(methods):
            parts["bodies"][i].set_facecolor(color_for(m)); parts["bodies"][i].set_alpha(0.6)
        ax.set_xticks(range(1, len(methods) + 1)); ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_ylabel(f"{primary} (↑)"); ax.set_title(f"Texture preservation ({primary}) — set={s}")
        ax.grid(axis="y", alpha=0.3); fig.tight_layout()
        fig.savefig(out / f"violin_{primary}_{s}.png", dpi=150); plt.close(fig)


def plot_heatmaps(df, out, names, primary="census_r1"):
    pv = per_vol(df)
    for s in spresent(df):
        sub = pv[pv["set"] == s]
        methods = [m for m in mpresent(df) if m in set(sub.method)]
        rois = sorted(sub.roi_id.unique())
        if not rois or not methods:
            continue
        mat = np.full((len(rois), len(methods)), np.nan)
        for i, r in enumerate(rois):
            for j, m in enumerate(methods):
                v = sub[(sub.roi_id == r) & (sub.method == m)][primary]
                if len(v):
                    mat[i, j] = v.mean()
        fig, ax = plt.subplots(figsize=(1.3 * len(methods) + 3, 0.32 * len(rois) + 2))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_yticks(range(len(rois))); ax.set_yticklabels([names.get(r, str(r)) for r in rois], fontsize=7)
        ax.set_title(f"{primary} per ROI × method — set={s}")
        fig.colorbar(im, ax=ax, shrink=0.6, label=primary); fig.tight_layout()
        fig.savefig(out / f"heatmap_{primary}_{s}.png", dpi=150); plt.close(fig)


def plot_radius_robustness(df, out):
    if "census_r2" not in df.columns:
        return
    pv = per_vol(df); methods = mpresent(df)
    sub = pv[pv["set"] == (spresent(df)[0])]  # first set (blur)
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(1.4 * len(methods) + 2, 4.5))
    ax.bar(x - 0.2, [sub[sub.method == m]["census_r1"].mean() for m in methods], 0.4, label="r=1")
    ax.bar(x + 0.2, [sub[sub.method == m]["census_r2"].mean() for m in methods], 0.4, label="r=2")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("census |corr| (↑)"); ax.set_title(f"Census radius robustness (set={spresent(df)[0]})")
    ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(out / "radius_robustness.png", dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--labels-json", type=Path, default=None)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.input)
    names = roi_names(args.labels_json)
    primaries = [m for m in PRIMARY_CANDIDATES if m in df.columns] or ["census_r1"]
    for primary in primaries:      # headline + violin + heatmap per present census variant
        plot_headline(df, args.output_dir, primary)
        plot_violins(df, args.output_dir, primary)
        plot_heatmaps(df, args.output_dir, names, primary)
    plot_radius_robustness(df, args.output_dir)
    print(f"Wrote headline/violin/heatmap for {primaries} + radius_robustness → {args.output_dir}")


if __name__ == "__main__":
    main()
