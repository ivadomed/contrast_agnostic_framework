#!/usr/bin/env python
"""
Plot Level-1 texture metrics.

Reads the long CSV(s) from compute_texture_metrics.py and produces:
  1. violin/box of each metric per method       (the vertical-axis proof:
     SynthSeg near the texture-destroying floor, PALETTE + image-driven controls high),
  2. per-method × 31-ROI heatmap for each metric (preservation is uniform across anatomy).

The 2×2 (texture × contrast-coverage) figure is assembled separately, once the
contrast-manifold re-run provides the coverage axis — not produced here.

Usage:
  python plot_texture_metrics.py --input "outputs/data/metrics_rank*.csv" \
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

METRICS = {"ngf":  "NGF (texture, ↑; 1/3=no-texture floor)",
           "nmi":  "NMI  (content preservation, ↑)",
           "lncc": "|LNCC|  (robustness appendix, ↑)"}
ETA_GRID_COLS = [("ngf_e0p5", 0.5), ("ngf", 1.0), ("ngf_e2p0", 2.0)]   # η robustness
# order: ours, then competitors, then controls
METHOD_ORDER = ["palette", "synthseg_em", "synthseg_noem", "auglab_default", "gamma", "histeq"]


def load(input_glob: str) -> pd.DataFrame:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise SystemExit(f"No CSVs match {input_glob}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def roi_names(labels_json: Path | None) -> dict[int, str]:
    if labels_json is None or not labels_json.exists():
        return {}
    labels = json.load(open(labels_json)).get("labels", {})
    return {int(v): k for k, v in labels.items() if int(v) != 0}


def order_present(df) -> list[str]:
    return [m for m in METHOD_ORDER if m in set(df.method)]


def plot_violins(df, out_dir):
    methods = order_present(df)
    per_vol = df.groupby(["method", "subject", "session", "roi_id"], as_index=False)[list(METRICS)].mean()
    for metric, label in METRICS.items():
        data = [per_vol[per_vol.method == m][metric].dropna().values for m in methods]
        fig, ax = plt.subplots(figsize=(1.4 * len(methods) + 2, 4.5))
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for i, m in enumerate(methods):
            color = "#2166ac" if m == "palette" else ("#b2182b" if m.startswith("synthseg") else "#999999")
            parts["bodies"][i].set_facecolor(color)
            parts["bodies"][i].set_alpha(0.65)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"Texture preservation — {metric.upper()}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"violin_{metric}.png", dpi=150)
        plt.close(fig)


def plot_heatmaps(df, out_dir, names):
    methods = order_present(df)
    per_roi = (df.groupby(["roi_id", "method"], as_index=False)[list(METRICS)].mean())
    rois = sorted(per_roi.roi_id.unique())
    ylabels = [names.get(r, str(r)) for r in rois]
    for metric in METRICS:
        mat = np.full((len(rois), len(methods)), np.nan)
        for i, r in enumerate(rois):
            for j, m in enumerate(methods):
                sel = per_roi[(per_roi.roi_id == r) & (per_roi.method == m)]
                if len(sel):
                    mat[i, j] = sel[metric].values[0]
        fig, ax = plt.subplots(figsize=(1.3 * len(methods) + 3, 0.32 * len(rois) + 2))
        vmin = {"nmi": 1.0, "ngf": 1.0 / 3}.get(metric, 0.0)
        vmax = 2.0 if metric == "nmi" else 1.0
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_yticks(range(len(rois)));    ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_title(f"{metric.upper()} per ROI × method")
        fig.colorbar(im, ax=ax, shrink=0.6, label=metric.upper())
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{metric}.png", dpi=150)
        plt.close(fig)


def plot_eta_robustness(df, out_dir):
    """NGF vs η multiplier, one line per method — shows the PALETTE≫SynthSeg direction holds."""
    cols = [c for c, _ in ETA_GRID_COLS if c in df.columns]
    if len(cols) < 2:
        return
    methods = order_present(df)
    per_vol = df.groupby(["method", "subject", "session", "roi_id"], as_index=False)[cols].mean()
    xs = [mult for c, mult in ETA_GRID_COLS if c in df.columns]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in methods:
        sub = per_vol[per_vol.method == m]
        ys = [sub[c].mean() for c, _ in ETA_GRID_COLS if c in df.columns]
        color = "#2166ac" if m == "palette" else ("#b2182b" if m.startswith("synthseg") else "#999999")
        ax.plot(xs, ys, marker="o", label=m, color=color)
    ax.axhline(1.0 / 3, ls="--", c="k", alpha=0.5, label="1/3 floor")
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("η multiplier (× MAD noise estimate)"); ax.set_ylabel("NGF (mean)")
    ax.set_title("NGF η-robustness"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "eta_robustness.png", dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--labels-json", type=Path, default=None)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.input)
    names = roi_names(args.labels_json)
    plot_violins(df, args.output_dir)
    plot_heatmaps(df, args.output_dir, names)
    plot_eta_robustness(df, args.output_dir)
    print(f"Wrote violin_*.png, heatmap_*.png, eta_robustness.png → {args.output_dir}")


if __name__ == "__main__":
    main()
