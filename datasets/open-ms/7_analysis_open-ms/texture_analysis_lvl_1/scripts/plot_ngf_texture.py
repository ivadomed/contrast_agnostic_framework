#!/usr/bin/env python
"""
Plot the adopted NGF texture-preservation results (see LITERATURE_REVIEW.md, FINDINGS.md).

Scope: `foreground`-interior only (erode x3, n=30, whole-brain, properly powered) — the
general structure-preservation claim. The lesion ROI is NOT reported here: it is open-ms's
only real label but is not the analysis's target (the claim is general structure preservation,
not MS-lesion-specific texture), and it carries an unresolved boundary-given-for-free confound
that erosion cannot fix without destroying the sample (see LITERATURE_REVIEW.md
§Boundary-artifact diagnosis for why — kept there as methodology/reasoning, not as a result).

FLAIR and T1w are plotted as SEPARATE files (not grouped in one chart):
  ngf_headline_<contrast>.png              : bar chart, mean NGF per method.
  ngf_violin_foreground_interior_<contrast>.png : per-subject distribution behind the headline
    paired-Wilcoxon test.

No reference/floor line is drawn (by request) — the bars/violins carry the comparison on their
own. (The 1/3 chance-floor derivation + empirical confirmation still lives in
LITERATURE_REVIEW.md for anyone who wants the reasoning; it just isn't drawn on the chart.)

Colors: fixed categorical order (dataviz skill reference palette, slots 1-5), one color per
method, consistent across all files.

Usage:
  python plot_ngf_texture.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
DATA = THIS_DIR.parent / "outputs" / "data"
OUT = THIS_DIR.parent / "outputs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["palette", "auglab_default", "synthseg_em", "synthseg_noem", "baseline_kmeans_label_remap_voronoi"]
METHOD_LABEL = {"palette": "palette", "auglab_default": "auglab", "synthseg_em": "synthseg_em",
                "synthseg_noem": "synthseg_noem", "baseline_kmeans_label_remap_voronoi": "kmeans_label_remap_voronoi"}
# dataviz skill reference palette (references/palette.md), slots 1-5, fixed order (not cycled).
COLOR = {
    "palette":              "#2a78d6",  # slot 1 blue
    "auglab_default":       "#1baf7a",  # slot 2 aqua
    "synthseg_em":          "#eda100",  # slot 3 yellow
    "synthseg_noem":        "#008300",  # slot 4 green
    "baseline_kmeans_label_remap_voronoi": "#4a3aa7",  # slot 5 violet
}
CONTRASTS = [("flair", "FLAIR"), ("t1w", "T1w")]
# (set-suffix, file-suffix, title annotation) -- "noblur" is the primary isolated-fill ablation;
# "blur" is the confirmatory with-blur, spatial-matched, usual-deployed-config run.
CONDITIONS = [("noblur", "", ""), ("blur", "_blur", ", usual blur")]


def load(pattern):
    files = sorted(glob.glob(str(DATA / pattern)))
    if not files:
        raise SystemExit(f"No CSVs match {pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def per_subject(df, roi):
    sub = df[df.roi_id == roi]
    return sub.groupby(["method", "subject"])[["ngf_all", "ngf_edge"]].mean().reset_index()


def plot_headline(contrast, label, cond, file_suffix, title_note):
    fg = load(f"ngf_*_{cond}_eroded_rank*.csv")
    fg = fg[(fg.roi_id == "foreground") & (fg.set == f"{contrast}_{cond}_eroded")]
    methods = [m for m in METHOD_ORDER if m in set(fg.method)]
    means = fg.groupby("method")["ngf_all"].mean()

    fig, ax = plt.subplots(figsize=(7, 5.2))
    x = range(len(methods))
    bars = ax.bar(x, [means[m] for m in methods], 0.6,
                  color=[COLOR[m] for m in methods], edgecolor="k", linewidth=0.6)
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=20, ha="right")
    ax.set_ylabel("NGF (ngf_all, ↑)"); ax.set_ylim(0, 1.0)
    ax.set_title(f"{label} — structure preservation (foreground-interior, n=30{title_note})")
    ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"ngf_headline_{contrast}{file_suffix}.png", dpi=150); plt.close(fig)


def plot_violin(contrast, label, cond, file_suffix, title_note):
    fg = load(f"ngf_*_{cond}_eroded_rank*.csv")
    fg = fg[fg.set == f"{contrast}_{cond}_eroded"]
    methods = [m for m in METHOD_ORDER if m in set(fg.method)]

    fig, ax = plt.subplots(figsize=(7, 5))
    pv = per_subject(fg, "foreground")
    data = [pv[pv.method == m]["ngf_all"].dropna().values for m in methods]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for i, m in enumerate(methods):
        parts["bodies"][i].set_facecolor(COLOR[m]); parts["bodies"][i].set_alpha(0.65)
    parts["cmedians"].set_color("#0b0b0b")
    ax.set_xticks(range(1, len(methods)+1))
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=20, ha="right")
    ax.set_title(f"{label} — per-subject foreground-interior NGF (n=30{title_note})")
    ax.set_ylabel("NGF (ngf_all, ↑)")
    ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"ngf_violin_foreground_interior_{contrast}{file_suffix}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    written = []
    for cond, file_suffix, title_note in CONDITIONS:
        for contrast, label in CONTRASTS:
            plot_headline(contrast, label, cond, file_suffix, title_note)
            plot_violin(contrast, label, cond, file_suffix, title_note)
            written.append(f"ngf_headline_{contrast}{file_suffix}.png / "
                            f"ngf_violin_foreground_interior_{contrast}{file_suffix}.png")
    print(f"Wrote {', '.join(written)} → {OUT}")
