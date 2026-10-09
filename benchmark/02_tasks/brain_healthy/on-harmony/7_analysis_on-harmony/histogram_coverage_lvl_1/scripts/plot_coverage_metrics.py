#!/usr/bin/env python
"""
Plot the per-group coverage results (Pillar 2):
  (a) macro-averaged Coverage per method (bootstrap 95% CI),
  (b) Vendi diversity per method (with real reference line),
  (c) per-(scanner×contrast)-group Coverage heatmap (method × group).
Only Coverage + Vendi are plotted (Precision/Recall/Density dropped — see GROUNDING_AUDIT.md).

Usage:
  run_job --gpus 0 --wait -- .venv/bin/python plot_coverage_metrics.py --output-dir outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL = {"palette": "PALETTE", "synthseg_em": "SynthSeg-EM",
         "synthseg_noem": "SynthSeg-noEM", "auglab_default": "auglab-default"}
COLOR = {"palette": "#2E7D32", "synthseg_em": "#C62828",
         "synthseg_noem": "#EF6C00", "auglab_default": "#1565C0"}


def _bar(ax, methods, vals, los, his, colors, title, ylim=None, refline=None):
    xs = np.arange(len(methods))
    ax.bar(xs, vals, color=colors, alpha=0.85, yerr=[los, his], capsize=4,
           ecolor="#333", error_kw={"lw": 1.2})
    for x, v in zip(xs, vals):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([LABEL.get(m, m) for m in methods], rotation=25, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    if refline is not None:
        ax.axhline(refline, ls="--", c="k", lw=1, label=f"real ref ({refline:.1f})"); ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    d = ap.parse_args().output_dir

    df = pd.read_csv(d / "coverage_metrics.csv")
    summ = json.loads((d / "coverage_summary.json").read_text())
    cfg = summ["config"]
    methods = [m for m in summ["methods"] if m in set(df["method"])]
    dfi = df.set_index("method")

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6])

    ax0 = fig.add_subplot(gs[0])
    _bar(ax0, methods,
         [dfi.loc[m, "macro_coverage"] for m in methods],
         [max(0, dfi.loc[m, "macro_coverage"] - dfi.loc[m, "macro_coverage_ci_lo"]) for m in methods],
         [max(0, dfi.loc[m, "macro_coverage_ci_hi"] - dfi.loc[m, "macro_coverage"]) for m in methods],
         [COLOR[m] for m in methods],
         "Coverage (Naeem 2020)\nper scanner×contrast → macro-avg\n(does synth reach each real regime)")

    ax1 = fig.add_subplot(gs[1])
    _bar(ax1, methods,
         [dfi.loc[m, "vendi"] for m in methods],
         [max(0, dfi.loc[m, "vendi"] - dfi.loc[m, "vendi_ci_lo"]) for m in methods],
         [max(0, dfi.loc[m, "vendi_ci_hi"] - dfi.loc[m, "vendi"]) for m in methods],
         [COLOR[m] for m in methods],
         "Vendi (Friedman & Dieng 2023)\nglobal diversity of synth cloud",
         refline=cfg.get("vendi_real_reference"))

    # per-group heatmap
    ax2 = fig.add_subplot(gs[2])
    pg = pd.read_csv(d / "coverage_per_group.csv")
    if len(pg):
        mat = pg.pivot(index="group", columns="method", values="coverage").reindex(columns=methods)
        mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]
        im = ax2.imshow(mat.values, cmap="viridis", aspect="auto", vmin=0,
                        vmax=float(np.nanmax(mat.values)) if np.isfinite(np.nanmax(mat.values)) else 1)
        ax2.set_xticks(range(len(methods))); ax2.set_xticklabels([LABEL.get(m, m) for m in methods], rotation=25, ha="right", fontsize=8)
        ax2.set_yticks(range(len(mat.index))); ax2.set_yticklabels(mat.index, fontsize=7)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                             color="white" if v < 0.5 * np.nanmax(mat.values) else "black")
        plt.colorbar(im, ax=ax2, shrink=0.8, label="Coverage")
    ax2.set_title("Per (scanner×contrast) Coverage", fontsize=10)

    fig.suptitle(
        f"Histogram-manifold coverage — 31-class regional histograms | "
        f"PCA {int(cfg['pca_variance']*100)}% ({cfg['pca_dims']}d), k={cfg['prdc_k']}, "
        f"{cfg['n_viable_groups']} groups, {cfg['n_boot']} bootstraps",
        fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(d / f"coverage_metrics.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pair = pd.read_csv(d / "coverage_paired.csv")
    (d / "coverage_paired.md").write_text(
        f"# Paired {summ['methods'][0]} vs synthseg_em (bootstrap)\n\n" + pair.to_markdown(index=False) + "\n")
    print(df.to_string(index=False)); print(pair.to_string(index=False))
    print(f"Saved → {d/'coverage_metrics.png'}")


if __name__ == "__main__":
    main()
