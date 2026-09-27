#!/usr/bin/env python
"""
Plots for the cross-contrast-NGF vs. real-fill-ladder pilot (see compute_cross_contrast_ngf.py
for the full writeup of what's being tested and why).

Fig 1 (ngf_vs_ladder_heatmaps.png): two diverging heatmaps, rows = (train->eval) direction,
columns = tumor region. Left panel is the REAL effect we're trying to explain (mean rung4->5
Dice delta, pooled across patients). Right panel is the candidate explanation (per-patient
Spearman rho between cross-contrast NGF and that delta). Putting them side by side makes the
central negative finding visible at a glance: the row with the largest |mean delta| (t2w->t1n)
does not line up with the row/column that has the strongest rho.

Fig 2 (ngf_vs_ladder_scatter_highlights.png): 4 per-patient scatter panels — the 3 nominally
significant cells (t2w->t1n/NCR, t2f->t1c/SNFH, t1n->t2w/ET) plus the "flagship" t2w->t1n/SNFH
cell (not significant, but the one that actually carries the population-level degradation) —
so the reader can see the actual point clouds behind the correlation numbers, not just p-values.

Uses RdBu / RdBu_r diverging colormaps, matching this project's existing heatmap convention
(00_commun_scripts/00_03_evaluate/aggregate_from_config.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "outputs" / "data"
PLOTS_DIR = THIS_DIR.parent / "outputs" / "plots"

TRAIN_EVAL_ORDER = [
    ("t1n", "t1c"), ("t1n", "t2w"), ("t1n", "t2f"),
    ("t2w", "t1n"), ("t2w", "t1c"), ("t2w", "t2f"),
    ("t2f", "t1n"), ("t2f", "t1c"), ("t2f", "t2w"),
]
REGION_ORDER = ["NCR", "SNFH", "ET"]
REGION_LABEL = {"NCR": "necrotic core\n(NCR)", "SNFH": "edema\n(SNFH)", "ET": "enhancing\n(ET)"}


def _text_color_for(rgba):
    r, g, b, _ = rgba
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.55 else "black"


def plot_heatmaps(cell_df: pd.DataFrame, out_path: Path):
    n_rows = len(TRAIN_EVAL_ORDER)
    delta_mat = np.full((n_rows, len(REGION_ORDER)), np.nan)
    rho_mat = np.full((n_rows, len(REGION_ORDER)), np.nan)
    p_mat = np.full((n_rows, len(REGION_ORDER)), np.nan)
    n_mat = np.full((n_rows, len(REGION_ORDER)), np.nan)

    for i, (tr, ev) in enumerate(TRAIN_EVAL_ORDER):
        for j, region in enumerate(REGION_ORDER):
            row = cell_df[(cell_df["train"] == tr) & (cell_df["eval"] == ev) & (cell_df["region"] == region)]
            if row.empty:
                continue
            r = row.iloc[0]
            delta_mat[i, j] = r["mean_delta_dice"]
            rho_mat[i, j] = r["spearman_rho"]
            p_mat[i, j] = r["p_value"]
            n_mat[i, j] = r["n"]

    row_labels = [f"{tr}→{ev}" for tr, ev in TRAIN_EVAL_ORDER]
    col_labels = [REGION_LABEL[r] for r in REGION_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(9, 7))

    def draw_panel(ax, mat, title, cmap, vlim, fmt, star_mask=None):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(title, fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#999")
                    continue
                txt = fmt(v)
                if star_mask is not None and star_mask[i, j]:
                    txt += "*"
                color = _text_color_for(im.cmap(im.norm(v)))
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08, orientation="horizontal")
        cbar.ax.tick_params(labelsize=7)
        return im

    draw_panel(axes[0], delta_mat, "Real effect: mean Δ Dice\n(rung4→5, real-fill − noise-fill)",
               "RdBu", 0.15, lambda v: f"{v:+.3f}")
    star = p_mat < 0.05
    draw_panel(axes[1], rho_mat, "Candidate explanation: Spearman ρ\n(NGF vs. Δ Dice, across patients)",
               "RdBu", 0.5, lambda v: f"{v:+.2f}", star_mask=star)

    fig.suptitle("BraTS2024-glioma: does cross-contrast texture similarity (NGF) explain\n"
                 "where real-fill helps or hurts cross-contrast generalization?", fontsize=11, y=1.02)
    fig.text(0.5, -0.02, "* p < 0.05 (uncorrected, n=33-70 patients per cell, 27 cells tested)",
              ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_scatter_highlights(merged: pd.DataFrame, cell_df: pd.DataFrame, out_path: Path):
    highlights = [
        ("t2w", "t1n", "SNFH", "flagship: drives the pooled degradation, NOT significant"),
        ("t2w", "t1n", "NCR", "significant, hypothesis-consistent (+)"),
        ("t2f", "t1c", "SNFH", "significant, hypothesis-consistent (+)"),
        ("t1n", "t2w", "ET", "significant, WRONG direction (−)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, (tr, ev, region, note) in zip(axes.flat, highlights):
        sub = merged[(merged["train"] == tr) & (merged["eval"] == ev) & (merged["region"] == region)]
        cell = cell_df[(cell_df["train"] == tr) & (cell_df["eval"] == ev) & (cell_df["region"] == region)].iloc[0]
        ax.scatter(sub["ngf_all"], sub["delta_dice"], s=28, alpha=0.75,
                   color="#3b6fa0", edgecolors="white", linewidths=0.5)
        if len(sub) >= 3:
            coef = np.polyfit(sub["ngf_all"], sub["delta_dice"], 1)
            xs = np.linspace(sub["ngf_all"].min(), sub["ngf_all"].max(), 50)
            ax.plot(xs, np.polyval(coef, xs), color="#c0392b", linewidth=1.5)
        ax.axhline(0, color="#999", linewidth=1, linestyle="--")
        ax.set_xlabel("NGF (texture similarity, symmetric)", fontsize=9)
        ax.set_ylabel("Δ Dice (real-fill − noise-fill)", fontsize=9)
        ax.set_title(f"{tr}→{ev}, {region}\n{note}", fontsize=9)
        ax.text(0.03, 0.03, f"n={int(cell['n'])}  ρ={cell['spearman_rho']:+.3f}  p={cell['p_value']:.3f}",
                transform=ax.transAxes, fontsize=8, color="#333",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#ccc", alpha=0.85))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Per-patient view behind the 4 headline cells", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_scatter_all(merged: pd.DataFrame, cell_df: pd.DataFrame, out_path: Path):
    """All 27 (train,eval,region) cells, same grid layout/order as the heatmap, so this figure
    and ngf_vs_ladder_heatmaps.png can be read side by side."""
    n_rows, n_cols = len(TRAIN_EVAL_ORDER), len(REGION_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.0 * n_rows), sharex=False)

    for i, (tr, ev) in enumerate(TRAIN_EVAL_ORDER):
        for j, region in enumerate(REGION_ORDER):
            ax = axes[i, j]
            sub = merged[(merged["train"] == tr) & (merged["eval"] == ev) & (merged["region"] == region)]
            row = cell_df[(cell_df["train"] == tr) & (cell_df["eval"] == ev) & (cell_df["region"] == region)]
            if sub.empty or row.empty:
                ax.axis("off")
                continue
            r = row.iloc[0]
            sig = r["p_value"] < 0.05
            ax.scatter(sub["ngf_all"], sub["delta_dice"], s=16, alpha=0.7,
                       color="#3b6fa0", edgecolors="white", linewidths=0.3)
            if len(sub) >= 3:
                coef = np.polyfit(sub["ngf_all"], sub["delta_dice"], 1)
                xs = np.linspace(sub["ngf_all"].min(), sub["ngf_all"].max(), 50)
                ax.plot(xs, np.polyval(coef, xs), color="#c0392b" if sig else "#c0392b",
                        linewidth=2.0 if sig else 1.0, alpha=1.0 if sig else 0.6)
            ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
            title = f"n={int(r['n'])}  ρ={r['spearman_rho']:+.2f}  p={r['p_value']:.3f}"
            ax.set_title(title, fontsize=7.5, fontweight="bold" if sig else "normal",
                         color="#c0392b" if sig else "#333")
            ax.tick_params(labelsize=6.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if i == 0:
                ax.annotate(REGION_LABEL[region].replace("\n", " "), xy=(0.5, 1.35),
                            xycoords="axes fraction", ha="center", fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{tr}→{ev}\nΔ Dice", fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel("NGF", fontsize=8)

    fig.suptitle("All 27 cells: per-patient NGF (x) vs. real-fill Δ Dice (y)\n"
                 "bold red title/line = p < 0.05 (uncorrected)", fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=PLOTS_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cell_df = pd.read_csv(args.data_dir / "ngf_vs_ladder_cell_correlations.csv")
    merged = pd.read_csv(args.data_dir / "ngf_vs_ladder_per_patient.csv")

    plot_heatmaps(cell_df, args.out_dir / "ngf_vs_ladder_heatmaps.png")
    plot_scatter_highlights(merged, cell_df, args.out_dir / "ngf_vs_ladder_scatter_highlights.png")
    plot_scatter_all(merged, cell_df, args.out_dir / "ngf_vs_ladder_scatter_all.png")


if __name__ == "__main__":
    main()
