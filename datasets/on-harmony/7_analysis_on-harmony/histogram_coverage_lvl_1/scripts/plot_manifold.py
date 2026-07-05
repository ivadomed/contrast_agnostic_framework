#!/usr/bin/env python
"""
Qualitative manifold plots (Pillar 2): PCA and UMAP of the 31-class regional-histogram
feature space — real ON-Harmony vs each synthesis method.

  PCA  (linear, variance-preserving) — fit on balanced real, project all.
  UMAP (McInnes et al. 2018)         — one joint embedding fit on real+synth, subsets shown.

Layout: panel 1 = real coloured by modality (reference); panels 2..5 = real (grey) with one
method overlaid, so you can see where each method's cloud sits relative to the real manifold.

Usage:
  run_job --gpus 0 --cpus 8 --mem 32G --wait -- .venv/bin/python plot_manifold.py \\
    --real-csv real_regional_hist31.csv --synth-csv synth_regional_hist31.csv --output-dir outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import umap

SEED = 42
METHOD_ORDER = ["palette", "synthseg_em", "synthseg_noem", "auglab_default"]
MLABEL = {"palette": "PALETTE", "synthseg_em": "SynthSeg-EM",
          "synthseg_noem": "SynthSeg-noEM", "auglab_default": "auglab-default"}
MCOLOR = {"palette": "#2E7D32", "synthseg_em": "#C62828",
          "synthseg_noem": "#EF6C00", "auglab_default": "#1565C0"}
MOD_COLORS = {"T1w": "#E53935", "T2w": "#1E88E5", "FLAIR": "#43A047", "dwi": "#FB8C00",
              "bold": "#8E24AA", "epi": "#00ACC1", "GRE": "#6D4C41"}
REAL_META = {"subject", "session", "modality_id", "acq_tag", "scanner_model",
             "scanner_vendor", "cohort_category", "image_path", "mask_path", "label_map_path"}
SYNTH_META = {"method", "subject", "session", "key", "run"}
PLOT_SYNTH_N = 500   # subsample per method for legibility


def fcols(df, meta):
    return [c for c in df.columns if c not in meta and not c.startswith("diagnostics_")]


def balance_real(df, cap=100, seed=SEED):
    df = df.copy()
    df["_mod"] = df["modality_id"].astype(str).str.split("_").str[0]
    df["_sc"] = df.get("scanner_model", "unknown").astype(str)
    return pd.concat([g if len(g) <= cap else g.sample(cap, random_state=seed)
                      for _, g in df.groupby(["_mod", "_sc"])], ignore_index=True)


def _panels(ax_list, real_xy, real_mod, method_xy, title_prefix):
    # panel 0: real by modality
    ax = ax_list[0]
    for mod in sorted(set(real_mod)):
        mm = real_mod == mod
        ax.scatter(real_xy[mm, 0], real_xy[mm, 1], s=6, alpha=0.5,
                   c=MOD_COLORS.get(mod, "#888"), label=mod, linewidths=0)
    ax.set_title(f"{title_prefix}: real by modality", fontsize=9)
    ax.legend(fontsize=6, markerscale=2, loc="best", framealpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    # panels 1..: each method over real-grey
    for ax, m in zip(ax_list[1:], METHOD_ORDER):
        if m not in method_xy:
            ax.axis("off"); continue
        ax.scatter(real_xy[:, 0], real_xy[:, 1], s=5, alpha=0.25, c="#BBBBBB", linewidths=0)
        xy = method_xy[m]
        ax.scatter(xy[:, 0], xy[:, 1], s=6, alpha=0.5, c=MCOLOR[m], linewidths=0)
        ax.set_title(f"{title_prefix}: {MLABEL[m]} vs real", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-csv", type=Path, required=True)
    ap.add_argument("--synth-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--real-cap-per-group", type=int, default=100)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    real = balance_real(pd.read_csv(a.real_csv), a.real_cap_per_group)
    synth = pd.read_csv(a.synth_csv)
    common = sorted(set(fcols(real, REAL_META)) & set(fcols(synth, SYNTH_META)))
    methods = [m for m in METHOD_ORDER if m in set(synth["method"])]
    real_mod = real["modality_id"].astype(str).str.split("_").str[0].to_numpy()

    imp = SimpleImputer(strategy="mean").fit(real[common].to_numpy(float))
    scal = StandardScaler().fit(imp.transform(real[common].to_numpy(float)))

    def prep(df):
        return scal.transform(imp.transform(df[common].to_numpy(float)))

    Rs = prep(real)
    # subsample synth per method for plotting
    synth_sub = {}
    for m in methods:
        sm = synth[synth["method"] == m]
        if len(sm) > PLOT_SYNTH_N:
            sm = sm.iloc[rng.choice(len(sm), PLOT_SYNTH_N, replace=False)]
        synth_sub[m] = prep(sm)

    # ── PCA (fit on real) ──
    pca = PCA(n_components=2, random_state=SEED).fit(Rs)
    real_pca = pca.transform(Rs)
    meth_pca = {m: pca.transform(v) for m, v in synth_sub.items()}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10)); axes = axes.ravel()
    _panels(axes, real_pca, real_mod, meth_pca,
            f"PCA (PC1 {pca.explained_variance_ratio_[0]*100:.0f}%, PC2 {pca.explained_variance_ratio_[1]*100:.0f}%)")
    if len(methods) < 5:
        axes[-1].axis("off")
    fig.suptitle("31-class regional-histogram manifold — PCA (fit on balanced real)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(a.output_dir / f"manifold_pca.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ── UMAP (joint fit on real + synth subsets) ──
    blocks = [Rs] + [synth_sub[m] for m in methods]
    sizes = [len(Rs)] + [len(synth_sub[m]) for m in methods]
    emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED).fit_transform(np.vstack(blocks))
    off = np.cumsum([0] + sizes)
    real_um = emb[off[0]:off[1]]
    meth_um = {m: emb[off[i + 1]:off[i + 2]] for i, m in enumerate(methods)}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10)); axes = axes.ravel()
    _panels(axes, real_um, real_mod, meth_um, "UMAP")
    if len(methods) < 5:
        axes[-1].axis("off")
    fig.suptitle("31-class regional-histogram manifold — UMAP (McInnes 2018, joint fit)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(a.output_dir / f"manifold_umap.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved manifold_pca.png + manifold_umap.png → {a.output_dir}")


if __name__ == "__main__":
    main()
