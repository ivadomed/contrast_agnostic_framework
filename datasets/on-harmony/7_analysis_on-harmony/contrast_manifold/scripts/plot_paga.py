#!/usr/bin/env python
"""
PAGA connectivity between real contrast/scanner clusters and synthetic data.

Replaces the homegrown UMAP "connectivity curve" with PAGA (Partition-based
Graph Abstraction; Wolf et al., Genome Biology 2019; scanpy.tl.paga). PAGA
scores connectivity on the high-dimensional kNN graph — NOT the 2-D layout —
and normalises each group-pair connectivity against the number of inter-group
edges expected under a random graph (the null model our old curve lacked).

Two questions, two PAGA runs in the SAME PCA space:
  1. real-only graph   → baseline real_i ↔ real_j connectivity.
  2. real + synthetic  → real_i ↔ real_j connectivity AFTER synthetic nodes can
                         act as bridges, plus each real group's connectivity to
                         the synthetic blob.
The delta (with_synth − real_only) on the real×real block is the bridging
effect; real_i ↔ synthetic is how much synthetic merges into each real cluster.

NOTE: PAGA measures continuity/merging, not envelopment. A domain-randomised
generator that surrounds a real cluster from far away (high PRDC coverage) can
still show modest PAGA connectivity. Read this alongside PRDC, not instead of it.

Outputs (in --output_dir):
  paga_connectivity_to_synthetic.csv   per real group: size + conn. to synthetic
  paga_connectivity_to_synthetic.png   bar chart, coloured by modality
  paga_real_real_real_only.csv         real×real connectivity (no synthetic)
  paga_real_real_with_synth.csv        real×real connectivity (synthetic in graph)
  paga_real_real_delta.csv             per pair: before, after, delta (bridging)
  paga_heatmap_real_only.png           real×real heatmap (baseline)
  paga_heatmap_with_synth.png          real×real heatmap (with synthetic)
  paga_heatmap_delta.png               bridging-delta heatmap
  paga_graph.png                       PAGA node-link graph (supplementary)

Usage:
  run_job --gpus 0 --slot 0 --wait -- .venv/bin/python \\
    datasets/on-harmony/7_analysis_on-harmony/contrast_manifold/scripts/plot_paga.py \\
    --original_csv  .../on_harmony_features_normalized_combined_downsampled100_feat_selected.csv \\
    --synthetic_csv .../synthetic_v26_6_guidance_lhc_features_normalized_combined_feat_selected.csv \\
    --output_dir    .../regional_hist_64/umap/PAGA
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anndata as ad
import scanpy as sc

# Reuse the joint script's helpers (same directory → importable when run as a script).
from plot_umap_joint import (
    META_COLS,
    MODALITY_PALETTE,
    _save_mpl,
    apply_feature_filter,
    base_mod,
    build_matrix,
    feature_cols,
    load_csv,
    load_feature_config,
    run_pca,
    write_feature_log,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYNTH_GROUP = "synthetic"


def real_group_labels(meta: pd.DataFrame) -> np.ndarray:
    """One label per real point: '<base_modality> × <scanner>'."""
    mod = meta["modality"].apply(base_mod)
    return (mod + " × " + meta["scanner"].astype(str)).values


def paga_connectivity(
    X: np.ndarray, groups: np.ndarray, n_neighbors: int
) -> pd.DataFrame:
    """Run PAGA on PCA reps X with the given group labels.

    Returns a symmetric group×group connectivity DataFrame (0 = disconnected,
    ~1 = continuous/merged), indexed by group name.
    """
    adata = ad.AnnData(X=np.ascontiguousarray(X, dtype=np.float32))
    adata.obs["group"] = pd.Categorical(groups)
    # use_rep='X' → build the kNN graph directly on the PCA reps we pass in.
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.paga(adata, groups="group")
    cats = list(adata.obs["group"].cat.categories)
    C = adata.uns["paga"]["connectivities"].toarray()
    return pd.DataFrame(C, index=cats, columns=cats), adata


def _heatmap(df: pd.DataFrame, title: str, output_dir: Path, stem: str,
             cmap: str = "viridis", vmin=None, vmax=None, center_zero: bool = False) -> None:
    n = len(df)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.32 + 2), max(5, n * 0.32 + 2)))
    if center_zero:
        m = np.nanmax(np.abs(df.values)) or 1.0
        vmin, vmax, cmap = -m, m, "RdBu_r"
    im = ax.imshow(df.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(df.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(df.index, fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.7, label="PAGA connectivity")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    _save_mpl(fig, output_dir, stem)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--original_csv", type=Path, required=True)
    p.add_argument("--synthetic_csv", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_group_size", type=int, default=4,
                   help="Drop real groups with fewer points than this from the graph (default 4).")
    p.add_argument("--feature_config", type=Path, default=None,
                   help="YAML config with exclude_families/exclude_features.")
    return p.parse_args()


def main():
    args = parse_args()
    for pth in (args.original_csv, args.synthetic_csv):
        if not pth.exists():
            raise FileNotFoundError(pth)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_orig  = load_csv(args.original_csv)
    df_synth = load_csv(args.synthetic_csv)

    if args.feature_config:
        cfg = load_feature_config(args.feature_config)
        all_feat = sorted(set(feature_cols(df_orig)) | set(feature_cols(df_synth)))
        kept, excluded = apply_feature_filter(all_feat, cfg)
        write_feature_log(kept, excluded, args.output_dir, args.feature_config)
        df_orig  = df_orig.drop(columns=[c for c in excluded if c in df_orig.columns])
        df_synth = df_synth.drop(columns=[c for c in excluded if c in df_synth.columns])

    # Aligned, imputed feature matrices, then a single joint PCA space.
    X_orig, X_synth, _feat = build_matrix(df_orig, df_synth)
    meta_orig = df_orig[[c for c in META_COLS if c in df_orig.columns]].reset_index(drop=True)
    X_orig_pca, X_synth_pca, _pca = run_pca(X_orig, X_synth)

    groups_real = real_group_labels(meta_orig)

    # Drop tiny real groups (noisy PAGA connectivity) — applied to BOTH runs.
    sizes = pd.Series(groups_real).value_counts()
    keep_groups = set(sizes[sizes >= args.min_group_size].index)
    keep_mask = np.array([g in keep_groups for g in groups_real])
    n_dropped = (~keep_mask).sum()
    if n_dropped:
        dropped = sorted(set(groups_real) - keep_groups)
        log.info("Dropping %d real points in %d groups with <%d points: %s",
                 n_dropped, len(dropped), args.min_group_size, dropped)
    X_orig_pca = X_orig_pca[keep_mask]
    groups_real = groups_real[keep_mask]
    log.info("Real points: %d in %d groups | synthetic: %d",
             len(groups_real), len(set(groups_real)), len(X_synth_pca))

    # ── 1. real-only PAGA ──────────────────────────────────────────────────
    conn_real, _ = paga_connectivity(X_orig_pca, groups_real, args.n_neighbors)
    conn_real.to_csv(args.output_dir / "paga_real_real_real_only.csv")

    # ── 2. real + synthetic PAGA (same PCA space) ──────────────────────────
    X_all = np.vstack([X_orig_pca, X_synth_pca])
    groups_all = np.concatenate([groups_real, np.full(len(X_synth_pca), SYNTH_GROUP)])
    conn_all, adata_all = paga_connectivity(X_all, groups_all, args.n_neighbors)

    real_cats = list(conn_real.index)
    conn_all_rr = conn_all.loc[real_cats, real_cats]
    conn_all_rr.to_csv(args.output_dir / "paga_real_real_with_synth.csv")

    # Bridging delta on the real×real block
    delta = conn_all_rr - conn_real
    delta.to_csv(args.output_dir / "paga_real_real_delta.csv")

    # Long-form bridging table (upper triangle only)
    rows = []
    for i, ga in enumerate(real_cats):
        for gb in real_cats[i + 1:]:
            rows.append({
                "group_a": ga, "group_b": gb,
                "conn_real_only": round(float(conn_real.loc[ga, gb]), 6),
                "conn_with_synth": round(float(conn_all_rr.loc[ga, gb]), 6),
                "bridging_delta": round(float(delta.loc[ga, gb]), 6),
            })
    pd.DataFrame(rows).sort_values("bridging_delta", ascending=False).to_csv(
        args.output_dir / "paga_bridging_pairs.csv", index=False)

    # ── Per real group: connectivity to the synthetic blob ─────────────────
    to_synth = conn_all.loc[real_cats, SYNTH_GROUP]
    sizes_kept = pd.Series(groups_real).value_counts()
    summary = pd.DataFrame({
        "group": real_cats,
        "modality": [g.split(" × ")[0] for g in real_cats],
        "n_real": [int(sizes_kept[g]) for g in real_cats],
        "conn_to_synthetic": [round(float(to_synth[g]), 6) for g in real_cats],
    }).sort_values("conn_to_synthetic", ascending=False)
    summary.to_csv(args.output_dir / "paga_connectivity_to_synthetic.csv", index=False)
    log.info("Mean real↔synthetic connectivity: %.3f", summary["conn_to_synthetic"].mean())

    # ── Plots ──────────────────────────────────────────────────────────────
    _heatmap(conn_real,    "PAGA real↔real (real-only graph)",
             args.output_dir, "paga_heatmap_real_only")
    _heatmap(conn_all_rr,  "PAGA real↔real (real + synthetic graph)",
             args.output_dir, "paga_heatmap_with_synth")
    _heatmap(delta,        "PAGA bridging delta (with_synth − real_only)",
             args.output_dir, "paga_heatmap_delta", center_zero=True)

    # Bar chart: connectivity to synthetic, coloured by modality
    fig, ax = plt.subplots(figsize=(max(7, len(summary) * 0.28 + 2), 5))
    colors = [MODALITY_PALETTE.get(m, "#888") for m in summary["modality"]]
    ax.bar(range(len(summary)), summary["conn_to_synthetic"].values, color=colors)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary["group"], rotation=90, fontsize=6)
    ax.set_ylabel("PAGA connectivity to synthetic")
    ax.set_title("How strongly synthetic data merges into each real group")
    ax.axhline(summary["conn_to_synthetic"].mean(), color="black", ls="--", lw=0.8,
               label=f"mean = {summary['conn_to_synthetic'].mean():.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_mpl(fig, args.output_dir, "paga_connectivity_to_synthetic")

    # Supplementary PAGA node-link graph
    try:
        ax = sc.pl.paga(adata_all, color="group", threshold=0.05, fontsize=5,
                        node_size_scale=0.5, show=False, frameon=False)
        fig = ax.get_figure() if hasattr(ax, "get_figure") else plt.gcf()
        fig.savefig(args.output_dir / "paga_graph.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved → %s", args.output_dir / "paga_graph.png")
    except Exception as e:
        log.warning("Could not render PAGA node-link graph: %s", e)

    log.info("All PAGA outputs saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
