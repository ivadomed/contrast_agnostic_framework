#!/usr/bin/env python
"""
Quantify how much synthetic data BRIDGES otherwise-separated real clusters —
the gap-filling you see on the UMAP plot, measured rigorously in feature space.

Metric: single-linkage / minimax BOTTLENECK gap reduction.
--------------------------------------------------------------------------------
To travel from real cluster A to real cluster B, the hardest step is the largest
jump across empty space you are forced to make — the *bottleneck* (minimax)
distance. Synthetic points act as steppingstones that shrink that jump.

For each pair of real clusters (A, B):
  bottleneck_real = min over paths using REAL points only of (max edge on path)
  bottleneck_all  = same, but paths may route THROUGH synthetic points
  reduction       = bottleneck_real - bottleneck_all      (>= 0 always)
  bridge_ratio    = reduction / bottleneck_real           (0=no help, 1=fully filled)

Why this is trustworthy:
  * Parameter-free: no k, no chosen threshold. Both bottlenecks are read off the
    minimum spanning tree, which exactly preserves minimax distances (this is
    classical single-linkage / 0-dim persistent homology).
  * Causal guarantee: adding synthetic can only lower the bottleneck, and any
    reduction is ONLY achievable via a synth-using route (a better real-only
    route would already be counted in bottleneck_real). So bridge_ratio > 0
    provably means synthetic is the bridge.
  * Fair across generators: PCA is fit on REAL ONLY and synthetic is projected
    in, so bottleneck_real is identical across generators; only the synthetic
    term moves.
  * Anchored: a uniform space-filling null (same #points, random in the real
    bounding box) gives the bridging you'd get from structureless filler. Real
    synthetic should beat it.

Outputs (in --output_dir):
  bridging_pairs.csv      per real-cluster pair: bottlenecks, reduction, ratio
  bridging_by_contrast.csv mean bridge_ratio per contrast×contrast block
  bridging_summary.csv    headline numbers (+ null control if --null)
  bridging_heatmap.png    real×real bridge_ratio heatmap

Usage:
  set_slot 0 .venv/bin/python .../scripts/quantify_bridging.py \\
    --original_csv .../on_harmony_..._feat_selected.csv \\
    --synthetic_csv .../synthetic_v26_6_..._feat_selected.csv \\
    --output_dir   .../umap/bridging --null
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
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from plot_umap_joint import (
    META_COLS,
    _save_mpl,
    apply_feature_filter,
    base_mod,
    build_matrix,
    feature_cols,
    load_csv,
    load_feature_config,
    run_pca_orig_only,
    write_feature_log,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def real_group_labels(meta: pd.DataFrame) -> np.ndarray:
    mod = meta["modality"].apply(base_mod)
    return (mod + " × " + meta["scanner"].astype(str)).values


def mst_edges(X: np.ndarray) -> list[tuple[float, int, int]]:
    """Edges (weight, u, v) of the Euclidean MST of X, ascending by weight."""
    D = squareform(pdist(X))
    mst = minimum_spanning_tree(D).tocoo()
    edges = sorted(zip(mst.data.tolist(), mst.row.tolist(), mst.col.tolist()))
    return edges


def set_bottlenecks(
    n_nodes: int,
    edges: list[tuple[float, int, int]],
    node_cluster: np.ndarray,
    cluster_ids: list[int],
) -> dict[tuple[int, int], float]:
    """
    Kruskal on the (MST) edges ascending. The first edge that connects a
    component containing cluster A to one containing cluster B is the minimax
    (bottleneck) distance between sets A and B. Nodes with cluster id -1
    (synthetic / null) carry no label but propagate connectivity as bridges.
    """
    parent = list(range(n_nodes))
    rank = [0] * n_nodes
    comp = [set() for _ in range(n_nodes)]
    for i, c in enumerate(node_cluster):
        if c >= 0:
            comp[i].add(int(c))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    result: dict[tuple[int, int], float] = {}
    n_pairs_total = len(cluster_ids) * (len(cluster_ids) - 1) // 2
    for w, u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        cu, cv = comp[ru], comp[rv]
        if cu and cv:
            for ca in cu:
                for cb in cv:
                    if ca == cb:
                        continue
                    key = (ca, cb) if ca < cb else (cb, ca)
                    if key not in result:
                        result[key] = float(w)
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        comp[ru] |= comp[rv]
        if rank[ru] == rank[rv]:
            rank[ru] += 1
        if len(result) == n_pairs_total:
            break
    return result


def bottleneck_matrix(
    X_points: np.ndarray,
    node_cluster: np.ndarray,
    cluster_ids: list[int],
) -> dict[tuple[int, int], float]:
    edges = mst_edges(X_points)
    return set_bottlenecks(len(X_points), edges, node_cluster, cluster_ids)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--original_csv", type=Path, required=True)
    p.add_argument("--synthetic_csv", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--min_group_size", type=int, default=4)
    p.add_argument("--null", action="store_true",
                   help="Also compute a uniform space-filling null (same #points).")
    p.add_argument("--feature_config", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    for pth in (args.original_csv, args.synthetic_csv):
        if not pth.exists():
            raise FileNotFoundError(pth)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_orig = load_csv(args.original_csv)
    df_synth = load_csv(args.synthetic_csv)

    if args.feature_config:
        cfg = load_feature_config(args.feature_config)
        all_feat = sorted(set(feature_cols(df_orig)) | set(feature_cols(df_synth)))
        kept, excluded = apply_feature_filter(all_feat, cfg)
        write_feature_log(kept, excluded, args.output_dir, args.feature_config)
        df_orig = df_orig.drop(columns=[c for c in excluded if c in df_orig.columns])
        df_synth = df_synth.drop(columns=[c for c in excluded if c in df_synth.columns])

    X_orig, X_synth, _feat = build_matrix(df_orig, df_synth)
    meta_orig = df_orig[[c for c in META_COLS if c in df_orig.columns]].reset_index(drop=True)

    # PCA fit on REAL only → synthetic projected in (fair, generator-independent real geometry)
    X_orig_pca, X_synth_pca, _pca = run_pca_orig_only(X_orig, X_synth)

    groups = real_group_labels(meta_orig)
    sizes = pd.Series(groups).value_counts()
    keep = set(sizes[sizes >= args.min_group_size].index)
    mask = np.array([g in keep for g in groups])
    if (~mask).sum():
        log.info("Dropping %d points in groups <%d: %s",
                 (~mask).sum(), args.min_group_size, sorted(set(groups) - keep))
    X_orig_pca = X_orig_pca[mask]
    groups = groups[mask]

    cat = sorted(set(groups))
    cat_idx = {c: i for i, c in enumerate(cat)}
    real_cluster = np.array([cat_idx[g] for g in groups])
    cluster_ids = list(range(len(cat)))
    log.info("Real points: %d in %d clusters | synthetic: %d",
             len(groups), len(cat), len(X_synth_pca))

    # ── bottleneck_real: real points only ──────────────────────────────────
    bn_real = bottleneck_matrix(X_orig_pca, real_cluster, cluster_ids)

    # ── bottleneck_all: real + synthetic (synth = cluster -1) ───────────────
    X_all = np.vstack([X_orig_pca, X_synth_pca])
    clust_all = np.concatenate([real_cluster, np.full(len(X_synth_pca), -1)])
    bn_all = bottleneck_matrix(X_all, clust_all, cluster_ids)

    # ── optional uniform space-filling null ─────────────────────────────────
    bn_null = None
    if args.null:
        lo, hi = X_orig_pca.min(0), X_orig_pca.max(0)
        rng = np.random.default_rng(42)
        X_null = rng.uniform(lo, hi, size=X_synth_pca.shape)
        X_all_null = np.vstack([X_orig_pca, X_null])
        clust_null = np.concatenate([real_cluster, np.full(len(X_null), -1)])
        bn_null = bottleneck_matrix(X_all_null, clust_null, cluster_ids)

    # ── per-pair table ──────────────────────────────────────────────────────
    rows = []
    for i in range(len(cat)):
        for j in range(i + 1, len(cat)):
            key = (i, j)
            br, ba = bn_real.get(key), bn_all.get(key)
            if br is None or ba is None:
                continue
            red = br - ba
            ratio = red / br if br > 0 else 0.0
            rec = {
                "group_a": cat[i], "group_b": cat[j],
                "contrast_a": cat[i].split(" × ")[0],
                "contrast_b": cat[j].split(" × ")[0],
                "bottleneck_real": round(br, 6),
                "bottleneck_all": round(ba, 6),
                "reduction": round(red, 6),
                "bridge_ratio": round(ratio, 6),
            }
            if bn_null is not None:
                bn = bn_null.get(key)
                rec["bridge_ratio_null"] = round((br - bn) / br, 6) if (bn is not None and br > 0) else np.nan
            rows.append(rec)
    pairs = pd.DataFrame(rows).sort_values("bridge_ratio", ascending=False)
    pairs.to_csv(args.output_dir / "bridging_pairs.csv", index=False)

    # ── per contrast×contrast aggregate ─────────────────────────────────────
    pairs["cpair"] = pairs.apply(
        lambda r: " ↔ ".join(sorted([r["contrast_a"], r["contrast_b"]])), axis=1)
    by_c = pairs.groupby("cpair").agg(
        n_pairs=("bridge_ratio", "size"),
        mean_bridge_ratio=("bridge_ratio", "mean"),
    ).round(4).sort_values("mean_bridge_ratio", ascending=False)
    by_c.to_csv(args.output_dir / "bridging_by_contrast.csv")

    # ── headline summary ────────────────────────────────────────────────────
    summary = {
        "n_cluster_pairs": len(pairs),
        "mean_bridge_ratio": round(float(pairs["bridge_ratio"].mean()), 6),
        "median_bridge_ratio": round(float(pairs["bridge_ratio"].median()), 6),
        "frac_pairs_ratio_gt_0.5": round(float((pairs["bridge_ratio"] > 0.5).mean()), 6),
        "mean_bottleneck_real": round(float(pairs["bottleneck_real"].mean()), 6),
        "mean_bottleneck_all": round(float(pairs["bottleneck_all"].mean()), 6),
    }
    if bn_null is not None:
        summary["mean_bridge_ratio_null"] = round(float(pairs["bridge_ratio_null"].mean()), 6)
        summary["bridge_ratio_minus_null"] = round(
            summary["mean_bridge_ratio"] - summary["mean_bridge_ratio_null"], 6)
    pd.DataFrame([summary]).to_csv(args.output_dir / "bridging_summary.csv", index=False)
    log.info("SUMMARY: %s", summary)

    # ── heatmap ─────────────────────────────────────────────────────────────
    M = np.full((len(cat), len(cat)), np.nan)
    for _, r in pairs.iterrows():
        i, j = cat_idx[r["group_a"]], cat_idx[r["group_b"]]
        M[i, j] = M[j, i] = r["bridge_ratio"]
    fig, ax = plt.subplots(figsize=(max(6, len(cat) * 0.32 + 2),) * 2)
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cat))); ax.set_xticklabels(cat, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cat))); ax.set_yticklabels(cat, fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.7, label="bridge_ratio (gap fraction filled by synthetic)")
    ax.set_title("Synthetic bridging of real clusters\n(1 = synthetic fully fills the gap)", fontsize=10)
    fig.tight_layout()
    _save_mpl(fig, args.output_dir, "bridging_heatmap")

    log.info("All bridging outputs saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
