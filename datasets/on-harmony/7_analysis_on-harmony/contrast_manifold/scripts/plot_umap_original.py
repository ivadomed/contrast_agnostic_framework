#!/usr/bin/env python
"""
UMAP of the ORIGINAL (real) data only — no synthetic data.

Same feature → PCA → UMAP pipeline as plot_umap_joint.py, but fit on the
original normalized CSV alone. Useful to inspect the real multi-scanner /
multi-contrast manifold on its own.

Pipeline:
  1. Load the original CSV (already Z-score normalized via normalize_combined.py).
  2. (optional) feature selection via --feature_config.
  3. Drop all-NaN + zero-variance columns, impute remaining NaNs with col mean.
  4. PCA to 95% explained variance (min 3 components).
  5. Fit UMAP on the original PCA scores.
  6. Plot 2-D (pdf/png) and 3-D (interactive HTML), colour=modality, symbol=scanner.

Outputs (in --output_dir):
  umap_original_2d.{pdf,png}
  umap_original_3d.html        (interactive Plotly, click-to-copy path)
  umap_original_3d_2d.html     (2-D companion with lasso select/remove)
  umap_original_coords.csv

Usage:
  run_job --gpus 0 --slot 0 --wait -- .venv/bin/python \\
    datasets/on-harmony/7_analysis_on-harmony/contrast_manifold/scripts/plot_umap_original.py \\
    --original_csv datasets/on-harmony/7_analysis_on-harmony/contrast_manifold/outputs/data/original/regional_hist_64/on_harmony_features_normalized_combined_downsampled100_feat_selected.csv \\
    --output_dir   datasets/on-harmony/7_analysis_on-harmony/contrast_manifold/outputs/plots/v19/v19_c_r1/regional_hist_64/umap
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
try:
    from cuml.manifold import UMAP as _CumlUMAP
    _CUML_AVAILABLE = True
except ImportError:
    _CumlUMAP = None
    _CUML_AVAILABLE = False
import umap
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Reuse the joint script's helpers (same directory → importable when run as a script).
from plot_umap_joint import (
    META_COLS,
    MODALITY_PALETTE,
    MPL_MARKERS,
    PLOTLY_SYMS,
    _CLICK_TO_COPY_JS,
    _LASSO_REMOVE_JS,
    _hover_text,
    _save_mpl,
    apply_feature_filter,
    base_mod,
    feature_cols,
    load_csv,
    load_feature_config,
    write_feature_log,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_matrix_original(df_orig: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Drop all-NaN + zero-variance columns and impute, on original only."""
    cols = sorted(feature_cols(df_orig))
    if not cols:
        raise ValueError("No feature columns found in original CSV.")
    X = df_orig[cols].to_numpy(dtype=np.float64)

    # Drop all-NaN columns
    nan_mask = np.isnan(X).mean(axis=0) < 1.0
    X = X[:, nan_mask]
    cols = [c for c, k in zip(cols, nan_mask) if k]

    # Impute
    X = SimpleImputer(strategy="mean").fit_transform(X)

    # Drop zero-variance columns
    var_mask = X.var(axis=0) > 0
    X = X[:, var_mask]
    cols = [c for c, k in zip(cols, var_mask) if k]

    log.info("Original feature matrix: %d rows × %d cols", len(X), len(cols))
    return X, cols


def run_pca_original(X: np.ndarray) -> tuple[np.ndarray, PCA]:
    pca_full = PCA(random_state=42)
    pca_full.fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = max(3, min(int(np.searchsorted(cumvar, 0.95)) + 1, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X)
    log.info("PCA original: %d components → %.1f%% variance", n_comp, cumvar[n_comp - 1] * 100)
    return X_pca, pca


def run_umap_original(X: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    log.info("Fitting UMAP on %d original points … (cuML=%s)", len(X), _CUML_AVAILABLE)
    if _CUML_AVAILABLE:
        reducer = _CumlUMAP(
            n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
            metric="euclidean", random_state=42, verbose=False,
        )
    else:
        reducer = umap.UMAP(
            n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
            metric="euclidean", random_state=42, verbose=False, low_memory=False,
        )
    emb = reducer.fit_transform(X)
    log.info("UMAP fit complete.")
    return emb


def plot_2d(emb: np.ndarray, meta: pd.DataFrame, output_dir: Path) -> None:
    meta = meta.copy()
    meta["base_mod"] = meta["modality"].apply(base_mod)
    scanners   = sorted(meta["scanner"].unique())
    modalities = sorted(meta["base_mod"].unique())
    sc_marker  = {sc: MPL_MARKERS[i % len(MPL_MARKERS)] for i, sc in enumerate(scanners)}

    fig, ax = plt.subplots(figsize=(11, 9))
    for sc in scanners:
        for mod in modalities:
            mask = ((meta["scanner"] == sc) & (meta["base_mod"] == mod)).values
            if not mask.any():
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=MODALITY_PALETTE.get(mod, "#888"), marker=sc_marker[sc],
                       alpha=0.80, s=22, linewidths=0, zorder=2)

    mod_h = [mpatches.Patch(color=MODALITY_PALETTE.get(m, "#888"), label=m) for m in modalities]
    sc_h  = [plt.Line2D([0], [0], marker=sc_marker[sc], color="gray",
                        linestyle="none", markersize=6, label=sc) for sc in scanners]
    ax.legend(handles=mod_h + sc_h, loc="best", fontsize=7, framealpha=0.7)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("Contrast Manifold — UMAP (original only)")
    _save_mpl(fig, output_dir, "umap_original_2d")


def plot_3d(emb: np.ndarray, meta: pd.DataFrame, output_dir: Path) -> None:
    meta = meta.copy().reset_index(drop=True)
    meta["base_mod"] = meta["modality"].apply(base_mod)
    scanners   = sorted(meta["scanner"].unique())
    symbol_map = {sc: PLOTLY_SYMS[i % len(PLOTLY_SYMS)] for i, sc in enumerate(scanners)}
    hover, paths = _hover_text(meta)

    traces: list = []
    traces_2d: list = []
    for sc in scanners:
        for mod in sorted(meta["base_mod"].unique()):
            mask = ((meta["scanner"] == sc) & (meta["base_mod"] == mod)).values
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            traces.append(go.Scatter3d(
                x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2],
                mode="markers",
                marker=dict(size=3, color=MODALITY_PALETTE.get(mod, "#888"),
                            symbol=symbol_map[sc], opacity=0.85),
                name=f"{mod} / {sc}",
                legendgroup=mod,
                text=[hover[i] for i in idx],
                customdata=[[paths[i]] for i in idx],
                hoverinfo="text",
            ))
            traces_2d.append(go.Scatter(
                x=emb[mask, 0], y=emb[mask, 1],
                mode="markers",
                marker=dict(size=6, color=MODALITY_PALETTE.get(mod, "#888"),
                            symbol=symbol_map[sc], opacity=0.85,
                            line=dict(width=0.5, color="white")),
                name=f"{mod} / {sc}",
                legendgroup=mod,
                text=[hover[i] for i in idx],
                customdata=[[paths[i]] for i in idx],
                hoverinfo="text",
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=0.1)),
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Contrast Manifold — UMAP (original only, 3D)",
        scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"),
        legend=dict(font=dict(size=9)),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    html_path = output_dir / "umap_original_3d.html"
    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        post_script=_CLICK_TO_COPY_JS,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    log.info("Saved → %s", html_path)
    for ext in ("png", "pdf"):
        try:
            fig.write_image(str(output_dir / f"umap_original_3d.{ext}"),
                            width=1400, height=1000, scale=2)
            log.info("Saved → %s", output_dir / f"umap_original_3d.{ext}")
        except Exception as e:
            log.warning("Could not save %s: %s", ext, e)

    # 2-D companion with lasso select/remove (go.Scatter3d does not support lasso2d)
    fig_2d = go.Figure(data=traces_2d)
    fig_2d.update_layout(
        title="UMAP (original only, UMAP-1×UMAP-2) — lasso to select",
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
        legend=dict(font=dict(size=9)),
        margin=dict(l=60, r=20, b=60, t=60),
        dragmode="lasso",
    )
    html_path_2d = output_dir / "umap_original_3d_2d.html"
    fig_2d.write_html(
        str(html_path_2d),
        include_plotlyjs="cdn",
        post_script=[_CLICK_TO_COPY_JS, _LASSO_REMOVE_JS],
        config={
            "modeBarButtonsToAdd": ["lasso2d", "select2d"],
            "displayModeBar": True,
            "scrollZoom": True,
        },
    )
    log.info("Saved 2D lasso → %s", html_path_2d)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--original_csv", type=Path, required=True,
                   help="Original normalized CSV (e.g. *_normalized_combined_downsampled100_feat_selected.csv)")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument(
        "--feature_config", type=Path, default=None,
        help="YAML config specifying exclude_families and exclude_features. "
             "When set, outputs go into a subfolder named after the config stem.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.original_csv.exists():
        raise FileNotFoundError(f"Original normalized CSV not found: {args.original_csv}")

    if args.feature_config:
        args.output_dir = args.output_dir / args.feature_config.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_orig = load_csv(args.original_csv)

    if args.feature_config:
        cfg = load_feature_config(args.feature_config)
        all_feat = sorted(feature_cols(df_orig))
        kept, excluded_list = apply_feature_filter(all_feat, cfg)
        write_feature_log(kept, excluded_list, args.output_dir, args.feature_config)
        df_orig = df_orig.drop(columns=[c for c in excluded_list if c in df_orig.columns])
        log.info("Feature filter applied: %d kept, %d excluded", len(kept), len(excluded_list))

    X, _feat_cols = build_matrix_original(df_orig)
    meta = df_orig[[c for c in META_COLS if c in df_orig.columns]].reset_index(drop=True)

    X_pca, _pca = run_pca_original(X)
    emb = run_umap_original(X_pca, n_neighbors=args.n_neighbors, min_dist=args.min_dist)

    coords = pd.concat(
        [meta, pd.DataFrame(emb, columns=["umap1", "umap2", "umap3"])], axis=1
    )
    coords_path = args.output_dir / "umap_original_coords.csv"
    coords.to_csv(coords_path, index=False)
    log.info("Saved coords → %s", coords_path)

    plot_2d(emb, meta, args.output_dir)
    plot_3d(emb, meta, args.output_dir)
    log.info("All original-only UMAP plots saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
