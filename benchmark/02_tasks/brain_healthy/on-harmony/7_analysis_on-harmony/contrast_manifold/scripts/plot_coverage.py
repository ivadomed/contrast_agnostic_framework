#!/usr/bin/env python
"""
Coverage analysis: how well does synthetic data tile the original contrast manifold?

Fits PCA on the *original* data only (to define the reference contrast manifold),
then projects synthetic onto those axes and measures:

  - Recall at ε  : fraction of original points with a synthetic neighbour within ε
                   (computed in full PC space, reported per modality + overall)
  - Coverage heatmap : PC1–PC2 grid coloured by NN distance to nearest synthetic
                       (green = well covered, red = gap)
  - Density comparison : KDE of original vs synthetic in PC1–PC2
  - Spread check : PC score distributions (detects mode collapse)
  - Hull coverage fraction: % of original convex-hull grid cells covered at ε=1

Outputs (in --output_dir):
  coverage_heatmap.{pdf,png}
  coverage_recall_curve.{pdf,png}
  coverage_pc_spread.{pdf,png}
  coverage_metrics.csv

Usage:
  .venv/bin/python analysis/contrast_manifold/scripts/plot_coverage.py \\
    --original_csv  .../on_harmony_features_normalized_combined_downsampled100_feat_selected.csv \\
    --synthetic_csv .../synthetic_v19_c_features_normalized_combined_feat_selected.csv \\
    --output_dir    .../coverage
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
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

META_COLS = [
    "subject", "session", "modality_id", "modality", "acq_tag",
    "scanner_model", "scanner_vendor", "scanner", "vendor",
    "cohort_category", "cohort", "image_path", "mask_path", "label_map_path",
]

MODALITY_PALETTE = {
    "T1w":   "#E53935",
    "T2w":   "#1E88E5",
    "FLAIR": "#43A047",
    "dwi":   "#FB8C00",
    "bold":  "#8E24AA",
    "epi":   "#00ACC1",
    "GRE":   "#6D4C41",
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    renames = {
        "modality_id": "modality", "scanner_model": "scanner",
        "scanner_vendor": "vendor", "cohort_category": "cohort",
    }
    df = df.rename(columns={k: v for k, v in renames.items()
                             if k in df.columns and v not in df.columns})
    for col in ["subject", "session", "modality", "scanner"]:
        if col not in df.columns:
            df[col] = "unknown"
    log.info("Loaded %s: %d rows", path.name, len(df))
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in META_COLS and not c.startswith("diagnostics_")]


def base_mod(m: str) -> str:
    return m.split("_")[0] if m else "unknown"


def build_matrix(
    df_orig: pd.DataFrame, df_synth: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    common = sorted(set(feature_cols(df_orig)) & set(feature_cols(df_synth)))
    X_orig  = df_orig[common].to_numpy(dtype=np.float64)
    X_synth = df_synth[common].to_numpy(dtype=np.float64)
    X_all   = np.vstack([X_orig, X_synth])

    nan_mask = np.isnan(X_all).mean(axis=0) < 1.0
    X_all    = X_all[:, nan_mask]

    X_all = SimpleImputer(strategy="mean").fit_transform(X_all)

    var_mask = X_all.var(axis=0) > 0
    X_all    = X_all[:, var_mask]

    n = len(df_orig)
    log.info("Feature matrix: %d orig + %d synth × %d features",
             n, len(df_synth), X_all.shape[1])
    return X_all[:n], X_all[n:]


def fit_pca_orig(
    X_orig: np.ndarray, X_synth: np.ndarray, var_threshold: float = 0.95
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """PCA fit on original only; project synthetic onto those axes."""
    pca_full = PCA(random_state=42).fit(X_orig)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = max(3, int(np.searchsorted(cumvar, var_threshold)) + 1)
    n_comp = min(n_comp, X_orig.shape[0] - 1, X_orig.shape[1])
    pca = PCA(n_components=n_comp, random_state=42).fit(X_orig)
    log.info("PCA (orig-only): %d components → %.1f%% orig variance",
             n_comp, cumvar[n_comp - 1] * 100)
    return pca.transform(X_orig), pca.transform(X_synth), pca


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_coverage_metrics(
    X_orig: np.ndarray,
    X_synth: np.ndarray,
    meta_orig: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-modality coverage metrics (computed in full PC space).

    ε values are set relative to the typical orig→orig NN distance (the "resolution"
    of the original manifold sampling), so they are meaningful across feature spaces.
    Fractions reported: 0.25×, 0.5×, 1×, 2×, 4× the reference scale.

    Also reports ε_50 / ε_90: the radius at which 50% / 90% of originals are covered.
    """
    nn_o2s = NearestNeighbors(n_neighbors=1).fit(X_synth)
    nn_s2o = NearestNeighbors(n_neighbors=1).fit(X_orig)
    nn_ss  = NearestNeighbors(n_neighbors=2).fit(X_synth)
    nn_oo  = NearestNeighbors(n_neighbors=2).fit(X_orig)

    d_o2s = nn_o2s.kneighbors(X_orig)[0][:, 0]
    d_s2o = nn_s2o.kneighbors(X_synth)[0][:, 0]
    d_ss  = nn_ss.kneighbors(X_synth)[0][:, 1]
    d_oo  = nn_oo.kneighbors(X_orig)[0][:, 1]   # orig self-NN (scale reference)

    ref_scale = float(np.median(d_oo))
    log.info("Reference scale (median orig self-NN dist): %.4f", ref_scale)
    eps_fracs = [0.25, 0.5, 1.0, 2.0, 4.0]
    eps_values = [ref_scale * f for f in eps_fracs]

    meta = meta_orig.copy().reset_index(drop=True)
    meta["base_mod"] = meta["modality"].apply(base_mod)

    records = []
    for mod in sorted(meta["base_mod"].unique()) + ["all"]:
        mask = np.ones(len(X_orig), dtype=bool) if mod == "all" \
               else (meta["base_mod"] == mod).values
        d = d_o2s[mask]
        eps_sorted = np.sort(d)
        n = len(d)

        row: dict = {
            "modality":                     mod,
            "n_original":                   int(mask.sum()),
            "ref_scale_median_orig_nn":     round(ref_scale, 4),
            "mean_nn_dist_orig_to_synth":   round(float(d.mean()), 4),
            "median_nn_dist_orig_to_synth": round(float(np.median(d)), 4),
            "p90_nn_dist_orig_to_synth":    round(float(np.percentile(d, 90)), 4),
            # ε where 50% / 90% of this modality's originals are covered
            "eps_50pct_recall":  round(float(eps_sorted[int(0.50 * n) - 1]), 4),
            "eps_90pct_recall":  round(float(eps_sorted[min(int(0.90 * n), n - 1)]), 4),
        }
        for frac, eps in zip(eps_fracs, eps_values):
            row[f"recall_{frac:.2f}x_scale"] = round(float((d <= eps).mean()), 4)
        records.append(row)

    df = pd.DataFrame(records)
    df["n_synthetic"] = len(X_synth)
    for frac, eps in zip(eps_fracs, eps_values):
        df[f"precision_{frac:.2f}x_scale"] = round(float((d_s2o <= eps).mean()), 4)
    df["synth_diversity_mean_self_nn"] = round(float(d_ss.mean()), 4)
    # diversity ratio: if < 1 synthetic is more collapsed than original; > 1 is broader
    df["synth_orig_spread_ratio"] = round(float(d_ss.mean() / max(ref_scale, 1e-9)), 4)
    return df


def compute_scanner_modality_metrics(
    X_orig: np.ndarray,
    X_synth: np.ndarray,
    meta_orig: pd.DataFrame,
    ref_scale: float,
) -> pd.DataFrame:
    """Per (vendor × base_modality) cluster coverage — captures scanner subclusters."""
    nn_o2s = NearestNeighbors(n_neighbors=1).fit(X_synth)
    d_o2s  = nn_o2s.kneighbors(X_orig)[0][:, 0]

    meta = meta_orig.copy().reset_index(drop=True)
    meta["base_mod"]   = meta["modality"].apply(base_mod)
    vendor_col         = "vendor" if "vendor" in meta.columns else "scanner"
    meta["vendor_str"] = meta[vendor_col].fillna("unknown") if vendor_col in meta.columns else "unknown"

    eps_fracs  = [0.25, 0.5, 1.0, 2.0, 4.0]
    eps_values = [ref_scale * f for f in eps_fracs]

    records = []
    for (vendor, mod), grp in meta.groupby(["vendor_str", "base_mod"]):
        idx = grp.index.values
        d   = d_o2s[idx]
        row: dict = {
            "vendor":              vendor,
            "modality":            mod,
            "cluster":             f"{vendor}/{mod}",
            "n_original":          len(d),
            "mean_nn_dist":        round(float(d.mean()), 4),
            "mean_nn_dist_scaled": round(float(d.mean() / max(ref_scale, 1e-9)), 3),
        }
        for frac, eps in zip(eps_fracs, eps_values):
            row[f"recall_{frac:.2f}x"] = round(float((d <= eps).mean()), 4)
        records.append(row)
    return pd.DataFrame(records)


def plot_scanner_modality_heatmap(
    df_sc: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Two-panel heatmap: vendor × modality, colored by coverage distance and recall."""
    vendors    = sorted(df_sc["vendor"].unique())
    modalities = sorted(df_sc["modality"].unique())
    nv, nm     = len(vendors), len(modalities)

    dist_mat   = np.full((nv, nm), np.nan)
    recall_mat = np.full((nv, nm), np.nan)
    n_mat      = np.zeros((nv, nm), dtype=int)
    for _, row in df_sc.iterrows():
        vi, mi = vendors.index(row["vendor"]), modalities.index(row["modality"])
        dist_mat[vi, mi]   = row["mean_nn_dist_scaled"]
        recall_mat[vi, mi] = row["recall_4.00x"] * 100
        n_mat[vi, mi]      = row["n_original"]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, nm * 1.4), max(3, nv * 1.4)))
    panels = [
        (axes[0], dist_mat,   "Mean NN dist / ref_scale\n(lower = better covered)", ".1f",  "RdYlGn_r"),
        (axes[1], recall_mat, "Recall at 4× scale (%)\n(higher = better covered)",  ".0f",  "RdYlGn"),
    ]
    for ax, data, title, fmt, cmap in panels:
        vmax = float(np.nanmax(data)) if not np.all(np.isnan(data)) else 1.0
        im   = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.set_xticks(range(nm)); ax.set_xticklabels(modalities, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(nv)); ax.set_yticklabels(vendors, fontsize=8)
        ax.set_title(title, fontsize=9)
        for vi in range(nv):
            for mi in range(nm):
                if np.isnan(data[vi, mi]):
                    ax.text(mi, vi, "—", ha="center", va="center", fontsize=8, color="gray")
                else:
                    n   = n_mat[vi, mi]
                    txt = f"{data[vi, mi]:{fmt}}\n(n={n})"
                    dark = data[vi, mi] > vmax * 0.55
                    ax.text(mi, vi, txt, ha="center", va="center", fontsize=7,
                            color="white" if dark else "black")

    fig.suptitle("Coverage by scanner vendor × modality", fontsize=11)
    fig.tight_layout()
    _save(fig, output_dir, "coverage_scanner_modality")


def _hull_coverage_fraction(
    X_orig_2d: np.ndarray,
    X_synth_2d: np.ndarray,
    eps: float = 1.0,
    grid_size: int = 60,
) -> float:
    """Fraction of original convex-hull grid cells covered by synthetic at ε."""
    pad = 0.3
    gx = np.linspace(X_orig_2d[:, 0].min() - pad, X_orig_2d[:, 0].max() + pad, grid_size)
    gy = np.linspace(X_orig_2d[:, 1].min() - pad, X_orig_2d[:, 1].max() + pad, grid_size)
    gxx, gyy = np.meshgrid(gx, gy)
    grid = np.column_stack([gxx.ravel(), gyy.ravel()])

    try:
        tri = Delaunay(X_orig_2d)
        inside = tri.find_simplex(grid) >= 0
    except Exception:
        inside = np.ones(len(grid), dtype=bool)

    if inside.sum() == 0:
        return 0.0

    nn = NearestNeighbors(n_neighbors=1).fit(X_synth_2d)
    dists = nn.kneighbors(grid[inside])[0][:, 0]
    return float((dists <= eps).mean())


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", output_dir / f"{stem}.pdf")


def plot_coverage_heatmap(
    X_orig_pca: np.ndarray,
    X_synth_pca: np.ndarray,
    meta_orig: pd.DataFrame,
    output_dir: Path,
    grid_size: int = 60,
) -> None:
    """
    Left: NN-distance heatmap (green = covered, red = gap) within original hull.
    Right: synthetic KDE (blue) overlaid with original KDE contours (red).
    """
    meta = meta_orig.copy().reset_index(drop=True)
    meta["base_mod"] = meta["modality"].apply(base_mod)
    modalities = sorted(meta["base_mod"].unique())

    xo, yo = X_orig_pca[:, 0], X_orig_pca[:, 1]
    xs, ys = X_synth_pca[:, 0], X_synth_pca[:, 1]

    pad = 0.5
    gx = np.linspace(xo.min() - pad, xo.max() + pad, grid_size)
    gy = np.linspace(yo.min() - pad, yo.max() + pad, grid_size)
    gxx, gyy = np.meshgrid(gx, gy)
    grid = np.column_stack([gxx.ravel(), gyy.ravel()])

    nn = NearestNeighbors(n_neighbors=1).fit(X_synth_pca[:, :2])
    dist_map = nn.kneighbors(grid)[0][:, 0].reshape(grid_size, grid_size)

    try:
        tri = Delaunay(X_orig_pca[:, :2])
        inside = tri.find_simplex(grid) >= 0
        dist_map = np.where(inside.reshape(grid_size, grid_size), dist_map, np.nan)
    except Exception:
        pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

    # ── Panel 1: coverage heatmap ─────────────────────────────────────────────
    im = ax1.pcolormesh(gxx, gyy, dist_map, cmap="RdYlGn_r", shading="auto")
    cb = plt.colorbar(im, ax=ax1, shrink=0.8)
    cb.set_label("Distance to nearest synthetic (PC space)", fontsize=8)

    for mod in modalities:
        mask = (meta["base_mod"] == mod).values
        ax1.scatter(xo[mask], yo[mask],
                    c=MODALITY_PALETTE.get(mod, "#888"), s=20, alpha=0.85,
                    edgecolors="white", linewidths=0.3, zorder=3, label=mod)

    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.set_title("Coverage heatmap — distance to nearest synthetic\n"
                  "(green = covered, red = gap)")
    ax1.legend(fontsize=7, loc="best", framealpha=0.7)

    # ── Panel 2: KDE comparison ───────────────────────────────────────────────
    kde_o = gaussian_kde(X_orig_pca[:, :2].T)
    kde_s = gaussian_kde(X_synth_pca[:, :2].T)
    zo = kde_o(grid.T).reshape(grid_size, grid_size)
    zs = kde_s(grid.T).reshape(grid_size, grid_size)

    ax2.pcolormesh(gxx, gyy, zs, cmap="Blues", shading="auto", alpha=0.65)
    ax2.contour(gxx, gyy, zo, levels=6, colors="darkred", linewidths=0.9, alpha=0.8)
    ax2.scatter(xs, ys, c="#111111", alpha=0.10, s=5, linewidths=0, zorder=1)
    for mod in modalities:
        mask = (meta["base_mod"] == mod).values
        ax2.scatter(xo[mask], yo[mask],
                    c=MODALITY_PALETTE.get(mod, "#888"), s=20, alpha=0.85,
                    edgecolors="white", linewidths=0.3, zorder=3)

    mod_h  = [mpatches.Patch(color=MODALITY_PALETTE.get(m, "#888"), label=m) for m in modalities]
    mod_h += [mpatches.Patch(color="#1a6fc4", alpha=0.6, label="synthetic density")]
    mod_h += [plt.Line2D([0], [0], color="darkred", linewidth=0.9, label="original density")]
    ax2.legend(handles=mod_h, fontsize=7, loc="best", framealpha=0.7)
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    ax2.set_title("Density: synthetic (blue) vs original (red contours)")

    fig.suptitle("Contrast Manifold Coverage — PC1 vs PC2 (original-only PCA axes)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_dir, "coverage_heatmap")


def plot_recall_curve(
    X_orig_pca: np.ndarray,
    X_synth_pca: np.ndarray,
    meta_orig: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Recall vs ε curve: how fast does coverage grow with neighbourhood radius?"""
    meta = meta_orig.copy().reset_index(drop=True)
    meta["base_mod"] = meta["modality"].apply(base_mod)
    modalities = sorted(meta["base_mod"].unique())

    nn = NearestNeighbors(n_neighbors=1).fit(X_synth_pca)
    dists_all = nn.kneighbors(X_orig_pca)[0][:, 0]
    eps_range = np.linspace(0, np.percentile(dists_all, 99), 300)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(eps_range, [(dists_all <= e).mean() for e in eps_range],
            "k-", linewidth=2.5, label="all", zorder=5)

    for mod in modalities:
        mask = (meta["base_mod"] == mod).values
        d = dists_all[mask]
        ax.plot(eps_range, [(d <= e).mean() for e in eps_range],
                color=MODALITY_PALETTE.get(mod, "#888"), linewidth=1.6,
                linestyle="--", label=mod, alpha=0.9)

    for level, label in [(0.9, "90%"), (0.5, "50%")]:
        ax.axhline(level, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
        ax.text(eps_range[-1] * 0.01, level + 0.01, label, fontsize=7, color="gray")

    ax.set_xlabel("ε — distance threshold in full PC space", fontsize=10)
    ax.set_ylabel("Recall (fraction of original points covered)", fontsize=10)
    ax.set_title("Coverage recall curve — how uniformly does synthetic tile the manifold?",
                 fontsize=10)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, eps_range[-1])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, output_dir, "coverage_recall_curve")


def plot_pc_spread(
    X_orig_pca: np.ndarray,
    X_synth_pca: np.ndarray,
    pca: PCA,
    output_dir: Path,
    n_pcs: int = 4,
) -> None:
    """
    PC score KDE — original vs synthetic.
    Similar or wider synthetic spread → good coverage; narrow → mode collapse.
    """
    n_pcs = min(n_pcs, pca.n_components_)
    ev = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, n_pcs, figsize=(4.5 * n_pcs, 4.5))
    if n_pcs == 1:
        axes = [axes]

    for ax, pc in zip(axes, range(n_pcs)):
        vo, vs = X_orig_pca[:, pc], X_synth_pca[:, pc]
        xlim = (min(vo.min(), vs.min()) - 0.5, max(vo.max(), vs.max()) + 0.5)
        xp = np.linspace(*xlim, 300)

        ax.fill_between(xp, gaussian_kde(vo)(xp), alpha=0.35, color="#E53935")
        ax.fill_between(xp, gaussian_kde(vs)(xp), alpha=0.35, color="#1E88E5")
        ax.plot(xp, gaussian_kde(vo)(xp), "#E53935", linewidth=1.5, label="original")
        ax.plot(xp, gaussian_kde(vs)(xp), "#1E88E5", linewidth=1.5, label="synthetic")

        ratio = vs.std() / max(vo.std(), 1e-9)
        ax.set_title(
            f"PC{pc+1} ({ev[pc]:.1f}%)\n"
            f"std  orig={vo.std():.2f}  synth={vs.std():.2f}  ratio={ratio:.2f}",
            fontsize=8,
        )
        ax.set_xlabel(f"PC{pc+1} score", fontsize=8)
        if pc == 0:
            ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "PC score distributions — original vs synthetic\n"
        "ratio > 1 = broader synthetic spread (good); ratio ≪ 1 = mode collapse",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, output_dir, "coverage_pc_spread")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_coverage_analysis(
    original_csv: Path,
    synthetic_csv: Path,
    output_dir: Path,
    grid_size: int = 60,
    n_pcs: int = 4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df_orig  = load_csv(original_csv)
    df_synth = load_csv(synthetic_csv)

    X_orig, X_synth = build_matrix(df_orig, df_synth)
    meta_cols = [c for c in META_COLS if c in df_orig.columns]
    meta_orig = df_orig[meta_cols].reset_index(drop=True)

    X_orig_pca, X_synth_pca, pca = fit_pca_orig(X_orig, X_synth)

    log.info("Computing coverage metrics …")
    metrics = compute_coverage_metrics(X_orig_pca, X_synth_pca, meta_orig)

    # Hull coverage at ε = 1× the reference scale (median orig self-NN distance in 2D)
    ref_2d = float(np.median(
        NearestNeighbors(n_neighbors=2).fit(X_orig_pca[:, :2])
        .kneighbors(X_orig_pca[:, :2])[0][:, 1]
    ))
    hull_cov = _hull_coverage_fraction(X_orig_pca[:, :2], X_synth_pca[:, :2], eps=ref_2d,
                                       grid_size=grid_size)
    metrics["hull_coverage_at_1x_scale_2d"] = round(hull_cov, 4)
    log.info("Hull coverage (ε=1×scale, PC1-PC2): %.1f%%", hull_cov * 100)

    csv_path = output_dir / "coverage_metrics.csv"
    metrics.to_csv(csv_path, index=False)
    log.info("Coverage metrics →\n%s", metrics.to_string(index=False))

    # Per (vendor × modality) breakdown — captures scanner subclusters
    ref_scale = float(metrics.loc[metrics["modality"] == "all", "ref_scale_median_orig_nn"].iloc[0])
    df_sc = compute_scanner_modality_metrics(X_orig_pca, X_synth_pca, meta_orig, ref_scale)
    df_sc.to_csv(output_dir / "coverage_scanner_modality.csv", index=False)
    log.info("Scanner×modality metrics →\n%s", df_sc.to_string(index=False))
    plot_scanner_modality_heatmap(df_sc, output_dir)

    plot_coverage_heatmap(X_orig_pca, X_synth_pca, meta_orig, output_dir, grid_size)
    plot_recall_curve(X_orig_pca, X_synth_pca, meta_orig, output_dir)
    plot_pc_spread(X_orig_pca, X_synth_pca, pca, output_dir, n_pcs)

    log.info("Coverage analysis complete → %s", output_dir)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--original_csv",  type=Path, required=True)
    p.add_argument("--synthetic_csv", type=Path, required=True)
    p.add_argument("--output_dir",    type=Path, required=True)
    p.add_argument("--grid_size",     type=int, default=60)
    p.add_argument("--n_pcs",         type=int, default=4)
    args = p.parse_args()
    run_coverage_analysis(
        args.original_csv, args.synthetic_csv, args.output_dir,
        grid_size=args.grid_size, n_pcs=args.n_pcs,
    )


if __name__ == "__main__":
    main()
