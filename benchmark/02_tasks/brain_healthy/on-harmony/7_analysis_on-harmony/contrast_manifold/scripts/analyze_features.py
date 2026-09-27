#!/usr/bin/env python
"""
Feature importance analysis for the joint original + synthetic_v19 UMAP.

Three complementary analyses:

  1. PCA loadings  — which features drive the top principal components
     (and therefore the UMAP, since UMAP runs on PCA-compressed data).
     Plot: signed loading bar charts for PC1-PC3, colored by radiomics family.

  2. Feature divergence — which features differ most between original and
     synthetic, measured as |mean_synth − mean_orig| in normalized units.
     Plot: horizontal bar chart ranked by divergence, colored by family.

  3. UMAP-axis correlation — Spearman r between each feature and each
     UMAP axis (computed on combined original + synthetic).
     Plot: heatmap (top features × 3 axes) + 3 ranked bar charts.

Outputs (in --output_dir):
  feature_pca_loadings.{pdf,png}
  feature_divergence.{pdf,png}
  feature_umap_corr_heatmap.{pdf,png}
  feature_umap_corr_axis1.{pdf,png}
  feature_umap_corr_axis2.{pdf,png}
  feature_umap_corr_axis3.{pdf,png}

Usage:
  run_job --gpus 0 --slot 0 --wait -- .venv/bin/python \\
    analysis/contrast_manifold/scripts/analyze_features.py
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
import umap
import yaml
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

META_COLS = [
    "subject", "session", "modality_id", "modality", "acq_tag",
    "scanner_model", "scanner_vendor", "scanner", "vendor",
    "cohort_category", "cohort", "image_path", "mask_path", "label_map_path",
]

FAMILY_COLORS = {
    "firstorder": "#E53935",
    "glcm":       "#1E88E5",
    "glrlm":      "#43A047",
    "glszm":      "#FB8C00",
    "gldm":       "#8E24AA",
    "ngtdm":      "#00ACC1",
    "shape":      "#6D4C41",
    "other":      "#9E9E9E",
}


def get_family(name: str) -> str:
    for fam in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape"):
        if fam in name.lower():
            return fam
    return "other"


def shorten(name: str) -> str:
    """Strip radiomics family prefix: 'glcm_Contrast' → 'Contrast'."""
    parts = name.split("_", 1)
    return parts[1] if len(parts) > 1 else name


def family_legend_handles() -> list:
    return [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]


# ── Data loading ──────────────────────────────────────────────────────────────


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    renames = {
        "modality_id":    "modality",
        "scanner_model":  "scanner",
        "scanner_vendor": "vendor",
        "cohort_category":"cohort",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns and v not in df.columns})
    for col in ["subject", "session", "modality", "scanner", "cohort", "image_path"]:
        if col not in df.columns:
            df[col] = "unknown"
    log.info("Loaded %s: %d rows", path.name, len(df))
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and not c.startswith("diagnostics_")]


# ── Feature selection helpers ─────────────────────────────────────────────────


def load_feature_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def apply_feature_filter(
    feat_names: list[str], config: dict
) -> tuple[list[str], list[str]]:
    """Return (kept, excluded) column name lists based on the config."""
    exclude_families = set(config.get("exclude_families", []))
    exclude_features = set(config.get("exclude_features", []))
    kept, excluded = [], []
    for name in feat_names:
        if get_family(name) in exclude_families or name in exclude_features:
            excluded.append(name)
        else:
            kept.append(name)
    return kept, excluded


def write_feature_log(
    kept: list[str],
    excluded: list[str],
    output_dir: Path,
    config_path: Path | None,
) -> None:
    lines = [
        "Feature selection log",
        f"Config: {config_path or 'none (all features kept)'}",
        f"Kept:     {len(kept)} features",
        f"Excluded: {len(excluded)} features",
        "",
        "=== EXCLUDED ===",
    ]
    for name in sorted(excluded):
        lines.append(f"  {name}  [{get_family(name)}]")
    lines += ["", "=== KEPT ==="]
    for name in sorted(kept):
        lines.append(f"  {name}  [{get_family(name)}]")
    log_path = output_dir / "feature_selection_log.txt"
    log_path.write_text("\n".join(lines) + "\n")
    log.info("Feature log → %s  (kept %d, excluded %d)", log_path, len(kept), len(excluded))


# ── Pipeline (matches plot_umap_joint.py exactly) ────────────────────────────


def build_matrix(
    df_orig: pd.DataFrame, df_synth: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols_orig  = set(feature_cols(df_orig))
    cols_synth = set(feature_cols(df_synth))
    common     = sorted(cols_orig & cols_synth)
    if not common:
        raise ValueError("No common feature columns between the two CSVs.")
    log.info("Common feature columns: %d", len(common))

    X_orig  = df_orig[common].to_numpy(dtype=np.float64)
    X_synth = df_synth[common].to_numpy(dtype=np.float64)
    X_all   = np.vstack([X_orig, X_synth])

    nan_mask = np.isnan(X_all).mean(axis=0) < 1.0
    X_all    = X_all[:, nan_mask]
    common   = [c for c, k in zip(common, nan_mask) if k]

    imputer = SimpleImputer(strategy="mean")
    X_all   = imputer.fit_transform(X_all)

    var_mask = X_all.var(axis=0) > 0
    X_all    = X_all[:, var_mask]
    common   = [c for c, k in zip(common, var_mask) if k]

    n_orig = len(df_orig)
    log.info("Feature matrix: %d rows × %d cols", len(X_all), len(common))
    return X_all[:n_orig], X_all[n_orig:], common


def run_pca(
    X_orig: np.ndarray, X_synth: np.ndarray
) -> tuple[np.ndarray, np.ndarray, PCA]:
    X_all = np.vstack([X_orig, X_synth])
    pca_full = PCA(random_state=42)
    pca_full.fit(X_all)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = max(3, min(int(np.searchsorted(cumvar, 0.95)) + 1,
                        X_all.shape[0] - 1, X_all.shape[1]))
    pca = PCA(n_components=n_comp, random_state=42)
    X_all_pca = pca.fit_transform(X_all)
    log.info("PCA: %d components → %.1f%% variance", n_comp, cumvar[n_comp - 1] * 100)
    n_orig = len(X_orig)
    return X_all_pca[:n_orig], X_all_pca[n_orig:], pca


def run_umap(
    X_orig_pca: np.ndarray, X_synth_pca: np.ndarray,
    n_neighbors: int, min_dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    X_all = np.vstack([X_orig_pca, X_synth_pca])
    n_orig = len(X_orig_pca)
    log.info("Fitting UMAP jointly on %d points …", len(X_all))
    reducer = umap.UMAP(
        n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
        metric="euclidean", random_state=42, verbose=False, low_memory=False,
    )
    emb = reducer.fit_transform(X_all)
    log.info("UMAP done.")
    return emb[:n_orig], emb[n_orig:]


# ── Save helper ───────────────────────────────────────────────────────────────


def save_fig(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        p = output_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", p)
    plt.close(fig)


# ── Analysis 1: PCA loadings ─────────────────────────────────────────────────


def plot_pca_loadings(
    pca: PCA, feat_cols: list[str], output_dir: Path, top_n: int = 15, n_pcs: int = 3
) -> None:
    """
    For each of the top n_pcs principal components, show a horizontal bar chart
    of the top_n features by |loading|, with signed bars (direction matters).
    Bars are colored by radiomics family.
    """
    components = pca.components_          # (n_components, n_features)
    ev         = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, n_pcs, figsize=(6 * n_pcs, 9), sharey=False)
    if n_pcs == 1:
        axes = [axes]

    seen_families: set[str] = set()

    for ax, k in zip(axes, range(n_pcs)):
        loadings = components[k]                                 # (n_features,)
        order    = np.argsort(np.abs(loadings))[::-1][:top_n]   # top_n by |loading|
        top_loads  = loadings[order]
        top_names  = [feat_cols[i]    for i in order]
        families   = [get_family(f)   for f in top_names]
        colors     = [FAMILY_COLORS[f] for f in families]
        short_names = [shorten(f)     for f in top_names]
        seen_families.update(families)

        # Bars sorted so the largest |loading| is at the top
        y = np.arange(top_n)
        ax.barh(y[::-1], top_loads, color=colors, edgecolor="none", height=0.7)
        ax.set_yticks(y[::-1])
        ax.set_yticklabels(short_names, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Loading (signed)")
        ax.set_title(f"PC{k + 1}  ({ev[k]:.1f}% var)", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(color=FAMILY_COLORS[f], label=f)
               for f in FAMILY_COLORS if f in seen_families]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"PCA feature loadings — top {top_n} features per component\n"
                 f"(features that drive the UMAP geometry)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, output_dir, "feature_pca_loadings")


# ── Analysis 2: Feature divergence ───────────────────────────────────────────


def compute_cohens_d(
    df_orig_raw: pd.DataFrame, df_synth_raw: pd.DataFrame
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Cohen's d = (mean_synth - mean_orig) / pooled_std for each feature
    shared between the two raw (un-normalized) CSVs.

    Cohen's d is normalization-independent: it measures the mean difference in
    units of the pooled within-group standard deviation, so features with very
    different raw scales are directly comparable.

    Returns (feat_cols, cohens_d, mean_orig, mean_synth).
    """
    common = sorted(set(feature_cols(df_orig_raw)) & set(feature_cols(df_synth_raw)))

    X_o = df_orig_raw[common].to_numpy(dtype=np.float64)
    X_s = df_synth_raw[common].to_numpy(dtype=np.float64)

    mo = np.nanmean(X_o, axis=0)
    ms = np.nanmean(X_s, axis=0)
    so = np.nanstd(X_o, axis=0, ddof=1)
    ss = np.nanstd(X_s, axis=0, ddof=1)

    n_o, n_s = X_o.shape[0], X_s.shape[0]
    pooled_std = np.sqrt(((n_o - 1) * so**2 + (n_s - 1) * ss**2) / (n_o + n_s - 2))
    pooled_std = np.where(pooled_std > 0, pooled_std, np.nan)

    d = (ms - mo) / pooled_std   # signed: positive = synthetic > original
    return common, d, mo, ms


def plot_feature_divergence(
    df_orig_raw: pd.DataFrame, df_synth_raw: pd.DataFrame,
    output_dir: Path, top_n: int = 30,
) -> None:
    """
    Cohen's d for each feature (raw, un-normalized values).

    Cohen's d = (mean_synth − mean_orig) / pooled_std.
    Bars are signed (direction matters), ranked by |d|.
    This is normalization-independent: the pooled within-group std is the
    reference, so no dataset's distribution is privileged.

    Rule of thumb: |d| < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, >0.8 large.
    """
    feat_cols_d, d, mo, ms = compute_cohens_d(df_orig_raw, df_synth_raw)

    abs_d  = np.abs(d)
    order  = np.argsort(abs_d)[::-1][:top_n]
    top_d     = d[order]
    top_absd  = abs_d[order]
    top_mo    = mo[order]
    top_ms    = ms[order]
    top_names  = [feat_cols_d[i]  for i in order]
    families   = [get_family(f)   for f in top_names]
    colors     = [FAMILY_COLORS[f] for f in families]
    short_names = [shorten(f)     for f in top_names]

    fig, ax = plt.subplots(figsize=(11, top_n * 0.36 + 2.5))
    y = np.arange(top_n)
    ax.barh(y[::-1], top_d, color=colors, edgecolor="none", height=0.7)

    # Reference lines for effect-size thresholds
    for thresh, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        for sign in (+1, -1):
            ax.axvline(sign * thresh, color="grey", linewidth=0.6,
                       linestyle="--", alpha=0.5)
    ax.text( 0.81, top_n - 0.5, "large",  fontsize=7, color="grey", va="top")
    ax.text(-0.81, top_n - 0.5, "large",  fontsize=7, color="grey", va="top", ha="right")

    # Annotate bars with raw means
    x_max = np.nanmax(np.abs(top_d)) * 1.05
    for i, (d_val, mo_val, ms_val) in enumerate(zip(top_d, top_mo, top_ms)):
        offset = 0.02 * x_max * np.sign(d_val) if d_val != 0 else 0.02 * x_max
        ax.text(d_val + offset, top_n - 1 - i,
                f"orig={mo_val:.3g}  synth={ms_val:.3g}",
                va="center", fontsize=6, color="#333333")

    ax.set_yticks(y[::-1])
    ax.set_yticklabels(short_names, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d  =  (mean_synth − mean_orig) / pooled_std\n"
                  "(positive = synthetic > original; dashed lines = effect-size thresholds)")
    ax.set_title(f"Feature divergence — top {top_n} features by |Cohen's d|\n"
                 f"(normalization-independent; computed on raw feature values)",
                 fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(color=FAMILY_COLORS[f], label=f)
               for f in FAMILY_COLORS if f in set(families)]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.8)
    fig.tight_layout()
    save_fig(fig, output_dir, "feature_divergence")


# ── Analysis 3: UMAP-axis correlation ────────────────────────────────────────


def compute_umap_correlations(
    X_all: np.ndarray, emb_all: np.ndarray, feat_cols: list[str]
) -> np.ndarray:
    """
    Spearman r between each feature and each UMAP axis.
    Returns corr matrix of shape (n_features, 3).
    """
    log.info("Computing Spearman correlations (%d features × 3 UMAP axes) …", len(feat_cols))
    n_feat = len(feat_cols)
    corr   = np.zeros((n_feat, 3))
    for j in range(n_feat):
        for k in range(3):
            r, _ = spearmanr(X_all[:, j], emb_all[:, k])
            corr[j, k] = r
    return corr


def plot_umap_corr_heatmap(
    corr: np.ndarray, feat_cols: list[str], output_dir: Path, top_n: int = 25
) -> None:
    """
    Heatmap: rows = top features (by max |r| across axes), cols = UMAP axes.
    Color = Spearman r (diverging, −1 to +1).
    """
    max_abs_r = np.abs(corr).max(axis=1)
    order     = np.argsort(max_abs_r)[::-1][:top_n]
    corr_sub  = corr[order]
    names_sub = [shorten(feat_cols[i]) for i in order]
    families  = [get_family(feat_cols[i]) for i in order]
    row_colors = [FAMILY_COLORS[f] for f in families]

    fig, ax = plt.subplots(figsize=(7, top_n * 0.40 + 3))
    im = ax.imshow(corr_sub, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["UMAP-1", "UMAP-2", "UMAP-3"], fontsize=10)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names_sub, fontsize=8)

    # Annotate each cell with r value
    for i in range(top_n):
        for j in range(3):
            r = corr_sub[i, j]
            ax.text(j, i, f"{r:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(r) > 0.5 else "black")

    # Family color dots on the left y-axis
    for i, col in enumerate(row_colors):
        ax.plot(-0.55, i, "s", color=col, markersize=8,
                transform=ax.get_yaxis_transform(), clip_on=False)

    plt.colorbar(im, ax=ax, label="Spearman r", shrink=0.6, pad=0.02)
    ax.set_title(f"UMAP-axis correlation — top {top_n} features\n"
                 f"(Spearman r between feature value and UMAP coordinate)",
                 fontsize=11)

    handles = [mpatches.Patch(color=FAMILY_COLORS[f], label=f)
               for f in FAMILY_COLORS if f in set(families)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_fig(fig, output_dir, "feature_umap_corr_heatmap")


def plot_umap_corr_per_axis(
    corr: np.ndarray, feat_cols: list[str], output_dir: Path, top_n: int = 15
) -> None:
    """
    For each UMAP axis: horizontal bar chart of top_n features by |Spearman r|,
    with signed bars.  One PDF/PNG per axis.
    """
    for k, axis_name in enumerate(["UMAP-1", "UMAP-2", "UMAP-3"]):
        rs    = corr[:, k]
        order = np.argsort(np.abs(rs))[::-1][:top_n]
        top_r = rs[order]
        top_names = [feat_cols[i]     for i in order]
        families  = [get_family(f)    for f in top_names]
        colors    = [FAMILY_COLORS[f] for f in families]
        short_names = [shorten(f)     for f in top_names]

        fig, ax = plt.subplots(figsize=(7, 7))
        y = np.arange(top_n)
        ax.barh(y[::-1], top_r, color=colors, edgecolor="none", height=0.7)
        ax.set_yticks(y[::-1])
        ax.set_yticklabels(short_names, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("Spearman r (signed)")
        ax.set_title(f"Top {top_n} features correlated with {axis_name}\n"
                     f"(positive r = higher feature value → higher UMAP coord)",
                     fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

        handles = [mpatches.Patch(color=FAMILY_COLORS[f], label=f)
                   for f in FAMILY_COLORS if f in set(families)]
        ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.8)
        fig.tight_layout()
        save_fig(fig, output_dir, f"feature_umap_corr_axis{k + 1}")


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--original_csv", type=Path,
        default=Path("analysis/contrast_manifold/outputs/data/original/roi_mask/"
                     "on_harmony_features_normalized_combined_downsampled100.csv"),
    )
    p.add_argument(
        "--synthetic_csv", type=Path,
        default=Path("analysis/contrast_manifold/outputs/data/synthetic_v19/roi_mask/"
                     "synthetic_v19_features_normalized_combined.csv"),
    )
    p.add_argument(
        "--original_csv_raw", type=Path,
        default=Path("analysis/contrast_manifold/outputs/data/original/roi_mask/"
                     "on_harmony_features.csv"),
        help="Raw (un-normalized) original features for divergence analysis.",
    )
    p.add_argument(
        "--synthetic_csv_raw", type=Path,
        default=Path("analysis/contrast_manifold/outputs/data/synthetic_v19/roi_mask/"
                     "synthetic_v19_features.csv"),
        help="Raw (un-normalized) synthetic features for divergence analysis.",
    )
    p.add_argument(
        "--output_dir", type=Path,
        default=Path("analysis/contrast_manifold/outputs/plots/feature_analysis"),
    )
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist",    type=float, default=0.1)
    p.add_argument("--top_n_loadings",    type=int, default=15,
                   help="Features per PC in loadings plot")
    p.add_argument("--top_n_divergence",  type=int, default=30,
                   help="Features shown in divergence plot")
    p.add_argument("--top_n_corr_heatmap",type=int, default=25,
                   help="Features shown in correlation heatmap")
    p.add_argument("--top_n_corr_axis",   type=int, default=15,
                   help="Features per axis in per-axis correlation plots")
    p.add_argument("--n_pcs", type=int, default=3,
                   help="Number of PCs to show in loadings plot")
    p.add_argument(
        "--feature_config", type=Path, default=None,
        help="YAML config specifying exclude_families and exclude_features. "
             "When set, outputs go into a 'feat_selected/' subfolder.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for p in (args.original_csv, args.synthetic_csv,
              args.original_csv_raw, args.synthetic_csv_raw):
        if not p.exists():
            raise FileNotFoundError(p)

    if args.feature_config:
        args.output_dir = args.output_dir / "feat_selected"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load combined-normalized features (for PCA / UMAP / correlation) ──
    df_orig  = load_csv(args.original_csv)
    df_synth = load_csv(args.synthetic_csv)

    # ── Feature selection ─────────────────────────────────────────────────
    if args.feature_config:
        cfg = load_feature_config(args.feature_config)
        all_feat = sorted(set(feature_cols(df_orig)) | set(feature_cols(df_synth)))
        kept, excluded_list = apply_feature_filter(all_feat, cfg)
        write_feature_log(kept, excluded_list, args.output_dir, args.feature_config)
        exclude_set = set(excluded_list)
        df_orig  = df_orig.drop(columns=[c for c in exclude_set if c in df_orig.columns])
        df_synth = df_synth.drop(columns=[c for c in exclude_set if c in df_synth.columns])
        log.info("Feature filter applied: %d kept, %d excluded", len(kept), len(excluded_list))

    X_orig, X_synth, feat_cols = build_matrix(df_orig, df_synth)

    # ── Load raw features (for divergence only) ───────────────────────────
    df_orig_raw  = load_csv(args.original_csv_raw)
    df_synth_raw = load_csv(args.synthetic_csv_raw)

    # Apply same feature filter to raw CSVs for divergence analysis
    if args.feature_config:
        df_orig_raw  = df_orig_raw.drop(
            columns=[c for c in exclude_set if c in df_orig_raw.columns])
        df_synth_raw = df_synth_raw.drop(
            columns=[c for c in exclude_set if c in df_synth_raw.columns])

    # ── PCA ───────────────────────────────────────────────────────────────
    X_orig_pca, X_synth_pca, pca = run_pca(X_orig, X_synth)

    # ── UMAP ──────────────────────────────────────────────────────────────
    emb_orig, emb_synth = run_umap(
        X_orig_pca, X_synth_pca,
        n_neighbors=args.n_neighbors, min_dist=args.min_dist,
    )

    # Combined arrays for correlation analysis
    X_all   = np.vstack([X_orig, X_synth])
    emb_all = np.vstack([emb_orig, emb_synth])

    # ── Analysis 1: PCA loadings ──────────────────────────────────────────
    log.info("─── Analysis 1: PCA loadings ───")
    plot_pca_loadings(pca, feat_cols, args.output_dir,
                      top_n=args.top_n_loadings, n_pcs=args.n_pcs)

    # ── Analysis 2: Feature divergence (on raw features, Cohen's d) ──────
    log.info("─── Analysis 2: Feature divergence ───")
    plot_feature_divergence(df_orig_raw, df_synth_raw, args.output_dir,
                            top_n=args.top_n_divergence)

    # ── Analysis 3: UMAP-axis correlation ────────────────────────────────
    log.info("─── Analysis 3: UMAP-axis correlation ───")
    corr = compute_umap_correlations(X_all, emb_all, feat_cols)
    plot_umap_corr_heatmap(corr, feat_cols, args.output_dir,
                           top_n=args.top_n_corr_heatmap)
    plot_umap_corr_per_axis(corr, feat_cols, args.output_dir,
                            top_n=args.top_n_corr_axis)

    log.info("All analyses saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
