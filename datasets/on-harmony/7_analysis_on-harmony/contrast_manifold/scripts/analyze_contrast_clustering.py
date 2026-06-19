#!/usr/bin/env python
"""
Minimal feature combinations that naturally cluster original MRI contrasts.

Pipeline:
  1. Load normalized original features (downsampled).
  2. Fit LDA on modality labels → discriminant axes that maximally separate contrasts.
  3. Rank individual features by one-way ANOVA F-score across modalities.
  4. For each: plot original-only, then overlay synthetic.

Outputs (in --output_dir):
  lda_original.{png,pdf}          — LDA scatter, original only (LD1 vs LD2)
  lda_with_synthetic.{png,pdf}    — same axes, synthetic overlaid
  lda_loadings.{png,pdf}          — top feature loadings per discriminant axis
  feature_fscore.{png,pdf}        — ANOVA F-score ranking (top 30)
  feature_scatter_top{k}.{png,pdf}— 2D scatter grids of top individual features
  lda_coords.csv                  — LD coordinates for original + synthetic

Usage:
  run_job --gpus 0 --slot 0 --wait -- .venv/bin/python analysis/contrast_manifold/scripts/analyze_contrast_clustering.py \\
    --output_dir analysis/contrast_manifold/outputs/plots/contrast_clustering_v19_c
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
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

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

MPL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", ">"]

SYNTH_COLOR = "#111111"
SYNTH_ALPHA = 0.18

FAMILY_COLORS = {
    "firstorder": "#E53935",
    "glcm":       "#1E88E5",
    "glrlm":      "#43A047",
    "glszm":      "#FB8C00",
    "gldm":       "#8E24AA",
    "ngtdm":      "#00ACC1",
    "shape":      "#6D4C41",
    "other":      "#888888",
}


def get_family(name: str) -> str:
    for fam in FAMILY_COLORS:
        if fam in name.lower():
            return fam
    return "other"


def short_name(name: str) -> str:
    for fam in FAMILY_COLORS:
        prefix = fam + "_"
        if name.lower().startswith(prefix):
            return f"{name[len(prefix):]} [{fam}]"
    return name


def base_mod(m: str) -> str:
    return m.split("_")[0] if m else "unknown"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    renames = {
        "modality_id": "modality", "scanner_model": "scanner",
        "scanner_vendor": "vendor", "cohort_category": "cohort",
    }
    df = df.rename(columns={k: v for k, v in renames.items()
                             if k in df.columns and v not in df.columns})
    for col in ["subject", "session", "modality", "scanner", "vendor", "cohort"]:
        if col not in df.columns:
            df[col] = "unknown"
    log.info("Loaded %s: %d rows", path.name, len(df))
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in META_COLS and not c.startswith("diagnostics_")]


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        p = output_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        log.info("Saved → %s", p)
    plt.close(fig)


# ── LDA ───────────────────────────────────────────────────────────────────────

def fit_lda(X: np.ndarray, y: np.ndarray, n_components: int | None = None):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X, y)
    return lda


def plot_lda_scatter(
    Z_orig: np.ndarray,
    labels_orig: np.ndarray,
    le: LabelEncoder,
    scanners: np.ndarray,
    output_dir: Path,
    Z_synth: np.ndarray | None = None,
    explained: np.ndarray | None = None,
) -> None:
    """LD1 vs LD2 scatter, optionally with synthetic overlaid."""
    modalities = [le.classes_[i] for i in np.unique(labels_orig)]
    sc_list    = sorted(set(scanners))
    sc_marker  = {sc: MPL_MARKERS[i % len(MPL_MARKERS)] for i, sc in enumerate(sc_list)}

    xlabel = f"LD1 ({explained[0]*100:.1f}%)" if explained is not None else "LD1"
    ylabel = f"LD2 ({explained[1]*100:.1f}%)" if explained is not None else "LD2"

    for with_synth in (False, True):
        if with_synth and Z_synth is None:
            continue
        fig, ax = plt.subplots(figsize=(10, 8))

        if with_synth:
            ax.scatter(Z_synth[:, 0], Z_synth[:, 1],
                       c=SYNTH_COLOR, alpha=SYNTH_ALPHA, s=10,
                       linewidths=0, zorder=1, label="synthetic")

        for sc in sc_list:
            for mod in modalities:
                mask = (le.transform([mod])[0] == labels_orig) & (scanners == sc)
                if not mask.any():
                    continue
                ax.scatter(Z_orig[mask, 0], Z_orig[mask, 1],
                           c=MODALITY_PALETTE.get(mod, "#888"),
                           marker=sc_marker[sc], alpha=0.85, s=28,
                           linewidths=0, zorder=2)

        # Modality centroids with labels
        for mod in modalities:
            mask = le.transform([mod])[0] == labels_orig
            cx, cy = Z_orig[mask, 0].mean(), Z_orig[mask, 1].mean()
            ax.annotate(mod, (cx, cy), fontsize=9, fontweight="bold",
                        color=MODALITY_PALETTE.get(mod, "#333"),
                        ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0))

        mod_h = [mpatches.Patch(color=MODALITY_PALETTE.get(m, "#888"), label=m)
                 for m in modalities]
        sc_h  = [plt.Line2D([0], [0], marker=sc_marker[sc], color="gray",
                             linestyle="none", markersize=6, label=sc) for sc in sc_list]
        handles = mod_h + sc_h
        if with_synth:
            handles += [mpatches.Patch(color=SYNTH_COLOR, alpha=0.5, label="synthetic")]
        ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.75, ncol=2)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        title = "LDA — original contrasts" + (" + synthetic" if with_synth else " only")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        stem = "lda_with_synthetic" if with_synth else "lda_original"
        _save(fig, output_dir, stem)


_CLICK_TO_COPY_JS = """
(function() {
    var gd = document.querySelector('.plotly-graph-div');
    gd.on('plotly_click', function(data) {
        var pt = data.points[0];
        var cd = pt.customdata;
        if (!cd) return;
        var text = cd[0] || cd;   // customdata is [path] or path directly
        var toast = document.createElement('div');
        toast.style.cssText = [
            'position:fixed','bottom:24px','left:50%','transform:translateX(-50%)',
            'background:#222','color:#fff','padding:10px 18px','border-radius:6px',
            'font-size:13px','font-family:monospace','z-index:9999',
            'max-width:80%','word-break:break-all','box-shadow:0 2px 8px rgba(0,0,0,.4)'
        ].join(';');
        if (text && text !== 'none') {
            // execCommand must run synchronously inside the click handler (Safari requirement).
            var el = document.createElement('textarea');
            el.value = text;
            el.style.cssText = 'position:fixed;opacity:0;pointer-events:none;';
            document.body.appendChild(el);
            el.select();
            var syncOk = document.execCommand('copy');
            document.body.removeChild(el);
            if (syncOk) {
                toast.textContent = '✓ Copied: ' + text;
            } else if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(function() {
                    toast.textContent = '✓ Copied: ' + text;
                }).catch(function() { toast.textContent = text; });
            } else {
                toast.textContent = text;
            }
        } else {
            toast.textContent = pt.text ? pt.text.replace(/<[^>]+>/g,' ') : 'No path available';
        }
        document.body.appendChild(toast);
        setTimeout(function() {
            toast.style.transition = 'opacity 0.4s';
            toast.style.opacity = '0';
            setTimeout(function() { document.body.removeChild(toast); }, 400);
        }, 3500);
    });
})();
"""


def plot_lda_scatter_3d(
    Z_orig: np.ndarray,
    labels_orig: np.ndarray,
    le: LabelEncoder,
    meta_orig: pd.DataFrame,
    output_dir: Path,
    Z_synth: np.ndarray | None = None,
    meta_synth: pd.DataFrame | None = None,
    explained: np.ndarray | None = None,
) -> None:
    """Interactive Plotly 3D LDA scatter (LD1 × LD2 × LD3).
    Click any point to copy its file path to the clipboard.
    """
    modalities = [le.classes_[i] for i in np.unique(labels_orig)]

    def _pct(i):
        return f" ({explained[i]*100:.1f}%)" if explained is not None and i < len(explained) else ""

    PLOTLY_SYMS = ["circle", "square", "diamond", "cross", "x",
                   "circle-open", "square-open", "diamond-open"]
    sc_list = sorted(meta_orig["scanner"].unique())
    sym_map = {sc: PLOTLY_SYMS[i % len(PLOTLY_SYMS)] for i, sc in enumerate(sc_list)}

    def _hover_and_path(meta: pd.DataFrame) -> tuple[list[str], list[str]]:
        hover, paths = [], []
        for _, r in meta.iterrows():
            path = str(r.get("image_path", "") or "")
            fname = Path(path).name if path else "—"
            hover.append(
                f"<b>{r.get('subject','?')}</b> {r.get('session','?')}<br>"
                f"modality: {r.get('modality','?')}<br>"
                f"scanner:  {r.get('scanner','?')}<br>"
                f"file: {fname}<br>"
                f"<i>{'click to copy path' if path else 'no path available'}</i>"
            )
            paths.append(path if path else "none")
        return hover, paths

    meta_orig_r  = meta_orig.reset_index(drop=True)
    hover_orig, paths_orig = _hover_and_path(meta_orig_r)

    # ── Pre-compute nearest synth per cluster in LDA space ───────────────────
    nearest_synth: dict[str, list[tuple[int, float]]] = {}
    if Z_synth is not None and meta_synth is not None:
        meta_s = meta_synth.reset_index(drop=True)

        # per-cluster → top-3 synthetic neighbours
        centroids: dict[str, np.ndarray] = {}
        records = []
        for mod in modalities:
            mask_mod = (meta_orig_r["base_mod"].values == mod)
            centroid = Z_orig[mask_mod, :3].mean(axis=0)
            centroids[mod] = centroid
            dists   = np.linalg.norm(Z_synth[:, :3] - centroid, axis=1)
            top_idx = np.argsort(dists)[:3]
            nearest_synth[mod] = [(int(i), float(dists[i])) for i in top_idx]
            for rank, (i, d) in enumerate(nearest_synth[mod], 1):
                path = str(meta_s.iloc[i].get("image_path", "") or "")
                records.append({
                    "cluster":    mod, "rank": rank, "synth_idx": i,
                    "distance":   round(d, 6), "image_path": path,
                    "subject":    meta_s.iloc[i].get("subject", "?"),
                    "session":    meta_s.iloc[i].get("session", "?"),
                    "modality_id": meta_s.iloc[i].get("modality",
                                   meta_s.iloc[i].get("modality_id", "?")),
                })
        pd.DataFrame(records).to_csv(output_dir / "nearest_synth_per_cluster.csv", index=False)
        log.info("Nearest synth CSV → %s", output_dir / "nearest_synth_per_cluster.csv")

        # per-synth-point → distance to every centroid
        cent_names = sorted(centroids.keys())
        cent_mat   = np.stack([centroids[m] for m in cent_names])   # (n_mod, 3)
        all_dists  = np.linalg.norm(
            Z_synth[:, :3, np.newaxis] - cent_mat.T[np.newaxis, :, :], axis=1
        )  # (n_synth, n_mod)
        nearest_idx = all_dists.argmin(axis=1)

        synth_rows = []
        for i in range(len(Z_synth)):
            row = {
                "synth_idx":        i,
                "image_path":       str(meta_s.iloc[i].get("image_path", "") or ""),
                "subject":          meta_s.iloc[i].get("subject", "?"),
                "session":          meta_s.iloc[i].get("session", "?"),
                "modality_id":      meta_s.iloc[i].get("modality",
                                    meta_s.iloc[i].get("modality_id", "?")),
                "nearest_centroid": cent_names[nearest_idx[i]],
                "ld1": round(float(Z_synth[i, 0]), 6),
                "ld2": round(float(Z_synth[i, 1]), 6),
                "ld3": round(float(Z_synth[i, 2]), 6),
            }
            for j, name in enumerate(cent_names):
                row[f"dist_to_{name}"] = round(float(all_dists[i, j]), 6)
            synth_rows.append(row)
        out_csv = output_dir / "synth_nearest_centroid.csv"
        pd.DataFrame(synth_rows).to_csv(out_csv, index=False)
        log.info("Synth→centroid distances CSV → %s", out_csv)

    for with_synth in (False, True):
        if with_synth and Z_synth is None:
            continue

        traces = []
        for sc in sc_list:
            for mod in modalities:
                mask = (
                    (meta_orig_r["base_mod"].values == mod)
                    & (meta_orig_r["scanner"].values == sc)
                )
                if not mask.any():
                    continue
                idx = np.where(mask)[0]
                traces.append(go.Scatter3d(
                    x=Z_orig[mask, 0], y=Z_orig[mask, 1], z=Z_orig[mask, 2],
                    mode="markers",
                    marker=dict(size=3.5, color=MODALITY_PALETTE.get(mod, "#888"),
                                symbol=sym_map[sc], opacity=0.88),
                    name=f"{mod} / {sc}",
                    legendgroup=mod,
                    text=[hover_orig[i] for i in idx],
                    customdata=[[paths_orig[i]] for i in idx],
                    hoverinfo="text",
                ))
        n_orig_traces = len(traces)

        if with_synth and Z_synth is not None and meta_synth is not None:
            hover_s, paths_s = _hover_and_path(meta_synth.reset_index(drop=True))
            traces.append(go.Scatter3d(
                x=Z_synth[:, 0], y=Z_synth[:, 1], z=Z_synth[:, 2],
                mode="markers",
                marker=dict(size=2, color=SYNTH_COLOR, opacity=0.15),
                name="synthetic",
                text=hover_s,
                customdata=[[p] for p in paths_s],
                hoverinfo="text",
            ))

            # ── Centroid + top-3 nearest synth traces ─────────────────────────
            clusters_ordered = []
            for mod in sorted(nearest_synth.keys()):
                color    = MODALITY_PALETTE.get(mod, "#888")
                mask_mod = (meta_orig_r["base_mod"].values == mod)
                centroid = Z_orig[mask_mod].mean(axis=0)
                traces.append(go.Scatter3d(
                    x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
                    mode="markers",
                    marker=dict(size=10, color=color, symbol="cross",
                                opacity=1.0, line=dict(color="white", width=2)),
                    name=f"⊕ {mod} centroid",
                    legendgroup=f"centroid_{mod}",
                    hovertext=f"<b>Centroid: {mod}</b>",
                    hoverinfo="text",
                    showlegend=True,
                ))
                top_pairs = nearest_synth[mod]
                top_x = [Z_synth[i, 0] for i, _ in top_pairs]
                top_y = [Z_synth[i, 1] for i, _ in top_pairs]
                top_z = [Z_synth[i, 2] for i, _ in top_pairs]
                top_hover, top_paths = [], []
                meta_s = meta_synth.reset_index(drop=True)
                for rank, (i, d) in enumerate(top_pairs, 1):
                    path  = str(meta_s.iloc[i].get("image_path", "") or "")
                    fname = Path(path).name if path else "—"
                    top_hover.append(
                        f"<b>#{rank} nearest → {mod}</b><br>"
                        f"dist: {d:.4f}<br>"
                        f"file: {fname}<br>"
                        f"<i>click to copy path</i>"
                    )
                    top_paths.append(path or "none")
                traces.append(go.Scatter3d(
                    x=top_x, y=top_y, z=top_z,
                    mode="markers",
                    marker=dict(size=9, color=color, symbol="diamond",
                                opacity=1.0, line=dict(color="black", width=1.5)),
                    name=f"top-3 → {mod}",
                    legendgroup=f"top3_{mod}",
                    text=top_hover,
                    customdata=[[p] for p in top_paths],
                    hoverinfo="text",
                    visible=False,
                    showlegend=True,
                ))
                clusters_ordered.append(mod)

            n_clusters = len(clusters_ordered)
            vis_all = (
                [True] * n_orig_traces + [True] +
                [True, False] * n_clusters
            )
            vis_top3 = (
                [True] * n_orig_traces + [False] +
                [True, True] * n_clusters
            )
            updatemenus = [dict(
                type="buttons", direction="left",
                x=0.0, xanchor="left", y=1.13, yanchor="top",
                showactive=True, font=dict(size=12),
                buttons=[
                    dict(label="⬛ All synthetic", method="update",
                         args=[{"visible": vis_all}]),
                    dict(label="💎 Top-3 per cluster", method="update",
                         args=[{"visible": vis_top3}]),
                ],
            )]
        else:
            updatemenus = []

        fig = go.Figure(data=traces)
        fig.update_layout(
            title="LDA — " + ("original + synthetic" if with_synth else "original contrasts only"),
            scene=dict(
                xaxis_title=f"LD1{_pct(0)}",
                yaxis_title=f"LD2{_pct(1)}",
                zaxis_title=f"LD3{_pct(2)}",
            ),
            legend=dict(font=dict(size=9)),
            margin=dict(l=0, r=0, b=0, t=60),
            updatemenus=updatemenus,
        )
        stem = "lda_3d_with_synthetic" if with_synth else "lda_3d_original"
        html_path = output_dir / f"{stem}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn",
                       post_script=_CLICK_TO_COPY_JS)
        log.info("Saved → %s", html_path)


def plot_lda_loadings(
    lda: LinearDiscriminantAnalysis,
    feat_names: list[str],
    output_dir: Path,
    n_top: int = 20,
    n_ld: int = 4,
    stem: str = "lda_loadings",
) -> None:
    """Horizontal bar chart of top feature loadings per discriminant axis."""
    n_ld  = min(n_ld, lda.scalings_.shape[1])
    scalings = lda.scalings_  # (n_features, n_components)

    fig, axes = plt.subplots(1, n_ld, figsize=(5.5 * n_ld, max(7, n_top * 0.45)),
                              sharey=False)
    if n_ld == 1:
        axes = [axes]

    for ax, ld_idx in zip(axes, range(n_ld)):
        vals  = scalings[:, ld_idx]
        order = np.argsort(np.abs(vals))[::-1][:n_top]
        top_v = vals[order]
        top_n = [feat_names[i] for i in order]
        colors = [FAMILY_COLORS.get(get_family(n), "#888") for n in top_n]
        labels = [short_name(n) for n in top_n]

        y = np.arange(len(top_v))
        ax.barh(y, top_v, color=colors, edgecolor="none", height=0.7)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Coefficient")
        ax.set_title(f"LD{ld_idx+1}", fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()
               if any(get_family(n) == f for n in feat_names)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=7, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"LDA discriminant loadings — top {n_top} features", fontsize=11)
    fig.tight_layout()
    _save(fig, output_dir, stem)
    log.info("LDA loadings plot saved.")


# ── Univariate F-score ────────────────────────────────────────────────────────

def compute_fscores(X: np.ndarray, y: np.ndarray, feat_names: list[str]) -> pd.DataFrame:
    """One-way ANOVA F-score per feature; higher = better modality separator."""
    groups = [X[y == c] for c in np.unique(y)]
    fscores, pvals = [], []
    for j in range(X.shape[1]):
        cols = [g[:, j] for g in groups]
        f, p = stats.f_oneway(*cols)
        fscores.append(f); pvals.append(p)
    df = pd.DataFrame({"feature": feat_names, "F": fscores, "p": pvals})
    df["family"] = df["feature"].apply(get_family)
    return df.sort_values("F", ascending=False).reset_index(drop=True)


def plot_fscores(df_f: pd.DataFrame, output_dir: Path, n_top: int = 30) -> None:
    top = df_f.head(n_top)
    colors = [FAMILY_COLORS.get(f, "#888") for f in top["family"]]
    labels = [short_name(n) for n in top["feature"]]

    fig, ax = plt.subplots(figsize=(9, max(6, n_top * 0.35)))
    y = np.arange(len(top))
    ax.barh(y, top["F"], color=colors, edgecolor="none", height=0.75)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("ANOVA F-score  (higher = better modality separator)")
    ax.set_title(f"Top {n_top} features by one-way ANOVA across modalities (original only)",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()
               if f in top["family"].values]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)
    fig.tight_layout()
    _save(fig, output_dir, "feature_fscore")
    log.info("F-score ranking plot saved.")


# ── Top-feature scatter grid ──────────────────────────────────────────────────

def plot_feature_grid(
    X_orig: np.ndarray,
    labels_orig: np.ndarray,
    le: LabelEncoder,
    feat_names: list[str],
    top_features: list[str],
    output_dir: Path,
    X_synth: np.ndarray | None = None,
    n_pairs: int = 6,
    stem: str = "feature_scatter",
) -> None:
    """
    Grid of 2D scatter plots for the top n_pairs feature pairs (consecutive top features).
    Each cell: feature_i vs feature_j, original coloured by modality, synthetic in black.
    """
    k = min(len(top_features), n_pairs + 1)
    top_features = top_features[:k]
    # Use consecutive pairs: (0,1), (0,2), (1,2), (0,3), (1,3), (2,3) — upper triangle
    pairs = [(i, j) for i in range(k) for j in range(i+1, k)][:n_pairs]
    if not pairs:
        return

    ncols = min(3, len(pairs))
    nrows = (len(pairs) + ncols - 1) // ncols
    modalities = [le.classes_[i] for i in np.unique(labels_orig)]

    for with_synth in (False, True):
        if with_synth and X_synth is None:
            continue
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows),
                                  squeeze=False)
        for ax_idx, (fi, fj) in enumerate(pairs):
            ax = axes[ax_idx // ncols][ax_idx % ncols]
            fn_i = top_features[fi]
            fn_j = top_features[fj]
            col_i = feat_names.index(fn_i)
            col_j = feat_names.index(fn_j)

            if with_synth:
                ax.scatter(X_synth[:, col_i], X_synth[:, col_j],
                           c=SYNTH_COLOR, alpha=SYNTH_ALPHA, s=8, linewidths=0, zorder=1)

            for mod in modalities:
                mask = le.transform([mod])[0] == labels_orig
                ax.scatter(X_orig[mask, col_i], X_orig[mask, col_j],
                           c=MODALITY_PALETTE.get(mod, "#888"),
                           alpha=0.80, s=20, linewidths=0, zorder=2)
                cx = X_orig[mask, col_i].mean()
                cy = X_orig[mask, col_j].mean()
                ax.annotate(mod, (cx, cy), fontsize=8, fontweight="bold",
                            color=MODALITY_PALETTE.get(mod, "#333"),
                            ha="center", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.5, lw=0))

            ax.set_xlabel(short_name(fn_i), fontsize=8)
            ax.set_ylabel(short_name(fn_j), fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)

        # Hide unused subplots
        for ax_idx in range(len(pairs), nrows * ncols):
            axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

        mod_h = [mpatches.Patch(color=MODALITY_PALETTE.get(m, "#888"), label=m)
                 for m in modalities]
        if with_synth:
            mod_h += [mpatches.Patch(color=SYNTH_COLOR, alpha=0.5, label="synthetic")]
        fig.legend(handles=mod_h, loc="lower center", ncol=len(mod_h),
                   fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))
        title = "Top individual features — " + ("original + synthetic" if with_synth else "original only")
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        suffix = "_with_synthetic" if with_synth else "_original"
        _save(fig, output_dir, stem + suffix)
    log.info("Feature scatter grid saved.")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--original_csv", type=Path,
        default=Path("analysis/contrast_manifold/outputs/data/original/roi_mask/"
                     "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"),
    )
    p.add_argument(
        "--synthetic_csv", type=Path,
        default=Path("analysis/contrast_manifold/outputs/data/synthetic_v19_c/roi_mask/"
                     "synthetic_v19_c_features_normalized_feat_selected.csv"),
    )
    p.add_argument(
        "--output_dir", type=Path,
        default=Path("analysis/contrast_manifold/outputs/plots/contrast_clustering_v19_c"),
    )
    p.add_argument("--n_top_scatter", type=int, default=5,
                   help="Number of top F-score features used in pairwise scatter grid.")
    p.add_argument("--n_top_lda", type=int, default=20,
                   help="Number of top features shown in LDA loading bars.")
    p.add_argument("--n_ld", type=int, default=4,
                   help="Number of LDA discriminant axes to show.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_orig  = load_csv(args.original_csv)
    df_synth = load_csv(args.synthetic_csv)

    # ── Align feature columns ─────────────────────────────────────────────
    fcols_orig  = set(feature_cols(df_orig))
    fcols_synth = set(feature_cols(df_synth))
    common = sorted(fcols_orig & fcols_synth)
    log.info("Common features: %d", len(common))

    X_orig  = df_orig[common].to_numpy(dtype=np.float64)
    X_synth = df_synth[common].to_numpy(dtype=np.float64)

    # Impute NaNs with column mean (fit on original)
    col_means = np.nanmean(X_orig, axis=0)
    for arr in (X_orig, X_synth):
        nan_mask = np.isnan(arr)
        arr[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Drop zero-variance columns (on original)
    var_mask = X_orig.var(axis=0) > 0
    X_orig   = X_orig[:, var_mask]
    X_synth  = X_synth[:, var_mask]
    common   = [c for c, k in zip(common, var_mask) if k]
    log.info("After dropping zero-var cols: %d features", len(common))

    df_orig["base_mod"] = df_orig["modality"].apply(base_mod)
    le = LabelEncoder()
    y_orig = le.fit_transform(df_orig["base_mod"].values)
    scanners = df_orig["scanner"].values if "scanner" in df_orig.columns else np.array(["unknown"] * len(df_orig))
    log.info("Modality classes: %s", list(le.classes_))

    # ── 1. ANOVA F-score ranking ──────────────────────────────────────────
    log.info("Computing ANOVA F-scores …")
    df_f = compute_fscores(X_orig, y_orig, common)
    df_f.to_csv(args.output_dir / "feature_fscores.csv", index=False)
    log.info("Top 10 features by F-score:\n%s", df_f.head(10)[["feature","F","family"]].to_string(index=False))
    plot_fscores(df_f, args.output_dir)

    # ── 2. LDA ────────────────────────────────────────────────────────────
    log.info("Fitting LDA …")
    n_classes = len(le.classes_)
    n_ld = min(args.n_ld, n_classes - 1)
    lda = fit_lda(X_orig, y_orig, n_components=n_ld)

    Z_orig  = lda.transform(X_orig)
    Z_synth = lda.transform(X_synth)
    explained = lda.explained_variance_ratio_
    log.info("LDA explained variance ratio: %s", np.round(explained, 3))

    # Save LDA coordinates
    df_lda_orig  = pd.concat([
        df_orig[["subject","session","modality","scanner"]].reset_index(drop=True),
        pd.DataFrame(Z_orig,  columns=[f"LD{i+1}" for i in range(n_ld)]),
    ], axis=1)
    df_lda_orig["source"] = "original"

    df_lda_synth = pd.concat([
        df_synth[["subject","session","modality","scanner"]].reset_index(drop=True)
        if all(c in df_synth.columns for c in ["subject","session","modality","scanner"])
        else pd.DataFrame(index=range(len(Z_synth))),
        pd.DataFrame(Z_synth, columns=[f"LD{i+1}" for i in range(n_ld)]),
    ], axis=1)
    df_lda_synth["source"] = "synthetic"

    pd.concat([df_lda_orig, df_lda_synth], ignore_index=True).to_csv(
        args.output_dir / "lda_coords.csv", index=False
    )

    meta_orig_df = df_orig[
        [c for c in ["subject","session","modality","scanner","base_mod","image_path"] if c in df_orig.columns]
    ].reset_index(drop=True)
    meta_synth_df = df_synth[
        [c for c in ["subject","session","modality","scanner","image_path"] if c in df_synth.columns]
    ].reset_index(drop=True)
    # base_mod needed for 3D scatter grouping
    if "base_mod" not in meta_synth_df.columns and "modality" in meta_synth_df.columns:
        meta_synth_df["base_mod"] = meta_synth_df["modality"].apply(base_mod)

    plot_lda_scatter(Z_orig, y_orig, le, scanners, args.output_dir,
                     Z_synth=Z_synth, explained=explained)
    plot_lda_scatter_3d(Z_orig, y_orig, le, meta_orig_df, args.output_dir,
                        Z_synth=Z_synth, meta_synth=meta_synth_df, explained=explained)
    plot_lda_loadings(lda, common, args.output_dir,
                      n_top=args.n_top_lda, n_ld=n_ld)

    # ── 3. Top-feature pairwise scatters ──────────────────────────────────
    top_feats = df_f["feature"].tolist()[:args.n_top_scatter]
    log.info("Top %d features for scatter grid: %s", len(top_feats), top_feats)
    plot_feature_grid(
        X_orig, y_orig, le, common, top_feats, args.output_dir,
        X_synth=X_synth, n_pairs=10, stem="feature_scatter_top5",
    )

    # ── 4. Scanner × modality LDA ─────────────────────────────────────────
    # Fits LDA on combined (vendor/base_mod) labels so scanner subclusters
    # become the discriminant targets rather than noise within modality clusters.
    vendor_col = "vendor" if "vendor" in df_orig.columns else "scanner"
    if vendor_col in df_orig.columns:
        log.info("Fitting scanner×modality LDA …")
        df_orig["sc_mod_label"] = (
            df_orig[vendor_col].fillna("unk").astype(str) + "/"
            + df_orig["base_mod"].astype(str)
        )
        le_sc = LabelEncoder()
        y_sc  = le_sc.fit_transform(df_orig["sc_mod_label"].values)
        n_sc_classes = len(le_sc.classes_)
        n_ld_sc      = min(args.n_ld, n_sc_classes - 1, X_orig.shape[0] - 1)
        log.info("Scanner×modality classes (%d): %s", n_sc_classes, list(le_sc.classes_))

        # Build a palette mapping each combined label → base modality color
        def _sc_mod_color(label: str) -> str:
            mod = label.split("/")[-1] if "/" in label else label
            return MODALITY_PALETTE.get(mod, "#888")

        lda_sc  = fit_lda(X_orig, y_sc, n_components=n_ld_sc)
        Z_sc_orig  = lda_sc.transform(X_orig)
        Z_sc_synth = lda_sc.transform(X_synth)
        explained_sc = lda_sc.explained_variance_ratio_

        # Scatter: colour = base modality, marker = vendor (reuse existing scanner info)
        for with_synth in (False, True):
            fig, ax = plt.subplots(figsize=(11, 8))
            if with_synth:
                ax.scatter(Z_sc_synth[:, 0], Z_sc_synth[:, 1],
                           c=SYNTH_COLOR, alpha=SYNTH_ALPHA, s=8, linewidths=0, zorder=1)
            sc_list_sc = sorted(df_orig[vendor_col].dropna().unique())
            sc_marker  = {sc: MPL_MARKERS[i % len(MPL_MARKERS)] for i, sc in enumerate(sc_list_sc)}
            for cl in le_sc.classes_:
                vendor, mod = (cl.split("/", 1) + ["?"])[:2]
                mask = df_orig["sc_mod_label"].values == cl
                if not mask.any():
                    continue
                color = _sc_mod_color(cl)
                marker = sc_marker.get(vendor, "o")
                ax.scatter(Z_sc_orig[mask, 0], Z_sc_orig[mask, 1],
                           c=color, marker=marker, alpha=0.85, s=28, linewidths=0, zorder=2)
            # Cluster centroids with (vendor/mod) label
            for cl in le_sc.classes_:
                mask = df_orig["sc_mod_label"].values == cl
                if mask.sum() < 2:
                    continue
                cx, cy = Z_sc_orig[mask, 0].mean(), Z_sc_orig[mask, 1].mean()
                ax.annotate(cl, (cx, cy), fontsize=7, fontweight="bold",
                            color=_sc_mod_color(cl), ha="center", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.55, lw=0))
            # Legend: modality color patches + vendor markers
            mods_in = sorted({cl.split("/")[-1] for cl in le_sc.classes_})
            mod_h = [mpatches.Patch(color=MODALITY_PALETTE.get(m, "#888"), label=m) for m in mods_in]
            sc_h  = [plt.Line2D([0], [0], marker=sc_marker[sc], color="gray",
                                linestyle="none", markersize=6, label=sc) for sc in sc_list_sc]
            handles = mod_h + sc_h
            if with_synth:
                handles += [mpatches.Patch(color=SYNTH_COLOR, alpha=0.5, label="synthetic")]
            ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.75, ncol=2)
            pct0 = f"{explained_sc[0]*100:.1f}%" if len(explained_sc) > 0 else ""
            pct1 = f"{explained_sc[1]*100:.1f}%" if len(explained_sc) > 1 else ""
            ax.set_xlabel(f"LD1 ({pct0})"); ax.set_ylabel(f"LD2 ({pct1})")
            title = "LDA (scanner×modality) — " + ("original + synthetic" if with_synth else "original only")
            ax.set_title(title, fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            stem = "lda_sc_mod_with_synthetic" if with_synth else "lda_sc_mod_original"
            _save(fig, args.output_dir, stem)

        # F-scores under scanner×modality labels
        df_f_sc = compute_fscores(X_orig, y_sc, common)
        df_f_sc.to_csv(args.output_dir / "feature_fscores_sc_mod.csv", index=False)
        log.info("Top 10 features (scanner×modality F-score):\n%s",
                 df_f_sc.head(10)[["feature", "F", "family"]].to_string(index=False))
        plot_lda_loadings(lda_sc, common, args.output_dir,
                          n_top=args.n_top_lda, n_ld=n_ld_sc,
                          stem="lda_sc_mod_loadings")

    log.info("All outputs → %s", args.output_dir)


if __name__ == "__main__":
    main()
