#!/usr/bin/env python
"""
Pairwise PCA plots: T1w + one target contrast + synthetic data.

For each target contrast, fits PCA on (T1w + target) original rows so that
PC1 captures the dominant axis of variation *between those two contrasts*.
Synthetic data is then projected onto those axes.

Produces per-pair:
  pca_pairwise_T1w_vs_{contrast}_2d.{png,pdf}
  pca_pairwise_T1w_vs_{contrast}_3d.html   (interactive, click to copy path)

Usage:
  run_job --gpus 0 --slot 0 --wait -- .venv/bin/python analysis/contrast_manifold/scripts/plot_pca_pairwise.py \\
      --original_csv  analysis/contrast_manifold/outputs/data/original/roi_mask/on_harmony_features_normalized_combined_downsampled100.csv \\
      --synthetic_csv analysis/contrast_manifold/outputs/data/synthetic_v19/roi_mask/synthetic_v19_features_normalized_combined.csv \\
      --output_dir    analysis/contrast_manifold/outputs/plots/v19/v19_c_r1/synthseg_mask_31/pca
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

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

PLOTLY_SYMS = ["circle", "square", "diamond", "cross", "x",
               "circle-open", "square-open", "diamond-open"]

SYNTH_COLOR = "#111111"
SYNTH_ALPHA = 0.20

_CLICK_JS = """
(function() {
    var gd = document.querySelector('.plotly-graph-div');
    gd.on('plotly_click', function(data) {
        var pt = data.points[0];
        var cd = pt.customdata;
        if (!cd) return;
        var text = Array.isArray(cd) ? cd[0] : cd;
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
            toast.textContent = pt.text ? pt.text.replace(/<[^>]+>/g,' ') : 'No path';
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


# ── Data helpers ──────────────────────────────────────────────────────────────

def base_mod(m: str) -> str:
    return m.split("_")[0] if isinstance(m, str) else "unknown"


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in META_COLS and not c.startswith("diagnostics_")]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    renames = {
        "modality_id": "modality",
        "scanner_model": "scanner",
        "scanner_vendor": "vendor",
        "cohort_category": "cohort",
    }
    df = df.rename(columns={k: v for k, v in renames.items()
                             if k in df.columns and v not in df.columns})
    for col in ["subject", "session", "modality", "scanner", "vendor", "cohort"]:
        if col not in df.columns:
            df[col] = "unknown"
    return df


def build_aligned_matrix(
    df_a: pd.DataFrame, df_b: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    fa = set(feature_cols(df_a))
    fb = set(feature_cols(df_b))
    common = sorted(fa & fb)
    Xa = df_a[common].to_numpy(dtype=np.float64)
    Xb = df_b[common].to_numpy(dtype=np.float64)
    X_all = np.vstack([Xa, Xb])
    # drop all-NaN then zero-variance
    nan_mask = np.isnan(X_all).mean(axis=0) < 1.0
    X_all = X_all[:, nan_mask]
    common = [c for c, k in zip(common, nan_mask) if k]
    imp = SimpleImputer(strategy="mean")
    X_all = imp.fit_transform(X_all)
    var_mask = X_all.var(axis=0) > 0
    X_all = X_all[:, var_mask]
    common = [c for c, k in zip(common, var_mask) if k]
    na = len(df_a)
    return X_all[:na], X_all[na:], common


def hover_text(row: pd.Series) -> tuple[str, str]:
    path = str(row.get("image_path", "") or "")
    fname = Path(path).name if path else "—"
    txt = (
        f"<b>{row.get('subject','?')}</b> {row.get('session','?')}<br>"
        f"modality: {row.get('modality','?')}<br>"
        f"scanner:  {row.get('scanner','?')}<br>"
        f"file:     {fname}<br>"
        f"<i>{'click to copy path' if path else 'no path'}</i>"
    )
    return txt, (path or "none")


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _save_mpl(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        p = output_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", p)
    plt.close(fig)


def plot_2d(
    X_t1w: np.ndarray, meta_t1w: pd.DataFrame,
    X_tgt: np.ndarray, meta_tgt: pd.DataFrame,
    X_synth: np.ndarray,
    pca: PCA,
    target_label: str,
    synth_label: str,
    output_dir: Path,
    stem: str,
) -> None:
    ev = pca.explained_variance_ratio_ * 100
    scanners = sorted(pd.concat([meta_t1w, meta_tgt])["scanner"].dropna().unique())
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    sym = {sc: markers[i % len(markers)] for i, sc in enumerate(scanners)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pairs = [(0, 1), (0, 2)]

    for ax, (xi, yi) in zip(axes, pairs):
        ax.scatter(X_synth[:, xi], X_synth[:, yi],
                   c=SYNTH_COLOR, alpha=SYNTH_ALPHA, s=8, zorder=1, label=synth_label)
        c_t1w = MODALITY_PALETTE["T1w"]
        c_tgt = MODALITY_PALETTE.get(base_mod(target_label), "#888")
        for sc in scanners:
            m_t1w = meta_t1w["scanner"].values == sc
            if m_t1w.any():
                ax.scatter(X_t1w[m_t1w, xi], X_t1w[m_t1w, yi],
                           c=c_t1w, marker=sym[sc], s=30, alpha=0.85,
                           zorder=3, label=f"T1w/{sc}" if xi == 0 else None)
            m_tgt = meta_tgt["scanner"].values == sc
            if m_tgt.any():
                ax.scatter(X_tgt[m_tgt, xi], X_tgt[m_tgt, yi],
                           c=c_tgt, marker=sym[sc], s=30, alpha=0.85,
                           zorder=3, label=f"{target_label}/{sc}" if xi == 0 else None)
        ax.set_xlabel(f"PC{xi+1} ({ev[xi]:.1f}%)")
        ax.set_ylabel(f"PC{yi+1} ({ev[yi]:.1f}%)")
        ax.set_title(f"T1w + {target_label} + {synth_label}  PC{xi+1}×PC{yi+1}")

    handles = [
        mpatches.Patch(color=MODALITY_PALETTE["T1w"], label="T1w"),
        mpatches.Patch(color=MODALITY_PALETTE.get(base_mod(target_label), "#888"),
                       label=target_label),
        mpatches.Patch(color=SYNTH_COLOR, alpha=SYNTH_ALPHA, label=synth_label),
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="best")
    fig.tight_layout()
    _save_mpl(fig, output_dir, stem)


def plot_3d(
    X_t1w: np.ndarray, meta_t1w: pd.DataFrame,
    X_tgt: np.ndarray, meta_tgt: pd.DataFrame,
    X_synth: np.ndarray, meta_synth: pd.DataFrame,
    pca: PCA,
    target_label: str,
    synth_label: str,
    output_dir: Path,
    stem: str,
) -> None:
    ev = pca.explained_variance_ratio_ * 100
    scanners = sorted(pd.concat([meta_t1w, meta_tgt])["scanner"].dropna().unique())
    sym = {sc: PLOTLY_SYMS[i % len(PLOTLY_SYMS)] for i, sc in enumerate(scanners)}

    traces: list[go.Scatter3d] = []

    # Synthetic background
    _hp_s = [hover_text(r) for _, r in meta_synth.iterrows()]
    h_s, p_s = [x[0] for x in _hp_s], [x[1] for x in _hp_s]
    traces.append(go.Scatter3d(
        x=X_synth[:, 0], y=X_synth[:, 1], z=X_synth[:, 2],
        mode="markers",
        marker=dict(size=2.5, color=SYNTH_COLOR, opacity=SYNTH_ALPHA),
        name=synth_label, text=h_s, customdata=[[p] for p in p_s], hoverinfo="text",
    ))

    # T1w
    c_t1w = MODALITY_PALETTE["T1w"]
    for sc in scanners:
        mask = meta_t1w["scanner"].values == sc
        if not mask.any():
            continue
        sub = meta_t1w[mask].reset_index(drop=True)
        _hp = [hover_text(r) for _, r in sub.iterrows()]
        ht, pt = [x[0] for x in _hp], [x[1] for x in _hp]
        traces.append(go.Scatter3d(
            x=X_t1w[mask, 0], y=X_t1w[mask, 1], z=X_t1w[mask, 2],
            mode="markers",
            marker=dict(size=4, color=c_t1w, symbol=sym.get(sc, "circle"), opacity=0.9),
            name=f"T1w / {sc}", legendgroup="T1w",
            text=ht, customdata=[[p] for p in pt], hoverinfo="text",
        ))

    # Target contrast
    c_tgt = MODALITY_PALETTE.get(base_mod(target_label), "#888888")
    for sc in scanners:
        mask = meta_tgt["scanner"].values == sc
        if not mask.any():
            continue
        sub = meta_tgt[mask].reset_index(drop=True)
        _hp = [hover_text(r) for _, r in sub.iterrows()]
        ht, pt = [x[0] for x in _hp], [x[1] for x in _hp]
        traces.append(go.Scatter3d(
            x=X_tgt[mask, 0], y=X_tgt[mask, 1], z=X_tgt[mask, 2],
            mode="markers",
            marker=dict(size=4, color=c_tgt, symbol=sym.get(sc, "circle"), opacity=0.9),
            name=f"{target_label} / {sc}", legendgroup=target_label,
            text=ht, customdata=[[p] for p in pt], hoverinfo="text",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"PCA (pairwise fit) — T1w + {target_label} + {synth_label}",
        scene=dict(
            xaxis_title=f"PC1 ({ev[0]:.1f}%)",
            yaxis_title=f"PC2 ({ev[1]:.1f}%)",
            zaxis_title=f"PC3 ({ev[2]:.1f}%)",
        ),
        legend=dict(font=dict(size=9)),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    out = output_dir / f"{stem}.html"
    fig.write_html(str(out), include_plotlyjs="cdn", post_script=_CLICK_JS)
    log.info("Saved → %s", out)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original_csv",  type=Path, required=True)
    p.add_argument("--synthetic_csv", type=Path, required=True)
    p.add_argument("--output_dir",    type=Path, required=True)
    p.add_argument(
        "--targets", nargs="*", default=None,
        help="Base modality groups to pair with T1w (default: all). "
             "E.g. --targets T2w FLAIR bold",
    )
    p.add_argument(
        "--n_components", type=int, default=3,
        help="Number of PCA components (default 3; must be ≥ 3 for 3-D plot)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for p in (args.original_csv, args.synthetic_csv):
        if not p.exists():
            raise FileNotFoundError(p)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_orig  = load_csv(args.original_csv)
    df_synth = load_csv(args.synthetic_csv)

    # Derive synth label from path
    synth_label = (
        df_synth["image_path"].dropna()
        .str.extract(r"(synthetic_[^/]+)")[0]
        .dropna()
        .iloc[0]
        if not df_synth["image_path"].dropna().empty
        else "synthetic"
    )

    orig_base_mod  = df_orig["modality"].apply(base_mod)
    synth_base_mod = df_synth["modality"].apply(base_mod)

    df_t1w_full = df_orig[orig_base_mod == "T1w"].copy()
    if df_t1w_full.empty:
        raise ValueError("No T1w rows found in original CSV")

    groups = sorted(g for g in orig_base_mod.unique() if g != "T1w")
    targets = args.targets if args.targets else groups
    log.info("Target contrasts: %s", targets)

    # Build synth feature matrix once (aligned against all originals)
    X_orig_all, X_synth_all, common_all = build_aligned_matrix(df_orig, df_synth)
    meta_synth = df_synth[[c for c in ["subject","session","modality","scanner",
                                        "vendor","cohort","image_path"] if c in df_synth.columns]].copy()

    for tgt in targets:
        df_tgt = df_orig[orig_base_mod == tgt].copy()
        if df_tgt.empty:
            log.warning("No rows for '%s', skipping", tgt)
            continue

        log.info("Fitting PCA on T1w (%d) + %s (%d) …", len(df_t1w_full), tgt, len(df_tgt))

        # Slice the pre-aligned matrices to T1w + target rows
        t1w_idx  = df_orig.index[orig_base_mod == "T1w"]
        tgt_idx  = df_orig.index[orig_base_mod == tgt]
        pair_idx = np.concatenate([
            np.where(df_orig.index.isin(t1w_idx))[0],
            np.where(df_orig.index.isin(tgt_idx))[0],
        ])
        n_t1w = len(t1w_idx)

        X_pair = X_orig_all[pair_idx]
        # Re-impute + drop zero-variance within this pair
        imp = SimpleImputer(strategy="mean")
        X_pair_imp = imp.fit_transform(X_pair)
        # Impute synth while imp still sees full feature count
        X_synth_imp = imp.transform(X_synth_all)
        var_mask = X_pair_imp.var(axis=0) > 0
        X_pair = X_pair_imp[:, var_mask]

        pca = PCA(n_components=min(args.n_components, X_pair.shape[1], X_pair.shape[0]))
        pca.fit(X_pair)
        ev = pca.explained_variance_ratio_ * 100
        log.info("  PC1=%.1f%%  PC2=%.1f%%  PC3=%.1f%%", ev[0], ev[1], ev[2] if len(ev)>2 else 0)

        X_t1w_pca = pca.transform(X_pair[:n_t1w])
        X_tgt_pca = pca.transform(X_pair[n_t1w:])

        # Project synthetic onto the same pairwise PCA axes
        X_synth_pca = pca.transform(X_synth_imp[:, var_mask])

        meta_t1w = df_t1w_full.reset_index(drop=True)
        meta_tgt = df_tgt.reset_index(drop=True)

        stem = f"pca_pairwise_T1w_vs_{tgt}"
        plot_2d(X_t1w_pca, meta_t1w, X_tgt_pca, meta_tgt, X_synth_pca,
                pca, tgt, synth_label, args.output_dir, stem + "_2d")
        plot_3d(X_t1w_pca, meta_t1w, X_tgt_pca, meta_tgt, X_synth_pca, meta_synth,
                pca, tgt, synth_label, args.output_dir, stem + "_3d")

    log.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
