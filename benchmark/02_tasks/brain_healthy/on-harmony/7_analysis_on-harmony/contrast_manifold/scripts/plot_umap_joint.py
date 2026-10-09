#!/usr/bin/env python
"""
Joint UMAP of original (downsampled + normalized) and synthetic_v19 (normalized) features.

Both CSVs must already be Z-score normalized via normalize_original.py /
normalize_synthetic.py so that feature scales are consistent.

Pipeline:
  1. Load and combine both CSVs.
  2. Drop all-NaN and zero-variance columns (residual after per-dataset normalization).
  3. Impute any remaining NaNs with the column mean.
  4. PCA to 95% explained variance (min 3 components).
  5. Fit UMAP jointly on original + synthetic (co-embedding).
  6. Plot 2-D and 3-D: original colored by modality (symbol=scanner), synthetic in black.

Outputs (in --output_dir):
  umap_joint_2d.{pdf,png}
  umap_joint_3d.html  (interactive Plotly)
  umap_joint_coords.csv

Usage:
  run_job --gpus 0 --slot 0 --wait -- .venv/bin/python \\
    analysis/contrast_manifold/scripts/plot_umap_joint.py \\
    --original_csv analysis/contrast_manifold/outputs/data/original/roi_mask/on_harmony_features_normalized_downsampled100.csv \\
    --synthetic_csv analysis/contrast_manifold/outputs/data/synthetic_v19/roi_mask/synthetic_v19_features_normalized.csv \\
    --output_dir analysis/contrast_manifold/outputs/plots/umap_joint_v19
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import yaml
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
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
import umap
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

MPL_MARKERS  = ["o", "s", "^", "D", "v", "P", "X", "*", "h", ">"]
PLOTLY_SYMS  = ["circle", "square", "diamond", "cross", "x",
                "circle-open", "square-open", "diamond-open"]

SYNTH_COLOR = "#111111"
SYNTH_ALPHA = 0.20
SYNTH_LABEL = "synthetic_v19"

_CLICK_TO_COPY_JS = """
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
            // Off-screen (not hidden) so Safari's execCommand('copy') can select it.
            var el = document.createElement('textarea');
            el.value = text;
            el.setAttribute('readonly', '');
            el.style.cssText = 'position:absolute;left:-9999px;top:-9999px;font-size:1px;';
            document.body.appendChild(el);
            el.select();
            el.setSelectionRange(0, el.value.length);
            var syncOk = document.execCommand('copy');
            document.body.removeChild(el);
            if (syncOk) {
                toast.textContent = '\\u2713 Copied: ' + text;
            } else if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(function() {
                    toast.textContent = '\\u2713 Copied: ' + text;
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

_LASSO_REMOVE_JS = """
(function() {
    var gd = document.querySelector('.plotly-graph-div');
    var _origData = null;   // saved before first removal
    var _selection = null;  // {curveIndex: [pointIndices]}

    // ── Buttons ───────────────────────────────────────────────────────────────
    var wrap = document.createElement('div');
    wrap.style.cssText = [
        'position:fixed','top:16px','right:16px',
        'display:flex','gap:8px','z-index:9999'
    ].join(';');

    function mkBtn(label, bg, handler) {
        var b = document.createElement('button');
        b.textContent = label;
        b.style.cssText = [
            'padding:6px 14px','border:none','border-radius:4px',
            'font:13px/1.4 sans-serif','cursor:pointer',
            'background:' + bg,'color:#fff','display:none',
            'box-shadow:0 2px 6px rgba(0,0,0,.35)'
        ].join(';');
        b.onclick = handler;
        return b;
    }

    var rmBtn    = mkBtn('\\u232b Remove selected', '#c62828', doRemove);
    var resetBtn = mkBtn('\\u21ba Reset',            '#424242', doReset);
    wrap.appendChild(rmBtn);
    wrap.appendChild(resetBtn);
    document.body.appendChild(wrap);

    // ── Selection events ──────────────────────────────────────────────────────
    gd.on('plotly_selected', function(ev) {
        if (!ev || !ev.points || ev.points.length === 0) {
            rmBtn.style.display = 'none'; _selection = null; return;
        }
        var byCurve = {};
        ev.points.forEach(function(pt) {
            var c = pt.curveNumber;
            if (!byCurve[c]) byCurve[c] = [];
            byCurve[c].push(pt.pointIndex);
        });
        _selection = byCurve;
        rmBtn.style.display = 'inline-block';
    });

    gd.on('plotly_deselect', function() {
        rmBtn.style.display = 'none'; _selection = null;
    });

    // ── Remove ────────────────────────────────────────────────────────────────
    function doRemove() {
        if (!_selection) return;

        // Save original data on the first removal
        if (!_origData) {
            _origData = gd.data.map(function(tr) {
                var copy = {};
                Object.keys(tr).forEach(function(k) {
                    var v = tr[k];
                    copy[k] = Array.isArray(v) ? v.slice() : v;
                });
                return copy;
            });
        }

        var newData = gd.data.map(function(tr, ci) {
            var remove = new Set(_selection[ci] || []);
            if (remove.size === 0) return tr;
            var copy = {};
            Object.keys(tr).forEach(function(k) {
                var v = tr[k];
                copy[k] = Array.isArray(v)
                    ? v.filter(function(_, i) { return !remove.has(i); })
                    : v;
            });
            return copy;
        });

        Plotly.react(gd, newData, gd.layout);
        _selection = null;
        rmBtn.style.display = 'none';
        resetBtn.style.display = 'inline-block';
    }

    // ── Reset ─────────────────────────────────────────────────────────────────
    function doReset() {
        if (!_origData) return;
        Plotly.react(gd, _origData, gd.layout);
        _origData = null;
        resetBtn.style.display = 'none';
    }
})();
"""


# ── Feature selection helpers ─────────────────────────────────────────────────

def get_family(name: str) -> str:
    for fam in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape"):
        if fam in name.lower():
            return fam
    return "other"


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


def base_mod(m: str) -> str:
    return m.split("_")[0] if m else "unknown"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Harmonise column names across both extraction scripts
    renames = {
        "modality_id": "modality",
        "scanner_model": "scanner",
        "scanner_vendor": "vendor",
        "cohort_category": "cohort",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns and v not in df.columns})
    for col in ["subject", "session", "modality", "scanner", "vendor", "cohort"]:
        if col not in df.columns:
            df[col] = "unknown"
    log.info("Loaded %s: %d rows", path.name, len(df))
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and not c.startswith("diagnostics_")]


def build_matrix(df_orig: pd.DataFrame, df_synth: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Combine original and synthetic, align columns, drop all-NaN + zero-var,
    impute.  Returns (X_orig, X_synth, feat_cols).
    """
    cols_orig  = set(feature_cols(df_orig))
    cols_synth = set(feature_cols(df_synth))
    common     = sorted(cols_orig & cols_synth)
    log.info("Common feature columns: %d (orig=%d, synth=%d)", len(common), len(cols_orig), len(cols_synth))

    if not common:
        raise ValueError(
            "No common feature columns between original and synthetic CSVs. "
            "Check that both were produced with the same radiomics config."
        )

    X_orig  = df_orig[common].to_numpy(dtype=np.float64)
    X_synth = df_synth[common].to_numpy(dtype=np.float64)
    X_all   = np.vstack([X_orig, X_synth])

    # Drop all-NaN columns (across combined set)
    nan_mask = np.isnan(X_all).mean(axis=0) < 1.0
    X_all   = X_all[:, nan_mask]
    common  = [c for c, k in zip(common, nan_mask) if k]

    # Impute
    imputer = SimpleImputer(strategy="mean")
    X_all   = imputer.fit_transform(X_all)

    # Drop zero-variance columns
    var_mask = X_all.var(axis=0) > 0
    X_all    = X_all[:, var_mask]
    common   = [c for c, k in zip(common, var_mask) if k]

    n_orig = len(df_orig)
    log.info("Final feature matrix: %d rows × %d cols", len(X_all), len(common))
    return X_all[:n_orig], X_all[n_orig:], common


def run_pca(X_orig: np.ndarray, X_synth: np.ndarray) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Joint PCA: fit on original+synthetic combined."""
    X_all = np.vstack([X_orig, X_synth])
    pca_full = PCA(random_state=42)
    pca_full.fit(X_all)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = max(3, min(int(np.searchsorted(cumvar, 0.95)) + 1, X_all.shape[0] - 1, X_all.shape[1]))
    pca = PCA(n_components=n_comp, random_state=42)
    X_all_pca = pca.fit_transform(X_all)
    log.info("PCA joint: %d components → %.1f%% variance", n_comp, cumvar[n_comp - 1] * 100)
    n_orig = len(X_orig)
    return X_all_pca[:n_orig], X_all_pca[n_orig:], pca


def run_pca_orig_only(X_orig: np.ndarray, X_synth: np.ndarray) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Original-only PCA: fit on original only, project synthetic onto those axes."""
    pca_full = PCA(random_state=42)
    pca_full.fit(X_orig)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = max(3, min(int(np.searchsorted(cumvar, 0.95)) + 1, X_orig.shape[0] - 1, X_orig.shape[1]))
    pca = PCA(n_components=n_comp, random_state=42)
    X_orig_pca = pca.fit_transform(X_orig)
    X_synth_pca = pca.transform(X_synth)
    log.info("PCA original-only: %d components → %.1f%% variance", n_comp, cumvar[n_comp - 1] * 100)
    return X_orig_pca, X_synth_pca, pca


def _nearest_synth_in_space(
    X_orig: np.ndarray,
    meta_orig: pd.DataFrame,
    X_synth: np.ndarray,
    meta_synth: pd.DataFrame,
    top_k: int = 3,
    output_dir: Path | None = None,
) -> dict[str, list[tuple[int, float]]]:
    """
    Per contrast cluster, compute centroid and find top_k nearest synthetic points.
    Distances are Euclidean in the space of X_orig / X_synth — pass [:, :3] for 3D.

    If output_dir is given, saves nearest_synth_per_cluster.csv there.
    Returns {cluster: [(synth_idx, distance), ...]} sorted ascending.
    """
    meta = meta_orig.copy().reset_index(drop=True)
    meta["base_mod"] = meta["modality"].apply(base_mod)

    records: list[dict] = []
    nearest: dict[str, list[tuple[int, float]]] = {}

    for mod in sorted(meta["base_mod"].unique()):
        mask     = (meta["base_mod"] == mod).values
        centroid = X_orig[mask].mean(axis=0)
        dists    = np.linalg.norm(X_synth - centroid, axis=1)
        top_idx  = np.argsort(dists)[:top_k]
        nearest[mod] = [(int(i), float(dists[i])) for i in top_idx]

        log.info("Centroid [%s] — top-%d nearest synthetic:", mod, top_k)
        for rank, (i, d) in enumerate(nearest[mod], 1):
            path = str(meta_synth.iloc[i].get("image_path", "") or "")
            log.info("  #%d  dist=%.4f  %s", rank, d, path)
            records.append({
                "cluster":     mod,
                "rank":        rank,
                "synth_idx":   i,
                "distance":    round(d, 6),
                "image_path":  path,
                "subject":     meta_synth.iloc[i].get("subject", "?"),
                "session":     meta_synth.iloc[i].get("session", "?"),
                "modality_id": meta_synth.iloc[i].get(
                    "modality", meta_synth.iloc[i].get("modality_id", "?")),
            })

    if output_dir is not None:
        csv_path = output_dir / "nearest_synth_per_cluster.csv"
        pd.DataFrame(records).to_csv(csv_path, index=False)
        log.info("Nearest synth CSV → %s", csv_path)

    return nearest


def run_umap_joint(
    X_orig: np.ndarray,
    X_synth: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    target_weight: float = 0.0,
    meta_orig: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    X_all = np.vstack([X_orig, X_synth])
    n_orig = len(X_orig)

    y = None
    if target_weight > 0.0 and meta_orig is not None:
        # Semi-supervised: label original points by base modality, -1 = unlabeled (synthetic)
        mods = meta_orig["modality"].apply(base_mod).values
        classes = {m: i for i, m in enumerate(sorted(set(mods)))}
        y_orig  = np.array([classes[m] for m in mods], dtype=float)
        y_synth = np.full(len(X_synth), -1.0)
        y = np.concatenate([y_orig, y_synth])
        log.info(
            "Semi-supervised UMAP: target_weight=%.2f, %d classes: %s",
            target_weight, len(classes), sorted(classes),
        )

    log.info(
        "Fitting UMAP jointly on %d original + %d synthetic = %d points … (cuML=%s)",
        n_orig, len(X_synth), len(X_all), _CUML_AVAILABLE,
    )
    if _CUML_AVAILABLE and y is None:
        reducer = _CumlUMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=42,
            verbose=False,
        )
    else:
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            target_weight=target_weight if y is not None else 0.0,
            random_state=42,
            verbose=False,
            low_memory=False,
        )
    emb = reducer.fit_transform(X_all, y=y)
    log.info("UMAP fit complete.")
    graph = getattr(reducer, "graph_", None)  # None when cuML is used
    return emb[:n_orig], emb[n_orig:], graph


def compute_umap_connectivity(
    graph: sp.csr_matrix,
    n_orig: int,
    meta_orig: pd.DataFrame,
    emb_orig: np.ndarray,
    emb_synth: np.ndarray,
    output_dir: Path,
) -> np.ndarray:
    """
    Graph-theoretic connectivity analysis on UMAP's high-dim fuzzy graph.

    Metrics (Carlsson & Mémoli 2010; Fiedler 1973; Fischer et al. 2004):
      1. Threshold filtration — tracks β₀ (connected components) as edge-weight
         cutoff τ sweeps from max→0. n_comp_real = components among real nodes
         only; n_comp_real_in_combined = how many distinct components real nodes
         occupy in the full (real+synth) graph. The gap between the two is the
         number of real-cluster pairs that synth bridges at each τ.
      2. Fiedler value (λ₂ of the graph Laplacian) — algebraic connectivity.
      3. Max-bottleneck path between real groups via the max spanning tree.

    τ_zoom is set to the median edge weight — the threshold at which exactly
    half the manifold's local structure is retained (data-driven, no arbitrary
    choice). Below this, edges are weak forced connections; above, they are
    meaningful local similarities.

    Saves: filtration.csv, filtration.png, filtration_zoomed.png,
           connectivity_summary.csv, bottleneck.csv, umap_components_2d.png
    Returns component labels for real nodes at τ_best (for coords CSV).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Symmetrize (UMAP graph is not guaranteed symmetric)
    G = (graph + graph.T) / 2
    G_real = G[:n_orig, :n_orig]

    # Median edge weight → principled zoom cutoff
    tau_zoom = float(np.median(G.data))
    log.info("UMAP graph median edge weight (τ_zoom) = %.4f", tau_zoom)

    # ── 1. Threshold filtration ──────────────────────────────────────────────
    taus = np.linspace(float(G.data.max()), 0.0, 60)
    records = []
    labels_at_best: np.ndarray = np.zeros(n_orig, dtype=int)
    tau_best = 0.0
    gap_max = 0

    for tau in taus:
        def _mask(g, t=tau):
            g2 = g.copy()
            g2.data[g2.data < t] = 0
            g2.eliminate_zeros()
            return g2

        # Real-only subgraph components
        nc_real, lbl_real = connected_components(_mask(G_real), directed=False)

        # Real nodes' component membership in combined graph:
        # how many distinct groups do real nodes belong to when synth bridges exist?
        _, lbl_full = connected_components(_mask(G), directed=False)
        nc_real_in_combined = len(set(lbl_full[:n_orig]))

        gap = nc_real - nc_real_in_combined
        records.append({
            "tau":                   float(tau),
            "n_comp_real":           int(nc_real),
            "n_comp_real_in_combined": int(nc_real_in_combined),
            "bridge_gap":            int(gap),
        })

        if gap > gap_max:
            gap_max = gap
            labels_at_best = lbl_real
            tau_best = float(tau)

    filt_df = pd.DataFrame(records)
    filt_df.to_csv(output_dir / "connectivity_curve.csv", index=False)

    # τ_viz: highest τ where real-only has ≤ MAX_VIZ_COMP islands (readable structure)
    MAX_VIZ_COMP = 10
    viz_rows = filt_df[filt_df["n_comp_real"] <= MAX_VIZ_COMP]
    tau_viz  = float(viz_rows["tau"].max()) if len(viz_rows) else float(filt_df["tau"].min())
    n_comp_viz_pre = int(filt_df.loc[filt_df["tau"] == tau_viz, "n_comp_real"].values[0]) if tau_viz in filt_df["tau"].values else MAX_VIZ_COMP

    def _plot_connectivity_curve(df, path_stem, xlim=None):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["tau"], df["n_comp_real"],
                label="real only", color="steelblue")
        ax.plot(df["tau"], df["n_comp_real_in_combined"],
                label="real + synthetic", color="tomato", linestyle="--")
        ax.fill_between(df["tau"],
                        df["n_comp_real"], df["n_comp_real_in_combined"],
                        alpha=0.15, color="tomato", label="bridging gap (AUC)")
        ax.axvline(tau_viz, color="green", linestyle="--", alpha=0.7,
                   label=f"τ_viz = {tau_viz:.3f} ({n_comp_viz_pre} islands)")
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_xlabel("Edge weight threshold τ  (← stronger connections)")
        ax.set_ylabel("Disconnected real-data islands")
        ax.set_title("Island persistence curve: real vs real + synthetic\n"
                     "(how many real islands exist as we require stronger connections)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save_mpl(fig, output_dir, path_stem)
        plt.close(fig)

    _plot_connectivity_curve(filt_df, "connectivity_curve")
    # Zoomed: τ ∈ [0, τ_viz] — the readable end where islands are few (≤10)
    zoomed = filt_df[filt_df["tau"] <= tau_viz]
    if len(zoomed) > 1:
        _plot_connectivity_curve(zoomed, "connectivity_curve_zoomed", xlim=(0, tau_viz))

    # AUC of bridging gap — sort ascending so trapezoid integrates left→right
    auc_df     = filt_df.sort_values("tau")
    auc_full   = float(np.trapezoid(auc_df["bridge_gap"], auc_df["tau"]))
    auc_zoomed_df = auc_df[auc_df["tau"] <= tau_viz]
    auc_zoomed = float(np.trapezoid(auc_zoomed_df["bridge_gap"],
                                    auc_zoomed_df["tau"])) if len(auc_zoomed_df) > 1 else float("nan")

    # ── 2. Fiedler value ─────────────────────────────────────────────────────
    def _fiedler(g: sp.csr_matrix) -> float:
        g = (g + g.T) / 2
        deg = np.array(g.sum(axis=1)).flatten()
        L = sp.diags(deg) - g
        if L.shape[0] < 4:
            return float("nan")
        try:
            vals = spla.eigsh(L.astype(float), k=2, which="SM",
                              return_eigenvectors=False, tol=1e-4, maxiter=5000)
            return float(sorted(vals)[1])
        except Exception:
            return float("nan")

    fv_real     = _fiedler(G_real)
    fv_combined = _fiedler(G)

    # ── 3. Max-bottleneck paths via max spanning tree ────────────────────────
    neg_G = G.copy()
    neg_G.data = -neg_G.data
    mst_neg = minimum_spanning_tree(neg_G)
    mst_neg.data = -mst_neg.data
    mst_sym = mst_neg + mst_neg.T
    mst_nx = nx.from_scipy_sparse_array(mst_sym, edge_attribute="weight")

    mod_col  = "modality_id"   if "modality_id"   in meta_orig.columns else "modality"
    scan_col = "scanner_model" if "scanner_model"  in meta_orig.columns else "scanner"
    groups = (meta_orig[mod_col] + " × " + meta_orig[scan_col]).values
    unique_groups = np.unique(groups)
    group_reps = {g: int(np.where(groups == g)[0][0]) for g in unique_groups}

    bottleneck_rows = []
    group_list = list(group_reps.keys())
    for i, ga in enumerate(group_list):
        for gb in group_list[i + 1:]:
            na, nb = group_reps[ga], group_reps[gb]
            try:
                path = nx.shortest_path(mst_nx, na, nb)
                bw   = min(mst_nx[path[k]][path[k + 1]]["weight"]
                           for k in range(len(path) - 1))
                plen = len(path) - 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                bw, plen = 0.0, -1
            bottleneck_rows.append({"group_a": ga, "group_b": gb,
                                    "bottleneck_weight": bw, "path_hops": plen})

    bottleneck_df = (pd.DataFrame(bottleneck_rows)
                       .sort_values("bottleneck_weight", ascending=False))
    bottleneck_df.to_csv(output_dir / "bottleneck.csv", index=False)

    # ── 4. Component coloring on 2D UMAP embedding ───────────────────────────
    # τ_viz already computed above; recompute component labels at that threshold
    # Recompute component labels at τ_viz
    G_viz = G_real.copy()
    G_viz.data[G_viz.data < tau_viz] = 0
    G_viz.eliminate_zeros()
    n_comp_viz, labels_viz = connected_components(G_viz, directed=False)

    # Combined graph at τ_viz — to see which real islands merge when synth is added
    G_viz_full = G.copy()
    G_viz_full.data[G_viz_full.data < tau_viz] = 0
    G_viz_full.eliminate_zeros()
    _, lbl_combined = connected_components(G_viz_full, directed=False)
    # Real nodes' component IDs in the combined graph (relabelled 0..k)
    real_comp_combined_raw = lbl_combined[:n_orig]
    uid, inv = np.unique(real_comp_combined_raw, return_inverse=True)
    labels_combined = inv   # 0..n_merged_islands-1
    n_comp_merged = len(uid)

    log.info("Islands at τ_viz=%.3f — real-only: %d  after synth bridges: %d",
             tau_viz, n_comp_viz, n_comp_merged)

    def _island_label(labels, meta, cid):
        mask = labels == cid
        if mod_col in meta.columns:
            dominant = meta.loc[mask, mod_col].value_counts().index[0]
            return f"island {cid}: {dominant} (n={mask.sum()})"
        return f"island {cid} (n={mask.sum()})"

    cmap_real   = plt.get_cmap("tab10", max(n_comp_viz,    2))
    cmap_merged = plt.get_cmap("tab10", max(n_comp_merged, 2))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── Plot 1: real data only, no synth ─────────────────────────────────────
    ax = axes[0]
    for cid in range(n_comp_viz):
        mask = labels_viz == cid
        if not mask.any():
            continue
        ax.scatter(emb_orig[mask, 0], emb_orig[mask, 1],
                   color=cmap_real(cid), s=22, alpha=0.9,
                   label=_island_label(labels_viz, meta_orig, cid))
    ax.set_title(f"Real data islands (τ = {tau_viz:.3f})\n{n_comp_viz} disconnected islands")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=1.5)

    # ── Plot 2: real islands after synth bridging ─────────────────────────────
    ax = axes[1]
    ax.scatter(emb_synth[:, 0], emb_synth[:, 1],
               color="lightgray", s=3, alpha=0.2, zorder=1, label="synthetic")
    for cid in range(n_comp_merged):
        mask = labels_combined == cid
        if not mask.any():
            continue
        ax.scatter(emb_orig[mask, 0], emb_orig[mask, 1],
                   color=cmap_merged(cid), s=22, alpha=0.9, zorder=2,
                   label=_island_label(labels_combined, meta_orig, cid))
    ax.set_title(f"Real islands after synth bridging (τ = {tau_viz:.3f})\n"
                 f"{n_comp_viz} → {n_comp_merged} islands")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=1.5)

    fig.suptitle("Effect of synthetic data on manifold connectivity", fontsize=11)
    fig.tight_layout()
    _save_mpl(fig, output_dir, "umap_components_2d")
    plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────────────
    best_row = filt_df.loc[filt_df["bridge_gap"].idxmax()]
    summary = pd.DataFrame([{
        "fiedler_real_only":              fv_real,
        "fiedler_combined":               fv_combined,
        "fiedler_delta":                  fv_combined - fv_real,
        "tau_zoom_median_edge":           tau_zoom,
        "tau_best_max_bridge_gap":        tau_best,
        "n_comp_real_at_tau_best":        int(best_row["n_comp_real"]),
        "n_comp_real_in_combined_at_tau_best": int(best_row["n_comp_real_in_combined"]),
        "bridge_gap_at_tau_best":         int(best_row["bridge_gap"]),
        "auc_bridge_gap_full":            auc_full,
        "auc_bridge_gap_zoomed":          auc_zoomed,
    }])
    summary.to_csv(output_dir / "connectivity_summary.csv", index=False)

    log.info(
        "Connectivity | Fiedler real=%.4f → combined=%.4f (Δ=%.4f) | "
        "τ*=%.3f: %d real clusters → %d when synth added (gap=%d) | "
        "AUC(zoomed)=%.4f",
        fv_real, fv_combined, fv_combined - fv_real,
        tau_best, int(best_row["n_comp_real"]),
        int(best_row["n_comp_real_in_combined"]),
        int(best_row["bridge_gap"]),
        auc_zoomed,
    )
    return labels_at_best


def remove_outliers(
    X_pca: np.ndarray,
    meta: pd.DataFrame,
    std_thresh: float,
    output_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Drop rows whose max |z-score| across any PC exceeds std_thresh.
    Logs removed scans to outliers_removed.csv in output_dir.
    """
    stds = X_pca.std(axis=0)
    stds[stds == 0] = 1.0
    z = np.abs((X_pca - X_pca.mean(axis=0)) / stds)
    max_z = z.max(axis=1)
    keep = max_z <= std_thresh
    n_removed = (~keep).sum()
    if n_removed:
        removed = meta[~keep].copy()
        removed["max_pc_zscore"] = max_z[~keep]
        out_path = output_dir / "outliers_removed.csv"
        removed.to_csv(out_path, index=False)
        log.info(
            "Outlier removal (|z| > %.1f): removed %d / %d synthetic scans → %s",
            std_thresh, n_removed, len(X_pca), out_path,
        )
        for _, row in removed.iterrows():
            log.info("  REMOVED  %s  %s  %s  max|z|=%.2f",
                     row.get("subject","?"), row.get("session","?"),
                     row.get("image_path","?"), row["max_pc_zscore"])
    else:
        log.info("Outlier removal (|z| > %.1f): no outliers found", std_thresh)
    return X_pca[keep], meta[keep].reset_index(drop=True)


def _save_mpl(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        p = output_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        log.info("Saved → %s", p)
    plt.close(fig)


def _scatter_layer(ax, X, meta, sc_marker):
    meta = meta.copy()
    meta["base_mod"] = meta["modality"].apply(base_mod)
    for sc in sorted(meta["scanner"].unique()):
        for mod in sorted(meta["base_mod"].unique()):
            mask = ((meta["scanner"] == sc) & (meta["base_mod"] == mod)).values
            if not mask.any():
                continue
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=MODALITY_PALETTE.get(mod, "#888"), marker=sc_marker[sc],
                       alpha=0.80, s=22, linewidths=0, zorder=2)


def plot_pca(X_orig_pca, meta_orig, X_synth_pca, pca, output_dir,
             stem: str = "pca_joint_2d", include_synth: bool = True):
    """2-panel scatter (PC1×PC2, PC1×PC3). Set include_synth=False for original-only view."""
    meta_orig = meta_orig.copy()
    meta_orig["base_mod"] = meta_orig["modality"].apply(base_mod)
    scanners   = sorted(meta_orig["scanner"].unique())
    modalities = sorted(meta_orig["base_mod"].unique())
    sc_marker  = {sc: MPL_MARKERS[i % len(MPL_MARKERS)] for i, sc in enumerate(scanners)}

    ev = pca.explained_variance_ratio_ * 100
    pairs = [
        (0, 1, f"PC1 ({ev[0]:.1f}%)", f"PC2 ({ev[1]:.1f}%)"),
        (0, 2, f"PC1 ({ev[0]:.1f}%)", f"PC3 ({ev[2]:.1f}%)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (xi, yi, xlabel, ylabel) in zip(axes, pairs):
        if include_synth:
            ax.scatter(X_synth_pca[:, xi], X_synth_pca[:, yi],
                       c=SYNTH_COLOR, alpha=SYNTH_ALPHA, s=10, linewidths=0,
                       label=SYNTH_LABEL, zorder=1)
        for sc in scanners:
            for mod in modalities:
                mask = ((meta_orig["scanner"] == sc) & (meta_orig["base_mod"] == mod)).values
                if not mask.any():
                    continue
                ax.scatter(X_orig_pca[mask, xi], X_orig_pca[mask, yi],
                           c=MODALITY_PALETTE.get(mod, "#888"), marker=sc_marker[sc],
                           alpha=0.80, s=22, linewidths=0, zorder=2)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"PCA — {xlabel} vs {ylabel}")

    mod_h = [mpatches.Patch(color=MODALITY_PALETTE.get(m, "#888"), label=m) for m in modalities]
    sc_h  = [plt.Line2D([0], [0], marker=sc_marker[sc], color="gray",
                         linestyle="none", markersize=6, label=sc) for sc in scanners]
    handles = mod_h + sc_h
    if include_synth:
        handles += [mpatches.Patch(color=SYNTH_COLOR, alpha=0.5, label=SYNTH_LABEL)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, framealpha=0.7,
               bbox_to_anchor=(0.5, -0.04))
    title = ("PCA — original only" if not include_synth
             else "PCA — original (colour=modality, symbol=scanner) + synthetic (black)")
    fig.suptitle(title)
    fig.tight_layout()
    _save_mpl(fig, output_dir, stem)
    log.info("PCA 2D scatter saved → %s", stem)


def _hover_text(meta: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (hover_strings, paths) per row. Click-to-copy uses paths."""
    hover, paths = [], []
    for _, r in meta.iterrows():
        path = str(r.get("image_path", "") or "")
        fname = Path(path).name if path else "—"
        hover.append(
            f"<b>{r.get('subject','?')}</b> {r.get('session','?')}<br>"
            f"modality: {r.get('modality', r.get('modality_id','?'))}<br>"
            f"scanner:  {r.get('scanner', r.get('scanner_model','?'))}<br>"
            f"file:     {fname}<br>"
            f"<i>{'click to copy path' if path else 'no path'}</i>"
        )
        paths.append(path if path else "none")
    return hover, paths


def plot_pca_3d(X_orig_pca, meta_orig, X_synth_pca, meta_synth, pca, output_dir,
                stem: str = "pca_joint_3d", include_synth: bool = True,
                nearest_synth: dict | None = None,
                save_nearest_csv: bool = False):
    """Interactive 3D PCA scatter saved as HTML. Click point → copy path.

    When include_synth=True, cluster centroids and top-3 nearest synthetic points
    are always computed using the 3 displayed PCA dimensions (PC1-3) so they are
    visually correct. nearest_synth can optionally be passed in (pre-computed); if
    None, it is computed here. save_nearest_csv=True writes the CSV to output_dir.
    """
    meta_orig  = meta_orig.copy().reset_index(drop=True)
    meta_synth = meta_synth.copy().reset_index(drop=True)
    meta_orig["base_mod"] = meta_orig["modality"].apply(base_mod)
    scanners   = sorted(meta_orig["scanner"].unique())
    symbol_map = {sc: PLOTLY_SYMS[i % len(PLOTLY_SYMS)] for i, sc in enumerate(scanners)}
    ev = pca.explained_variance_ratio_ * 100
    hover_orig,  paths_orig  = _hover_text(meta_orig)
    hover_synth, paths_synth = _hover_text(meta_synth)

    traces: list = []

    # ── Original data ─────────────────────────────────────────────────────────
    for sc in scanners:
        for mod in sorted(meta_orig["base_mod"].unique()):
            mask = ((meta_orig["scanner"] == sc) & (meta_orig["base_mod"] == mod)).values
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            traces.append(go.Scatter3d(
                x=X_orig_pca[mask, 0], y=X_orig_pca[mask, 1], z=X_orig_pca[mask, 2],
                mode="markers",
                marker=dict(size=3, color=MODALITY_PALETTE.get(mod, "#888"),
                            symbol=symbol_map[sc], opacity=0.85),
                name=f"{mod} / {sc}",
                legendgroup=mod,
                text=[hover_orig[i] for i in idx],
                customdata=[[paths_orig[i]] for i in idx],
                hoverinfo="text",
            ))

    n_orig_traces = len(traces)

    # ── All-synthetic trace ───────────────────────────────────────────────────
    if include_synth:
        traces.append(go.Scatter3d(
            x=X_synth_pca[:, 0], y=X_synth_pca[:, 1], z=X_synth_pca[:, 2],
            mode="markers",
            marker=dict(size=2.5, color=SYNTH_COLOR, opacity=SYNTH_ALPHA),
            name=SYNTH_LABEL,
            text=hover_synth,
            customdata=[[p] for p in paths_synth],
            hoverinfo="text",
        ))

    # ── Centroid + Top-3 traces (one pair per cluster) ────────────────────────
    # Always compute in the 3D display space (PC1-3) so points appear visually closest.
    if include_synth and nearest_synth is None:
        nearest_synth = _nearest_synth_in_space(
            X_orig_pca[:, :3], meta_orig,
            X_synth_pca[:, :3], meta_synth,
            output_dir=output_dir if save_nearest_csv else None,
        )

    clusters_ordered: list[str] = []
    if nearest_synth and include_synth:
        meta_bm = meta_orig.copy()
        meta_bm["base_mod"] = meta_bm["modality"].apply(base_mod)

        for mod in sorted(nearest_synth.keys()):
            color    = MODALITY_PALETTE.get(mod, "#888")
            mask_mod = (meta_bm["base_mod"] == mod).values
            centroid = X_orig_pca[mask_mod].mean(axis=0)

            # Centroid star — always visible
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

            # Top-3 nearest synth — hidden by default, shown in "Top-3" mode
            top_pairs = nearest_synth[mod]
            top_x  = [X_synth_pca[i, 0] for i, _ in top_pairs]
            top_y  = [X_synth_pca[i, 1] for i, _ in top_pairs]
            top_z  = [X_synth_pca[i, 2] for i, _ in top_pairs]
            top_hover, top_paths = [], []
            for rank, (i, d) in enumerate(top_pairs, 1):
                path  = str(meta_synth.iloc[i].get("image_path", "") or "")
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

    # ── Layout ────────────────────────────────────────────────────────────────
    title = ("PCA — original only (3D)" if not include_synth
             else "PCA — original + synthetic (3D)")
    layout_kwargs: dict = dict(
        title=title,
        scene=dict(
            xaxis_title=f"PC1 ({ev[0]:.1f}%)",
            yaxis_title=f"PC2 ({ev[1]:.1f}%)",
            zaxis_title=f"PC3 ({ev[2]:.1f}%)",
        ),
        legend=dict(font=dict(size=9)),
        margin=dict(l=0, r=0, b=0, t=60),
    )

    if n_clusters > 0:
        # Trace order: [orig × N] [synth_all] [centroid, top3] × C
        # "All synthetic": show orig + synth_all + centroids, hide top-3
        vis_all = (
            [True]  * n_orig_traces +          # orig
            [True]  +                          # synth_all
            [True, False] * n_clusters         # centroid=show, top3=hide (interleaved)
        )
        # "Top-3 per cluster": show orig + centroids + top3, hide synth_all
        vis_top3 = (
            [True]  * n_orig_traces +          # orig
            [False] +                          # synth_all hidden
            [True, True]  * n_clusters         # centroid=show, top3=show
        )
        layout_kwargs["updatemenus"] = [dict(
            type="buttons",
            direction="left",
            x=0.0, xanchor="left",
            y=1.13, yanchor="top",
            showactive=True,
            font=dict(size=12),
            buttons=[
                dict(label="⬛ All synthetic",
                     method="update",
                     args=[{"visible": vis_all}]),
                dict(label="💎 Top-3 per cluster",
                     method="update",
                     args=[{"visible": vis_top3}]),
            ],
        )]

    fig = go.Figure(data=traces)
    fig.update_layout(**layout_kwargs)
    html_path = output_dir / f"{stem}.html"
    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        post_script=_CLICK_TO_COPY_JS,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    log.info("Saved → %s", html_path)

    # ── Companion 2D lasso-enabled HTML (PC1×PC2) ─────────────────────────────
    # go.Scatter3d does not support lasso2d; we generate a flat Scatter for selection.
    traces_2d: list = []
    for sc in scanners:
        for mod in sorted(meta_orig["base_mod"].unique()):
            mask = ((meta_orig["scanner"] == sc) & (meta_orig["base_mod"] == mod)).values
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            traces_2d.append(go.Scatter(
                x=X_orig_pca[mask, 0], y=X_orig_pca[mask, 1],
                mode="markers",
                marker=dict(size=6, color=MODALITY_PALETTE.get(mod, "#888"),
                            symbol=symbol_map[sc], opacity=0.85,
                            line=dict(width=0.5, color="white")),
                name=f"{mod} / {sc}",
                legendgroup=mod,
                text=[hover_orig[i] for i in idx],
                customdata=[[paths_orig[i]] for i in idx],
                hoverinfo="text",
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=0.1)),
            ))
    if include_synth:
        traces_2d.append(go.Scatter(
            x=X_synth_pca[:, 0], y=X_synth_pca[:, 1],
            mode="markers",
            marker=dict(size=3, color=SYNTH_COLOR, opacity=SYNTH_ALPHA,
                        line=dict(width=0)),
            name=SYNTH_LABEL,
            text=hover_synth,
            customdata=[[p] for p in paths_synth],
            hoverinfo="text",
        ))
    title_2d = ("PCA — original only (PC1×PC2)" if not include_synth
                else "PCA — original + synthetic (PC1×PC2) — lasso to select")
    fig_2d = go.Figure(data=traces_2d)
    fig_2d.update_layout(
        title=title_2d,
        xaxis_title=f"PC1 ({ev[0]:.1f}%)",
        yaxis_title=f"PC2 ({ev[1]:.1f}%)",
        legend=dict(font=dict(size=9)),
        margin=dict(l=60, r=20, b=60, t=60),
        dragmode="lasso",
    )
    stem_2d = stem.replace("_3d", "_2d") if "_3d" in stem else stem + "_2d"
    html_path_2d = output_dir / f"{stem_2d}.html"
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


def plot_2d(emb_orig, meta_orig, emb_synth, output_dir):
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(emb_synth[:, 0], emb_synth[:, 1],
               c=SYNTH_COLOR, alpha=SYNTH_ALPHA, s=10, linewidths=0,
               label=SYNTH_LABEL, zorder=1)

    meta_orig = meta_orig.copy()
    meta_orig["base_mod"] = meta_orig["modality"].apply(base_mod)
    scanners   = sorted(meta_orig["scanner"].unique())
    modalities = sorted(meta_orig["base_mod"].unique())
    sc_marker  = {sc: MPL_MARKERS[i % len(MPL_MARKERS)] for i, sc in enumerate(scanners)}

    for sc in scanners:
        for mod in modalities:
            mask = ((meta_orig["scanner"] == sc) & (meta_orig["base_mod"] == mod)).values
            if not mask.any():
                continue
            ax.scatter(emb_orig[mask, 0], emb_orig[mask, 1],
                       c=MODALITY_PALETTE.get(mod, "#888"), marker=sc_marker[sc],
                       alpha=0.80, s=22, linewidths=0, zorder=2)

    mod_h = [mpatches.Patch(color=MODALITY_PALETTE.get(m, "#888"), label=m) for m in modalities]
    sc_h  = [plt.Line2D([0], [0], marker=sc_marker[sc], color="gray",
                         linestyle="none", markersize=6, label=sc) for sc in scanners]
    sy_h  = [mpatches.Patch(color=SYNTH_COLOR, alpha=0.5, label=SYNTH_LABEL)]
    ax.legend(handles=mod_h + sc_h + sy_h, loc="best", fontsize=7, framealpha=0.7)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("Contrast Manifold — Joint UMAP (original + synthetic_v19)")
    _save_mpl(fig, output_dir, "umap_joint_2d")


def plot_3d(emb_orig, meta_orig, emb_synth, meta_synth, output_dir):
    meta_orig = meta_orig.copy().reset_index(drop=True)
    meta_synth = meta_synth.copy().reset_index(drop=True)
    meta_orig["base_mod"] = meta_orig["modality"].apply(base_mod)
    scanners   = sorted(meta_orig["scanner"].unique())
    symbol_map = {sc: PLOTLY_SYMS[i % len(PLOTLY_SYMS)] for i, sc in enumerate(scanners)}
    hover_orig,  paths_orig  = _hover_text(meta_orig)
    hover_synth, paths_synth = _hover_text(meta_synth)

    traces = []
    for sc in scanners:
        for mod in sorted(meta_orig["base_mod"].unique()):
            mask = ((meta_orig["scanner"] == sc) & (meta_orig["base_mod"] == mod)).values
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            traces.append(go.Scatter3d(
                x=emb_orig[mask, 0], y=emb_orig[mask, 1], z=emb_orig[mask, 2],
                mode="markers",
                marker=dict(size=3, color=MODALITY_PALETTE.get(mod, "#888"),
                            symbol=symbol_map[sc], opacity=0.85),
                name=f"{mod} / {sc}",
                legendgroup=mod,
                text=[hover_orig[i] for i in idx],
                customdata=[[paths_orig[i]] for i in idx],
                hoverinfo="text",
            ))

    traces.append(go.Scatter3d(
        x=emb_synth[:, 0], y=emb_synth[:, 1], z=emb_synth[:, 2],
        mode="markers",
        marker=dict(size=2.5, color=SYNTH_COLOR, opacity=SYNTH_ALPHA),
        name=SYNTH_LABEL,
        text=hover_synth,
        customdata=[[p] for p in paths_synth],
        hoverinfo="text",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Contrast Manifold — Joint UMAP (original + synthetic_v19, 3D)",
        scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"),
        legend=dict(font=dict(size=9)),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    html_path = output_dir / "umap_joint_3d.html"
    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        post_script=_CLICK_TO_COPY_JS,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    log.info("Saved → %s", html_path)
    for ext in ("png", "pdf"):
        try:
            fig.write_image(str(output_dir / f"umap_joint_3d.{ext}"),
                            width=1400, height=1000, scale=2)
            log.info("Saved → %s", output_dir / f"umap_joint_3d.{ext}")
        except Exception as e:
            log.warning("Could not save %s: %s", ext, e)


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


def _short_name(name: str) -> str:
    """Strip family prefix for display: 'glcm_Contrast' → 'Contrast (glcm)'."""
    for fam in FAMILY_COLORS:
        prefix = fam + "_"
        if name.lower().startswith(prefix):
            return f"{name[len(prefix):]} [{fam}]"
    return name


def plot_pca_loadings(
    pca: PCA,
    feat_cols: list[str],
    output_dir: Path,
    n_top: int = 15,
    n_pcs: int = 5,
) -> None:
    """
    Horizontal bar chart: top n_top features by |loading| for each of the first n_pcs PCs.
    Bars coloured by feature family. Positive loadings → right, negative → left.
    """
    n_pcs = min(n_pcs, pca.n_components_)
    ev = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, n_pcs, figsize=(5 * n_pcs, max(6, n_top * 0.45)),
                              sharey=False)
    if n_pcs == 1:
        axes = [axes]

    for ax, pc_idx in zip(axes, range(n_pcs)):
        loadings = pca.components_[pc_idx]
        order    = np.argsort(np.abs(loadings))[::-1][:n_top]
        top_vals = loadings[order]
        top_names = [feat_cols[i] for i in order]
        colors = [FAMILY_COLORS.get(get_family(n), "#888") for n in top_names]
        labels = [_short_name(n) for n in top_names]

        y_pos = np.arange(len(top_vals))
        ax.barh(y_pos, top_vals, color=colors, edgecolor="none", height=0.7)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Loading")
        ax.set_title(f"PC{pc_idx + 1} ({ev[pc_idx]:.1f}%)", fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    # Family legend
    handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()
               if any(get_family(n) == f for n in feat_cols)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=7, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"PCA loadings — top {n_top} features per component", fontsize=11)
    fig.tight_layout()
    _save_mpl(fig, output_dir, "pca_loadings")
    log.info("PCA loadings plot saved.")


def plot_pc_distributions(
    X_orig_pca: np.ndarray,
    meta_orig: pd.DataFrame,
    X_synth_pca: np.ndarray,
    pca: PCA,
    output_dir: Path,
    n_pcs: int = 4,
) -> None:
    """
    Violin plots of PC scores split by modality (original) and synthetic.
    Shows which PCs place synthetic near specific modalities.
    """
    n_pcs = min(n_pcs, pca.n_components_)
    ev    = pca.explained_variance_ratio_ * 100

    meta = meta_orig.copy().reset_index(drop=True)
    meta["base_mod"] = meta["modality"].apply(base_mod)
    modalities = sorted(meta["base_mod"].unique())
    groups = modalities + ["synthetic"]
    palette = {m: MODALITY_PALETTE.get(m, "#888") for m in modalities}
    palette["synthetic"] = SYNTH_COLOR

    fig, axes = plt.subplots(n_pcs, 1, figsize=(max(8, len(groups) * 0.9), 3.5 * n_pcs),
                              sharex=False)
    if n_pcs == 1:
        axes = [axes]

    for ax, pc_idx in zip(axes, range(n_pcs)):
        data_by_group = []
        for mod in modalities:
            mask = (meta["base_mod"] == mod).values
            data_by_group.append(X_orig_pca[mask, pc_idx])
        data_by_group.append(X_synth_pca[:, pc_idx])

        parts = ax.violinplot(data_by_group, positions=range(len(groups)),
                               showmedians=True, showextrema=False, widths=0.7)
        for i, (body, grp) in enumerate(zip(parts["bodies"], groups)):
            body.set_facecolor(palette[grp])
            body.set_alpha(0.65)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(f"PC{pc_idx + 1} score", fontsize=8)
        ax.set_title(f"PC{pc_idx + 1} ({ev[pc_idx]:.1f}% variance)", fontsize=9, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("PC score distributions — original modalities vs synthetic", fontsize=11)
    fig.tight_layout()
    _save_mpl(fig, output_dir, "pca_pc_distributions")
    log.info("PC distributions plot saved.")


def plot_loading_heatmap(
    pca: PCA,
    feat_cols: list[str],
    output_dir: Path,
    n_top: int = 30,
    n_pcs: int = 6,
) -> None:
    """
    Heatmap of PCA loadings: rows = top n_top features (by max |loading| across shown PCs),
    columns = PCs. Row labels coloured by feature family.
    """
    n_pcs = min(n_pcs, pca.n_components_)
    ev    = pca.explained_variance_ratio_ * 100
    L     = pca.components_[:n_pcs].T  # (n_features, n_pcs)

    # Pick top features by max |loading| across the shown PCs
    importance = np.abs(L).max(axis=1)
    top_idx    = np.argsort(importance)[::-1][:n_top]
    L_top      = L[top_idx]
    names_top  = [feat_cols[i] for i in top_idx]

    vmax = np.abs(L_top).max()
    fig, ax = plt.subplots(figsize=(n_pcs * 1.1 + 1.5, n_top * 0.32 + 1.5))
    im = ax.imshow(L_top, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="none")
    ax.set_xticks(range(n_pcs))
    ax.set_xticklabels([f"PC{i+1}\n({ev[i]:.1f}%)" for i in range(n_pcs)], fontsize=8)
    ax.set_yticks(range(n_top))
    labels = [_short_name(n) for n in names_top]
    ax.set_yticklabels(labels, fontsize=6.5)

    # Colour y-tick labels by family
    for tick, name in zip(ax.get_yticklabels(), names_top):
        tick.set_color(FAMILY_COLORS.get(get_family(name), "#333"))

    plt.colorbar(im, ax=ax, shrink=0.6, label="Loading")
    ax.set_title(f"PCA loading heatmap — top {n_top} features", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save_mpl(fig, output_dir, "pca_loading_heatmap")
    log.info("PCA loading heatmap saved.")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--original_csv", type=Path,
                   default=Path("analysis/contrast_manifold/outputs/data/original/roi_mask/"
                                "on_harmony_features_normalized_combined_downsampled100.csv"))
    p.add_argument("--synthetic_csv", type=Path,
                   default=Path("analysis/contrast_manifold/outputs/data/synthetic_v19/roi_mask/"
                                "synthetic_v19_features_normalized_combined.csv"))
    p.add_argument("--output_dir", type=Path,
                   default=Path("analysis/contrast_manifold/outputs/plots/umap_joint_v19"))
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--plot_pca", action="store_true",
                   help="Also save a PCA scatter plot (PC1 vs PC2 and PC1 vs PC3).")
    p.add_argument("--plot_loadings", action="store_true",
                   help="Save PCA loadings bar charts, PC distribution violins, and loading heatmap.")
    p.add_argument(
        "--feature_config", type=Path, default=None,
        help="YAML config specifying exclude_families and exclude_features. "
             "When set, outputs go into a 'feat_selected/' subfolder.",
    )
    p.add_argument(
        "--outlier_std", type=float, default=None,
        help="Remove original scans whose max |z-score| across any PC exceeds this "
             "threshold (applied after PCA, before UMAP). E.g. --outlier_std 5",
    )
    p.add_argument("--skip_umap", action="store_true",
                   help="Skip UMAP fitting and 2D/3D UMAP plots (useful when only PCA is needed).")
    p.add_argument(
        "--target_weight", type=float, default=0.0,
        help="Semi-supervised UMAP target weight (0 = fully unsupervised, 1 = fully supervised). "
             "When > 0, original points are labelled by base modality and synthetic points are "
             "unlabelled (-1), so contrast becomes the primary structure. Default: 0 (off).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.original_csv.exists():
        raise FileNotFoundError(f"Original normalized CSV not found: {args.original_csv}")
    if not args.synthetic_csv.exists():
        raise FileNotFoundError(f"Synthetic normalized CSV not found: {args.synthetic_csv}")

    if args.feature_config:
        args.output_dir = args.output_dir / args.feature_config.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    meta_cols_orig  = [c for c in META_COLS if c in df_orig.columns]
    meta_cols_synth = [c for c in META_COLS if c in df_synth.columns]
    meta_orig  = df_orig[meta_cols_orig].reset_index(drop=True)
    meta_synth = df_synth[meta_cols_synth].reset_index(drop=True)

    # Joint PCA (fit on original+synthetic combined)
    X_orig_pca, X_synth_pca, pca_joint = run_pca(X_orig, X_synth)
    # Original-only PCA (fit on original, synthetic projected)
    X_orig_pca_o, X_synth_pca_o, pca_orig = run_pca_orig_only(X_orig, X_synth)

    if args.outlier_std is not None:
        X_synth_pca,   meta_synth   = remove_outliers(X_synth_pca,   meta_synth, args.outlier_std, args.output_dir)
        X_synth_pca_o, _            = remove_outliers(X_synth_pca_o, meta_synth, args.outlier_std, args.output_dir)

    if args.plot_pca:
        # 1. Original only (joint PCA axes)
        plot_pca(X_orig_pca, meta_orig, X_synth_pca, pca_joint, args.output_dir,
                 stem="pca_joint_original_only_2d", include_synth=False)
        plot_pca_3d(X_orig_pca, meta_orig, X_synth_pca, meta_synth, pca_joint, args.output_dir,
                    stem="pca_joint_original_only_3d", include_synth=False)
        # 2. Original + synthetic (joint PCA axes) — nearest computed in joint-3D space
        plot_pca(X_orig_pca, meta_orig, X_synth_pca, pca_joint, args.output_dir,
                 stem="pca_joint_with_synth_2d", include_synth=True)
        plot_pca_3d(X_orig_pca, meta_orig, X_synth_pca, meta_synth, pca_joint, args.output_dir,
                    stem="pca_joint_with_synth_3d", include_synth=True)
        # 3. Original only (original-only PCA axes — unbiased reference)
        plot_pca(X_orig_pca_o, meta_orig, X_synth_pca_o, pca_orig, args.output_dir,
                 stem="pca_original_axes_only_2d", include_synth=False)
        plot_pca_3d(X_orig_pca_o, meta_orig, X_synth_pca_o, meta_synth, pca_orig, args.output_dir,
                    stem="pca_original_axes_only_3d", include_synth=False)
        # 4. Original + synthetic (original-only PCA axes — most diagnostic)
        #    nearest computed in orig-only-3D space; CSV saved here.
        plot_pca(X_orig_pca_o, meta_orig, X_synth_pca_o, pca_orig, args.output_dir,
                 stem="pca_original_axes_with_synth_2d", include_synth=True)
        plot_pca_3d(X_orig_pca_o, meta_orig, X_synth_pca_o, meta_synth, pca_orig, args.output_dir,
                    stem="pca_original_axes_with_synth_3d", include_synth=True,
                    save_nearest_csv=True)

    if args.plot_loadings:
        plot_pca_loadings(pca_joint, feat_cols, args.output_dir)
        plot_pc_distributions(X_orig_pca, meta_orig, X_synth_pca, pca_joint, args.output_dir)
        plot_loading_heatmap(pca_joint, feat_cols, args.output_dir)

    if args.skip_umap:
        log.info("All plots saved to %s (UMAP skipped)", args.output_dir)
        return

    emb_orig, emb_synth, umap_graph = run_umap_joint(
        X_orig_pca, X_synth_pca,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        target_weight=args.target_weight,
        meta_orig=meta_orig,
    )

    # Connectivity analysis on UMAP's high-dim fuzzy graph (reuses the already-fit graph)
    comp_labels_real = None
    if umap_graph is not None:
        conn_dir = args.output_dir / "connectivity"
        comp_labels_real = compute_umap_connectivity(
            umap_graph, len(X_orig_pca), meta_orig,
            emb_orig, emb_synth, conn_dir,
        )

    # Build coords CSV; include real-node component label at max-fragmentation τ
    comp_col_orig  = comp_labels_real if comp_labels_real is not None else np.full(len(emb_orig),  -1)
    comp_col_synth = np.full(len(emb_synth), -1)
    coords_orig  = pd.concat([meta_orig,
                               pd.DataFrame(emb_orig,  columns=["umap1","umap2","umap3"]),
                               pd.DataFrame({"umap_component": comp_col_orig})], axis=1)
    coords_synth = pd.concat([meta_synth,
                               pd.DataFrame(emb_synth, columns=["umap1","umap2","umap3"]),
                               pd.DataFrame({"umap_component": comp_col_synth})], axis=1)
    coords_all = pd.concat([coords_orig, coords_synth], axis=0, ignore_index=True)
    coords_path = args.output_dir / "umap_joint_coords.csv"
    coords_all.to_csv(coords_path, index=False)
    log.info("Saved coords → %s", coords_path)

    plot_2d(emb_orig, meta_orig, emb_synth, args.output_dir)
    plot_3d(emb_orig, meta_orig, emb_synth, meta_synth, args.output_dir)
    log.info("All plots saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
