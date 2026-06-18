#!/usr/bin/env python
"""
Strict, by-the-book H0 Betti curve of the real / synthetic point clouds, in
feature (PCA) space — a literature-clean companion to the UMAP "connectivity
curve" (which this does NOT replace).

WHAT MAKES THIS "BY THE BOOK"
-----------------------------
For a point cloud, the 0-dimensional persistent homology of the Vietoris-Rips
filtration is EXACTLY single-linkage clustering, which is read off the Euclidean
minimum spanning tree (Gower & Ross 1969; Carlsson & Mémoli 2010; Edelsbrunner
& Harer 2010):

  * Filtration parameter  ε  = Euclidean distance (the x-axis; NOT a chosen knob).
  * β0(ε) = number of connected components when all edges with weight ≤ ε are
            present  =  N − (#MST edges with weight ≤ ε).
  * β0(0) = N  (every point isolated),  β0(∞) = 1  (single component).
  * Each finite H0 bar is born at 0 and dies at an MST edge weight, so
        total H0 persistence  =  Σ (MST edge weights)  =  ∫ (β0(ε) − 1) dε
    (Cohen-Steiner, Edelsbrunner & Harer 2007 — total persistence, stable).

We compute the EXACT MST on the full pairwise distance matrix (not a kNN
approximation), so the curve is the exact Rips H0 Betti curve. Self-checks at
the bottom assert N → 1, exactly N−1 deaths, and the total-persistence identity.

BRIDGING (relative / image persistence; Barannikov et al. 2021 in spirit)
------------------------------------------------------------------------
  β0_real(ε)            : components among REAL points using real edges only.
  β0_real_in_comb(ε)    : number of distinct components of the REAL+SYNTH
                          filtration that contain ≥1 real point (real points may
                          merge early via synthetic steppingstones).
  bridging_gap(ε)       = β0_real(ε) − β0_real_in_comb(ε)   (≥ 0)
  total_persistence_reduction
                        = ∫ gap dε   (the AREA BETWEEN the two Betti curves;
                          since gap ≥ 0 this is also their L1 distance)
                        = total_persistence_real − total_persistence_real_in_comb
                        = reduction in the real clusters' H0 total persistence
                          caused by synthetic.  ← headline, full integral (no zoom).
                        NB: this is area BETWEEN curves, not "area under a curve",
                        so it is deliberately NOT called an AUC.

Fairness: PCA is fit on REAL ONLY (synthetic projected in), so β0_real and its
total persistence depend only on the real data, not on which generator is being
scored.

Outputs (in --output_dir):
  betti_curves.png             β0_real & β0_combined (the two strict Betti curves)
  betti_bridging.png           β0_real vs β0_real_in_comb, shaded gap
  betti_summary.csv            total persistences, persistence reduction, checks
  betti_deaths.csv             death scales (MST weights) for each curve

Usage:
  set_slot 0 .venv/bin/python .../scripts/betti_curve.py \\
    --original_csv .../on_harmony_..._feat_selected.csv \\
    --synthetic_csv .../synthetic_..._feat_selected.csv \\
    --output_dir   .../umap/betti
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
    build_matrix,
    feature_cols,
    load_csv,
    load_feature_config,
    run_pca_orig_only,
    write_feature_log,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def mst_edges_sorted(X: np.ndarray) -> list[tuple[float, int, int]]:
    """Edges (weight, u, v) of the exact Euclidean MST of X, ascending by weight."""
    D = squareform(pdist(X))
    mst = minimum_spanning_tree(D).tocoo()
    return sorted(zip(mst.data.tolist(), mst.row.tolist(), mst.col.tolist()))


def real_only_deaths(X_real: np.ndarray) -> np.ndarray:
    """H0 death scales of the real point cloud = sorted real Euclidean MST weights."""
    edges = mst_edges_sorted(X_real)
    return np.array(sorted(w for w, _, _ in edges))


def real_in_combined_deaths(X_all: np.ndarray, n_real: int) -> np.ndarray:
    """
    Death scales at which REAL points merge inside the REAL+SYNTH Rips filtration.
    A 'real death' occurs when two components that each already contain a real
    point are joined (possibly via synthetic intermediaries). Returns the sorted
    array of those merge scales (length n_real − 1 for a connected cloud).
    """
    edges = mst_edges_sorted(X_all)
    n = len(X_all)
    parent = list(range(n))
    rank = [0] * n
    has_real = [i < n_real for i in range(n)]
    deaths: list[float] = []
    for w, u, v in edges:
        ru, rv = _find(parent, u), _find(parent, v)
        if ru == rv:
            continue
        if has_real[ru] and has_real[rv]:
            deaths.append(w)  # two real-containing components merge → a real H0 death
        merged_real = has_real[ru] or has_real[rv]
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        has_real[ru] = merged_real
        if rank[ru] == rank[rv]:
            rank[ru] += 1
    return np.array(sorted(deaths))


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def betti0(deaths: np.ndarray, n_start: int, eps: np.ndarray) -> np.ndarray:
    """β0(ε) = n_start − (#deaths ≤ ε)."""
    return n_start - np.searchsorted(deaths, eps, side="right")


def verify_betti_curve(deaths: np.ndarray, n_start: int, name: str) -> dict:
    """Assert the curve is a genuine H0 Betti curve and return check diagnostics."""
    deaths = np.sort(deaths)
    eps = np.concatenate([[0.0], deaths, [deaths[-1] * 1.001]])
    b = betti0(deaths, n_start, eps)
    # 1) starts at N, ends at 1
    assert b[0] == n_start, f"{name}: β0(0)={b[0]} ≠ N={n_start}"
    assert b[-1] == 1, f"{name}: β0(∞)={b[-1]} ≠ 1"
    # 2) monotone non-increasing, integer-valued
    assert np.all(np.diff(b) <= 0), f"{name}: β0 not monotone"
    # 3) exactly N−1 deaths (one infinite component survives)
    assert len(deaths) == n_start - 1, f"{name}: {len(deaths)} deaths ≠ N−1={n_start-1}"
    # 4) total-persistence identity: ∫(β0−1)dε == Σ deaths
    grid = np.linspace(0, deaths[-1], 200_000)
    integral = np.trapz(betti0(deaths, n_start, grid) - 1, grid)
    total_pers = float(deaths.sum())
    rel_err = abs(integral - total_pers) / total_pers
    assert rel_err < 1e-3, f"{name}: ∫(β0−1)dε={integral:.4f} ≠ Σdeaths={total_pers:.4f}"
    log.info("BY-THE-BOOK CHECK ✓ [%s] N=%d→1, deaths=%d, total_persistence=%.4f "
             "(∫(β0−1)dε rel.err=%.1e)", name, n_start, len(deaths), total_pers, rel_err)
    return {"total_persistence": total_pers, "n_deaths": len(deaths),
            "integral_check_rel_err": rel_err}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original_csv", type=Path, required=True)
    p.add_argument("--synthetic_csv", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
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

    X_orig, X_synth, _ = build_matrix(df_orig, df_synth)
    # PCA fit on REAL only → generator-independent real geometry (fair baseline).
    X_orig_pca, X_synth_pca, _pca = run_pca_orig_only(X_orig, X_synth)
    n_real = len(X_orig_pca)
    X_all = np.vstack([X_orig_pca, X_synth_pca])
    log.info("Real points: %d | synthetic: %d | combined: %d", n_real, len(X_synth_pca), len(X_all))

    # ── Death scales (the persistence content) ──────────────────────────────
    d_real = real_only_deaths(X_orig_pca)                 # real cloud H0
    d_comb = real_only_deaths(X_all)                      # real+synth cloud H0
    d_rel  = real_in_combined_deaths(X_all, n_real)       # real H0 inside combined filtration

    # ── By-the-book verification (asserts) ──────────────────────────────────
    chk_real = verify_betti_curve(d_real, n_real, "real")
    chk_comb = verify_betti_curve(d_comb, len(X_all), "combined")
    chk_rel  = verify_betti_curve(d_rel,  n_real, "real_in_combined")

    tp_real, tp_comb, tp_rel = chk_real["total_persistence"], chk_comb["total_persistence"], chk_rel["total_persistence"]
    # Area BETWEEN the two Betti curves (= L1 distance, since gap ≥ 0)
    # = reduction in real H0 total persistence caused by synthetic.
    tp_reduction = tp_real - tp_rel
    frac_reduced = tp_reduction / tp_real if tp_real > 0 else 0.0
    assert tp_reduction >= -1e-9, f"total persistence reduction negative ({tp_reduction})"

    # ── Summary ─────────────────────────────────────────────────────────────
    summary = {
        "n_real": n_real, "n_synth": len(X_synth_pca),
        "total_persistence_real": round(tp_real, 6),
        "total_persistence_combined": round(tp_comb, 6),
        "total_persistence_real_in_combined": round(tp_rel, 6),
        "total_persistence_reduction": round(tp_reduction, 6),  # area BETWEEN the curves (L1 dist)
        "frac_real_persistence_reduced": round(frac_reduced, 6),
        "betti0_real_start": n_real, "betti0_real_end": 1,
        "n_deaths_real": chk_real["n_deaths"],
        "integral_check_rel_err_real": chk_real["integral_check_rel_err"],
    }
    pd.DataFrame([summary]).to_csv(args.output_dir / "betti_summary.csv", index=False)
    log.info("SUMMARY: %s", summary)

    # death scales table (padded to equal length for tidy CSV)
    max_len = max(len(d_real), len(d_comb), len(d_rel))
    def _pad(a): return np.concatenate([a, np.full(max_len - len(a), np.nan)])
    pd.DataFrame({"death_real": _pad(d_real),
                  "death_combined": _pad(d_comb),
                  "death_real_in_combined": _pad(d_rel)}).to_csv(
        args.output_dir / "betti_deaths.csv", index=False)

    # ── Plot 1: the two strict Betti curves ─────────────────────────────────
    eps_max = max(d_real[-1], d_comb[-1])
    eps = np.linspace(0, eps_max, 2000)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(eps, betti0(d_real, n_real, eps), where="post",
            label=f"real (total pers.={tp_real:.1f})", color="steelblue")
    ax.step(eps, betti0(d_comb, len(X_all), eps), where="post",
            label=f"real+synthetic (total pers.={tp_comb:.1f})", color="tomato")
    ax.set_xlabel("ε  (Euclidean distance in PCA space)")
    ax.set_ylabel(r"$\beta_0(\varepsilon)$  (connected components)")
    ax.set_yscale("log")
    ax.set_title("Strict H₀ Betti curves (exact Vietoris–Rips, via Euclidean MST)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_mpl(fig, args.output_dir, "betti_curves")

    # ── Plot 2: bridging via relative persistence ───────────────────────────
    eps2 = np.linspace(0, max(d_real[-1], d_rel[-1]), 2000)
    b_real = betti0(d_real, n_real, eps2)
    b_rel = betti0(d_rel, n_real, eps2)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(eps2, b_real, where="post", label="real only", color="steelblue")
    ax.step(eps2, b_rel, where="post", label="real (within real+synth)", color="tomato")
    ax.fill_between(eps2, b_rel, b_real, step="post", alpha=0.2, color="tomato",
                    label=f"area between curves = Δtotal H₀ pers. = {tp_reduction:.1f} "
                          f"({frac_reduced:.0%} of real pers.)")
    ax.set_xlabel("ε  (Euclidean distance in PCA space)")
    ax.set_ylabel(r"real $\beta_0(\varepsilon)$")
    ax.set_yscale("log")
    ax.set_title("Synthetic bridging = reduction in real H₀ total persistence")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_mpl(fig, args.output_dir, "betti_bridging")

    log.info("All strict-Betti outputs saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
