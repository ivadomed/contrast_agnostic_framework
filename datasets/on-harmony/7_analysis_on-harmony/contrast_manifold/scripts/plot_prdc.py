#!/usr/bin/env python
"""
PRDC + Vendi score evaluation per (contrast × scanner) cluster.

For each real (modality × scanner_model) group:
  1. Compute real nearest-neighbour statistics → boundary threshold (P95)
     and normalization scale (median NN distance).
  2. Split the FULL synthetic pool into:
       IND – within the real manifold boundary (dist ≤ P95_real_nn)
       OOD – intentional domain-randomization augmentations beyond it
  3. Run compute_prdc(real, IND) to measure core manifold alignment.
     k is chosen adaptively (≤ 3) to handle tiny clusters.
  4. Compute Vendi score on the OOD pool features (cosine kernel) to measure
     the diversity of the exploratory augmentation frontier.
  5. Save prdc_metrics.csv + prdc_metrics.json to --output_dir.
  6. Print a summary ASCII table.

Usage:
  run_job --gpus 0 --slot 0 --wait -- .venv/bin/python \\
    analysis/contrast_manifold/scripts/plot_prdc.py \\
    --original_csv  .../on_harmony_features_normalized_combined_downsampled100_feat_selected.csv \\
    --synthetic_csv .../synthetic_v22_2_lhc_features_normalized_combined_feat_selected.csv \\
    --output_dir    analysis/contrast_manifold/outputs/plots/v22/v22_2_lhc_r1/regional_hist_64/prdc
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from prdc import compute_prdc
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from vendi_score import vendi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

META_COLS = {
    "subject", "session", "modality_id", "modality", "acq_tag",
    "scanner_model", "scanner_vendor", "scanner", "vendor",
    "cohort_category", "cohort", "image_path", "mask_path", "label_map_path",
}

# IND/OOD split: synthetic point is IND if dist_to_real ≤ P{THRESHOLD_PCT} of real NN dists
THRESHOLD_PCT = 95
# Maximum k for PRDC (reduced adaptively for small clusters)
MAX_K = 3
# Minimum real samples to attempt PRDC at all
MIN_REAL = MAX_K + 1
# Minimum IND synthetic samples: prdc uses kth=nearest_k+1 internally, need n_ind >= k+2
MIN_IND  = MAX_K + 2
# Vendi eigendecomp cost spikes at n≈250 on this machine's LAPACK stack; stay below that
MAX_OOD_VENDI = 150


def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and not c.startswith("diagnostics_")]


def _extract_modality(df: pd.DataFrame) -> pd.Series:
    """Base modality from modality_id (e.g. 'T1w' from 'T1w_acq-...')."""
    col = next((c for c in ("modality", "modality_id") if c in df.columns), None)
    if col is None:
        return pd.Series(["unknown"] * len(df), index=df.index)
    return df[col].str.split("_").str[0].fillna("unknown")


def _extract_scanner(df: pd.DataFrame) -> pd.Series:
    col = next((c for c in ("scanner", "scanner_model") if c in df.columns), None)
    if col is None:
        return pd.Series(["unknown"] * len(df), index=df.index)
    return df[col].fillna("unknown")


# ── Per-group geometry ────────────────────────────────────────────────────────

def _real_nn_stats(real: np.ndarray) -> tuple[float, float]:
    """
    Returns (median_nn_dist, p95_nn_dist) of nearest-neighbour distances
    within the real cluster.  Used as normalization scale and IND boundary.
    """
    if len(real) < 2:
        return 1.0, 1.0
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto", metric="euclidean").fit(real)
    dists, _ = nn.kneighbors(real)
    nn_dists = dists[:, 1]  # exclude self (nearest is self at d=0)
    return float(np.median(nn_dists)), float(np.percentile(nn_dists, THRESHOLD_PCT))


def _split_ind_ood_precomputed(
    synth: np.ndarray,
    dist_to_real: np.ndarray,   # shape (n_synth,) — precomputed min dist to this group's real
    threshold: float,
    median_nn: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split synthetic into IND (dist_to_real ≤ threshold) and OOD.
    OOD distances are returned normalized by median_nn for cross-group comparability.
    dist_to_real is the precomputed per-synth minimum distance to this group's real points.
    """
    ind_mask = dist_to_real <= threshold
    ood_mask = ~ind_mask
    norm = max(median_nn, 1e-8)
    return synth[ind_mask], synth[ood_mask], dist_to_real[ood_mask] / norm


# ── Per-group evaluation ──────────────────────────────────────────────────────

def _run_group(
    label: str,
    real: np.ndarray,
    synth: np.ndarray,
    dist_to_real: np.ndarray,   # precomputed min dist from each synth point to this group
) -> dict:
    result: dict = {
        "group":        label,
        "n_real":       len(real),
        "n_synth_total": len(synth),
    }

    # Real geometry
    median_nn, p95_nn = _real_nn_stats(real)
    result["real_median_nn_dist"] = round(median_nn, 6)
    result["real_p95_nn_dist"]    = round(p95_nn,    6)

    if len(synth) == 0:
        result.update(_empty_metrics())
        return result

    # IND / OOD split using precomputed distances
    ind, ood, ood_norm_dists = _split_ind_ood_precomputed(synth, dist_to_real, p95_nn, median_nn)
    n_ind, n_ood = len(ind), len(ood)
    result["n_ind"] = n_ind
    result["n_ood"] = n_ood
    result["ood_pct"] = round(100.0 * n_ood / len(synth), 2)

    # PRDC on (real, IND)
    # prdc internally uses kth = nearest_k+1 via np.partition, so need n_ind >= k+2
    k = min(MAX_K, len(real) - 1, n_ind - 2) if n_ind > 1 else 0
    if len(real) >= MIN_REAL and n_ind >= MIN_IND and k >= 1:
        try:
            m = compute_prdc(real_features=real, fake_features=ind, nearest_k=k)
            result.update({
                "prdc_k":    k,
                "precision": round(m["precision"], 6),
                "recall":    round(m["recall"],    6),
                "density":   round(m["density"],   6),
                "coverage":  round(m["coverage"],  6),
            })
        except Exception as exc:
            log.warning("[%s] PRDC failed (k=%d): %s", label, k, exc)
            result.update({"prdc_k": k, **_empty_prdc()})
    else:
        log.info(
            "[%s] Skipping PRDC: n_real=%d  n_ind=%d  k=%d",
            label, len(real), n_ind, k,
        )
        result.update({"prdc_k": k, **_empty_prdc()})

    # OOD expansion stats
    if n_ood >= 2:
        result["ood_mean_norm_dist"] = round(float(np.mean(ood_norm_dists)), 6)
        result["ood_var_norm_dist"]  = round(float(np.var(ood_norm_dists)),  6)
        try:
            # score_X normalises rows (L2) then computes X@X.T → cosine kernel.
            # Cap subsample: eigendecomp is O(n³), 500 pts is sufficient for diversity.
            ood_v = ood
            if len(ood) > MAX_OOD_VENDI:
                rng   = np.random.default_rng(42)
                idx   = rng.choice(len(ood), MAX_OOD_VENDI, replace=False)
                ood_v = ood[idx]
            result["vendi_diversity_score"] = round(float(vendi.score_X(ood_v)), 6)
        except Exception as exc:
            log.warning("[%s] Vendi failed: %s", label, exc)
            result["vendi_diversity_score"] = None
    elif n_ood == 1:
        result["ood_mean_norm_dist"]    = round(float(ood_norm_dists[0]), 6)
        result["ood_var_norm_dist"]     = 0.0
        result["vendi_diversity_score"] = None
    else:
        result.update({"ood_mean_norm_dist": None, "ood_var_norm_dist": None,
                        "vendi_diversity_score": None})

    return result


def _empty_prdc() -> dict:
    return {"precision": None, "recall": None, "density": None, "coverage": None}


def _empty_metrics() -> dict:
    return {
        "n_ind": 0, "n_ood": 0, "ood_pct": 0.0,
        "prdc_k": 0, **_empty_prdc(),
        "ood_mean_norm_dist": None, "ood_var_norm_dist": None,
        "vendi_diversity_score": None,
    }


# ── ASCII table ───────────────────────────────────────────────────────────────

def _fmt(v, spec=".3f", na="—") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na
    return format(v, spec)


def _print_table(rows: list[dict]) -> None:
    COLS = [
        ("Contrast × Scanner",  "group",                "<36", str),
        ("N real",              "n_real",               ">7",  str),
        ("N IND",               "n_ind",                ">7",  str),
        ("N OOD",               "n_ood",                ">7",  str),
        ("OOD %",               "ood_pct",              ">7",  lambda v: _fmt(v, ".1f") + "%"),
        ("k",                   "prdc_k",               ">3",  str),
        ("Precision",           "precision",            ">10", lambda v: _fmt(v)),
        ("Recall",              "recall",               ">8",  lambda v: _fmt(v)),
        ("Density",             "density",              ">9",  lambda v: _fmt(v)),
        ("Coverage",            "coverage",             ">10", lambda v: _fmt(v)),
        ("OOD μ dist",          "ood_mean_norm_dist",   ">10", lambda v: _fmt(v)),
        ("OOD σ² dist",         "ood_var_norm_dist",    ">10", lambda v: _fmt(v)),
        ("Vendi",               "vendi_diversity_score",">7",  lambda v: _fmt(v, ".2f")),
    ]

    widths = [max(len(h), int(align.lstrip("<>^"))) for h, _, align, _ in COLS]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt_row(vals):
        parts = []
        for (_, _, align, _), w, v in zip(COLS, widths, vals):
            parts.append(f" {v:{align[0]}{w}} ")
        return "|" + "|".join(parts) + "|"

    print(sep)
    print(fmt_row([h for h, *_ in COLS]))
    print(sep)
    for r in rows:
        vals = [fn(r.get(key, None)) for _, key, _, fn in COLS]
        print(fmt_row(vals))
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original_csv",  type=Path, required=True)
    p.add_argument("--synthetic_csv", type=Path, required=True)
    p.add_argument("--output_dir",    type=Path, required=True)
    p.add_argument("--pca-variance",  type=float, default=0.90,
                   help="Fraction of variance retained by PCA before PRDC (default: 0.90)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_real  = pd.read_csv(args.original_csv)
    df_synth = pd.read_csv(args.synthetic_csv)

    fcols = [c for c in _feat_cols(df_real) if c in set(_feat_cols(df_synth))]
    if not fcols:
        log.error("No common feature columns between real and synthetic CSVs.")
        return
    pca_var = args.pca_variance
    log.info("Real: %d rows | Synth: %d rows | %d raw dims (→ PCA %.0f%%)",
             len(df_real), len(df_synth), len(fcols), pca_var * 100)

    # Impute any residual NaNs (should be none after normalization, but be safe)
    X_synth  = df_synth[fcols].fillna(0.0).values.astype(np.float32)
    df_real  = df_real.copy()
    X_real_all = df_real[fcols].fillna(0.0).values.astype(np.float32)

    df_real["_mod"] = _extract_modality(df_real)
    df_real["_sc"]  = _extract_scanner(df_real)

    # PCA projection: fit on real, transform both real and synth.
    pca = PCA(n_components=pca_var, svd_solver="full", random_state=42)
    pca.fit(X_real_all)
    n_pc = pca.n_components_
    log.info("PCA: %d → %d dims (%.0f%% variance, fit on real)", len(fcols), n_pc, pca_var * 100)
    X_real_all = pca.transform(X_real_all).astype(np.float32)
    X_synth    = pca.transform(X_synth).astype(np.float32)

    # Batch-compute synth→real_all distance matrix once (n_synth × n_real_all).
    # Avoids redundant brute-force NN queries in each group.
    log.info("Pre-computing synth→real distance matrix (%d × %d × %d dims) …",
             len(X_synth), len(X_real_all), n_pc)
    t0 = __import__("time").time()
    from sklearn.metrics import euclidean_distances
    D_full = euclidean_distances(X_synth, X_real_all)   # (n_synth, n_real_all)
    log.info("  Distance matrix computed in %.1fs", __import__("time").time() - t0)

    real_idx = df_real.reset_index(drop=True).index.to_numpy()   # 0..n_real-1

    groups = df_real.groupby(["_mod", "_sc"])
    log.info("%d (contrast × scanner) groups to evaluate", len(groups))

    all_results: list[dict] = []
    for (mod, sc), sub in sorted(groups):
        label  = f"{mod} × {sc}"
        g_idx  = sub.index.to_numpy()                       # row indices in df_real
        X_real = X_real_all[g_idx]
        # Per-synth min distance to this group's real points (column slice of D_full)
        dist_to_grp = D_full[:, g_idx].min(axis=1)          # (n_synth,)
        log.info("  %-40s  real=%d  synth=%d", label, len(X_real), len(X_synth))
        all_results.append(_run_group(label, X_real, X_synth, dist_to_grp))

    # ── Save artifacts ────────────────────────────────────────────────────────
    csv_path  = args.output_dir / "prdc_metrics.csv"
    json_path = args.output_dir / "prdc_metrics.json"

    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(all_results, indent=2, default=str))
    log.info("Saved → %s", csv_path)
    log.info("Saved → %s", json_path)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  PRDC + Vendi — per contrast × scanner  |  "
          f"IND boundary: P{THRESHOLD_PCT} real NN dist (normalised by median)  |  "
          f"max k={MAX_K}")
    print(f"{'='*140}")
    _print_table(all_results)
    print()


if __name__ == "__main__":
    main()
