#!/usr/bin/env python
"""
Z-score normalize original and synthetic features using a StandardScaler fit
on the COMBINED (original + synthetic) distribution.

This gives a fair joint embedding: neither dataset is treated as the reference.
Deviations from 0 in the normalized space reflect distance from the *joint*
mean, so the UMAP and PCA can reveal overlap or separation without pre-baking
the original distribution as ground truth.

Steps:
  1. Load raw original features (all rows) + raw synthetic features.
  2. Align to common feature columns.
  3. Impute NaNs with column mean (on combined data).
  4. Fit StandardScaler on the stacked matrix.
  5. Transform original → downsample to --n_per_contrast per modality
     (stratified by scanner) → save.
  6. Transform synthetic → save (no downsampling).

Inputs:
  analysis/contrast_manifold/outputs/data/original/roi_mask/on_harmony_features.csv
  analysis/contrast_manifold/outputs/data/synthetic_v19/roi_mask/synthetic_v19_features.csv

Outputs (never overwrites existing files):
  analysis/contrast_manifold/outputs/data/original/roi_mask/
      on_harmony_features_normalized_combined_downsampled100.csv
  analysis/contrast_manifold/outputs/data/synthetic_v19/roi_mask/
      synthetic_v19_features_normalized_combined.csv

Usage:
  run_job --gpus 0 --slot 0 --wait -- .venv/bin/python analysis/contrast_manifold/scripts/normalize_combined.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

META_COLS = [
    "subject", "session", "modality_id", "acq_tag",
    "scanner_model", "scanner_vendor", "cohort_category",
    "image_path", "mask_path", "label_map_path",
]

# WM-ratio normalisation
_WM_REF_COLS = [
    "left_cerebral_white_matter_firstorder_Mean",
    "right_cerebral_white_matter_firstorder_Mean",
]
# Suffixes whose values are in raw intensity units → divide by wm_ref
_INTENSITY_SUFFIXES = frozenset({
    "10Percentile", "90Percentile", "Maximum", "Mean",
    "MeanAbsoluteDeviation", "Median", "Minimum", "Range",
    "RobustMeanAbsoluteDeviation", "RootMeanSquared", "InterquartileRange",
})
# Suffixes in intensity² units → divide by wm_ref²
_INTENSITY2_SUFFIXES = frozenset({"Variance", "Energy", "TotalEnergy"})
# Dimensionless (Entropy, Skewness, Kurtosis, Uniformity) → unchanged


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and not c.startswith("diagnostics_")]


def apply_wm_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Divide intensity features by the mean of left+right cerebral WM Mean per scan."""
    ref_cols = [c for c in _WM_REF_COLS if c in df.columns]
    if not ref_cols:
        raise ValueError(
            f"WM reference columns not found. Need at least one of {_WM_REF_COLS}. "
            "This flag only works with synthseg_mask_31 features."
        )
    wm_ref = df[ref_cols].mean(axis=1).values  # (n,)
    if (wm_ref <= 0).any():
        n_bad = (wm_ref <= 0).sum()
        log.warning("WM reference ≤ 0 in %d rows — those rows will produce NaN/inf", n_bad)

    df = df.copy()
    for col in feature_cols(df):
        if "firstorder_" not in col:
            continue
        suffix = col.split("firstorder_")[-1]
        if suffix in _INTENSITY_SUFFIXES:
            df[col] = df[col].values / wm_ref
        elif suffix in _INTENSITY2_SUFFIXES:
            df[col] = df[col].values / (wm_ref ** 2)
    return df


def get_family(name: str) -> str:
    for fam in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape"):
        if fam in name.lower():
            return fam
    return "other"


def apply_feature_filter(feat_names: list[str], config: dict) -> tuple[list[str], list[str]]:
    exclude_families = set(config.get("exclude_families", []))
    exclude_features = set(config.get("exclude_features", []))
    kept, excluded = [], []
    for name in feat_names:
        if get_family(name) in exclude_families or name in exclude_features:
            excluded.append(name)
        else:
            kept.append(name)
    return kept, excluded


def align_common_features(
    df_a: pd.DataFrame, df_b: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """Return (common_feat_cols, meta_cols_a, meta_cols_b)."""
    fa = set(feature_cols(df_a))
    fb = set(feature_cols(df_b))
    common = sorted(fa & fb)
    if not common:
        raise ValueError("No common feature columns between original and synthetic CSVs.")
    log.info("Common feature columns: %d (orig has %d, synth has %d)", len(common), len(fa), len(fb))
    return common


def downsample_stratified(
    df: pd.DataFrame,
    group_col: str,
    strat_col: str,
    n_max: int,
    seed: int = 42,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for group_val, grp in df.groupby(group_col):
        if len(grp) <= n_max:
            parts.append(grp)
            continue
        strat_counts = grp[strat_col].value_counts()
        total = len(grp)
        sampled: list[pd.DataFrame] = []
        budget = n_max
        for stratum, cnt in strat_counts.items():
            alloc = max(1, round(cnt / total * n_max))
            alloc = min(alloc, cnt, budget)
            sampled.append(grp[grp[strat_col] == stratum].sample(alloc, random_state=seed))
            budget -= alloc
            if budget <= 0:
                break
        sampled_df = pd.concat(sampled, ignore_index=True)
        parts.append(sampled_df)
        log.info("  %-35s %d → %d rows (across %d scanners)",
                 group_val, len(grp), len(sampled_df), len(strat_counts))
    return pd.concat(parts, ignore_index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--original_csv", type=Path,
        default=Path("analysis/contrast_manifold/outputs/data/original/roi_mask/on_harmony_features.csv"),
    )
    p.add_argument(
        "--synthetic_csv", type=Path,
        default=Path("analysis/contrast_manifold/outputs/data/synthetic_v19/roi_mask/synthetic_v19_features.csv"),
    )
    p.add_argument(
        "--output_original", type=Path,
        default=Path("analysis/contrast_manifold/outputs/data/original/roi_mask/"
                     "on_harmony_features_normalized_combined_downsampled100.csv"),
    )
    p.add_argument(
        "--output_synthetic", type=Path,
        default=Path("analysis/contrast_manifold/outputs/data/synthetic_v19/roi_mask/"
                     "synthetic_v19_features_normalized_combined.csv"),
    )
    p.add_argument("--n_per_contrast", type=int, default=100)
    p.add_argument(
        "--feature_config", type=Path, default=None,
        help="YAML feature selection config. When set, excluded features are dropped "
             "before fitting the scaler, and outputs are renamed with '_feat_selected' suffix.",
    )
    p.add_argument(
        "--wm_normalize", action="store_true", default=False,
        help="Divide intensity features by the mean cerebral WM intensity per scan before "
             "fitting the StandardScaler. Outputs are renamed with '_wm_ratio' suffix. "
             "Only valid for synthseg_mask_31 features (requires WM region columns).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-rename outputs for optional flags (applied in order so suffixes stack)
    if args.wm_normalize:
        def _add_wm(p: Path) -> Path:
            return p.with_name(p.stem + "_wm_ratio" + p.suffix)
        args.output_original = _add_wm(args.output_original)
        args.output_synthetic = _add_wm(args.output_synthetic)

    if args.feature_config:
        def _add_suffix(p: Path) -> Path:
            return p.with_name(p.stem + "_feat_selected" + p.suffix)
        args.output_original = _add_suffix(args.output_original)
        args.output_synthetic = _add_suffix(args.output_synthetic)

    for src in (args.original_csv, args.synthetic_csv):
        if not src.exists():
            raise FileNotFoundError(src)
    for dst in (args.output_original, args.output_synthetic):
        if dst == args.original_csv or dst == args.synthetic_csv:
            raise ValueError(f"Output path must differ from input: {dst}")

    # ── Load ──────────────────────────────────────────────────────────────
    log.info("Loading original features …")
    df_orig  = pd.read_csv(args.original_csv)
    log.info("  %d rows × %d cols", *df_orig.shape)

    log.info("Loading synthetic features …")
    df_synth = pd.read_csv(args.synthetic_csv)
    log.info("  %d rows × %d cols", *df_synth.shape)

    # ── WM-ratio normalisation (per-scan, before any cross-scan scaling) ──
    if args.wm_normalize:
        log.info("Applying WM-ratio normalisation …")
        df_orig  = apply_wm_normalization(df_orig)
        df_synth = apply_wm_normalization(df_synth)
        log.info("  done")

    # ── Feature selection ─────────────────────────────────────────────────
    if args.feature_config:
        with open(args.feature_config) as f:
            cfg = yaml.safe_load(f) or {}
        all_feat = sorted(set(feature_cols(df_orig)) | set(feature_cols(df_synth)))
        kept, excluded_list = apply_feature_filter(all_feat, cfg)
        exclude_set = set(excluded_list)
        df_orig  = df_orig.drop(columns=[c for c in exclude_set if c in df_orig.columns])
        df_synth = df_synth.drop(columns=[c for c in exclude_set if c in df_synth.columns])
        log.info("Feature filter: kept %d, excluded %d", len(kept), len(excluded_list))
        # Write selection log alongside output
        args.output_original.parent.mkdir(parents=True, exist_ok=True)
        log_path = args.output_original.parent / "feature_selection_log.txt"
        lines = [
            "Feature selection log",
            f"Config: {args.feature_config}",
            f"Kept:     {len(kept)} features",
            f"Excluded: {len(excluded_list)} features",
            "", "=== EXCLUDED ===",
        ] + [f"  {n}  [{get_family(n)}]" for n in sorted(excluded_list)] + [
            "", "=== KEPT ===",
        ] + [f"  {n}  [{get_family(n)}]" for n in sorted(kept)]
        log_path.write_text("\n".join(lines) + "\n")
        log.info("Feature log → %s", log_path)

    common = align_common_features(df_orig, df_synth)

    X_orig  = df_orig[common].to_numpy(dtype=np.float64)
    X_synth = df_synth[common].to_numpy(dtype=np.float64)
    X_all   = np.vstack([X_orig, X_synth])
    n_orig  = len(df_orig)

    # ── Drop all-NaN columns ──────────────────────────────────────────────
    nan_mask = np.isnan(X_all).mean(axis=0) < 1.0
    X_all    = X_all[:, nan_mask]
    common   = [c for c, k in zip(common, nan_mask) if k]
    log.info("After dropping all-NaN cols: %d features", len(common))

    # ── Impute on combined ────────────────────────────────────────────────
    imputer = SimpleImputer(strategy="mean")
    X_all   = imputer.fit_transform(X_all)

    # ── Drop zero-variance columns ────────────────────────────────────────
    var_mask = X_all.var(axis=0) > 0
    X_all    = X_all[:, var_mask]
    common   = [c for c, k in zip(common, var_mask) if k]
    log.info("After dropping zero-variance cols: %d features", len(common))

    # ── Fit scaler on COMBINED data ───────────────────────────────────────
    scaler  = StandardScaler()
    X_norm  = scaler.fit_transform(X_all)

    X_orig_norm  = X_norm[:n_orig]
    X_synth_norm = X_norm[n_orig:]

    log.info(
        "Combined scaler stats:\n"
        "  original  — mean=%.4f  std=%.4f\n"
        "  synthetic — mean=%.4f  std=%.4f\n"
        "  combined  — mean=%.4f  std=%.4f",
        X_orig_norm.mean(),  X_orig_norm.std(),
        X_synth_norm.mean(), X_synth_norm.std(),
        X_norm.mean(),       X_norm.std(),
    )

    # ── Rebuild DataFrames ────────────────────────────────────────────────
    meta_orig  = [c for c in META_COLS if c in df_orig.columns]
    meta_synth = [c for c in META_COLS if c in df_synth.columns]

    df_orig_norm  = pd.concat([
        df_orig[meta_orig].reset_index(drop=True),
        pd.DataFrame(X_orig_norm, columns=common),
    ], axis=1)
    df_synth_norm = pd.concat([
        df_synth[meta_synth].reset_index(drop=True),
        pd.DataFrame(X_synth_norm, columns=common),
    ], axis=1)

    # ── Downsample original ───────────────────────────────────────────────
    log.info("Downsampling original to %d per contrast …", args.n_per_contrast)
    df_orig_down = downsample_stratified(
        df_orig_norm,
        group_col="modality_id",
        strat_col="scanner_model",
        n_max=args.n_per_contrast,
    )
    log.info("Downsampled original: %d rows", len(df_orig_down))

    # ── Save ──────────────────────────────────────────────────────────────
    args.output_original.parent.mkdir(parents=True, exist_ok=True)
    args.output_synthetic.parent.mkdir(parents=True, exist_ok=True)

    df_orig_down.to_csv(args.output_original, index=False)
    log.info("Saved original → %s", args.output_original)

    df_synth_norm.to_csv(args.output_synthetic, index=False)
    log.info("Saved synthetic → %s", args.output_synthetic)


if __name__ == "__main__":
    main()
