#!/usr/bin/env python
"""
Run the full contrast-manifold analysis for every synthetic version.

Outputs land in a clean hierarchy:
  plots/<major_version>/<run_name>/<mask_type>/pca/
  plots/<major_version>/<run_name>/<mask_type>/umap/
  plots/<major_version>/<run_name>/<mask_type>/contrast_clustering/
  plots/<major_version>/<run_name>/<mask_type>/feature_analysis/

where <mask_type> is one of:  roi_mask  |  synthseg_mask_31  |  synthseg_mask_31_WM_ratio

Usage (all versions, roi_mask, 4 parallel slots):
  for rank in 0 1 2 3; do
    set_slot $rank .venv/bin/python analysis/contrast_manifold/scripts/run_all_analysis.py \\
      --rank $rank --world-size 4 > /tmp/analysis_rank${rank}.log 2>&1 < /dev/null &
  done

Usage (all versions, synthseg_mask_31):
  for rank in 0 1 2 3; do
    set_slot $rank .venv/bin/python analysis/contrast_manifold/scripts/run_all_analysis.py \\
      --mask-type synthseg_mask_31 \\
      --rank $rank --world-size 4 > /tmp/analysis_synthseg_rank${rank}.log 2>&1 < /dev/null &
  done

Usage (single version):
  .venv/bin/python analysis/contrast_manifold/scripts/run_all_analysis.py --only v21_5_r3
  .venv/bin/python analysis/contrast_manifold/scripts/run_all_analysis.py --only v19_c_r1 --mask-type synthseg_mask_31
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS      = Path(__file__).parent
DATA_ROOT    = PROJECT_ROOT / "analysis" / "contrast_manifold" / "outputs" / "data"
PLOTS_ROOT   = PROJECT_ROOT / "analysis" / "contrast_manifold" / "outputs" / "plots"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Original CSV registry ────────────────────────────────────────────────────
# Maps mask_type → (norm_csv, raw_csv) for the original ON-Harmony data.
ORIG_CSVS: dict[str, tuple[Path, Path]] = {
    "roi_mask": (
        DATA_ROOT / "original" / "roi_mask"
        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        DATA_ROOT / "original" / "roi_mask" / "on_harmony_features.csv",
    ),
    "synthseg_mask_31": (
        DATA_ROOT / "original" / "synthseg_mask_31"
        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        DATA_ROOT / "original" / "synthseg_mask_31" / "on_harmony_features.csv",
    ),
    "synthseg_mask_31_WM_ratio": (
        DATA_ROOT / "original" / "synthseg_mask_31"
        / "on_harmony_features_normalized_combined_downsampled100_wm_ratio_feat_selected.csv",
        DATA_ROOT / "original" / "synthseg_mask_31" / "on_harmony_features.csv",
    ),
    "synthseg_mask_7": (
        DATA_ROOT / "original" / "synthseg_mask_7"
        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        DATA_ROOT / "original" / "synthseg_mask_7" / "on_harmony_features.csv",
    ),
    "curia_embeddings": (
        DATA_ROOT / "original" / "curia_embeddings"
        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        DATA_ROOT / "original" / "curia_embeddings" / "on_harmony_features.csv",
    ),
    "histogram_256": (
        DATA_ROOT / "original" / "histogram_256"
        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        DATA_ROOT / "original" / "histogram_256" / "on_harmony_features.csv",
    ),
    "regional_hist_64": (
        DATA_ROOT / "original" / "regional_hist_64"
        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        DATA_ROOT / "original" / "regional_hist_64" / "on_harmony_features.csv",
    ),
    "hog_972": (
        DATA_ROOT / "original" / "hog_972"
        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        DATA_ROOT / "original" / "hog_972" / "on_harmony_features.csv",
    ),
    "regional_hist_13_64": (
        DATA_ROOT / "original" / "regional_hist_13_64"
        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        DATA_ROOT / "original" / "regional_hist_13_64" / "on_harmony_features.csv",
    ),
    "hog3d_512": (
        DATA_ROOT / "original" / "hog3d_512"
        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        DATA_ROOT / "original" / "hog3d_512" / "on_harmony_features.csv",
    ),
}

# ─── Version registry ─────────────────────────────────────────────────────────
# Each entry: (major_version, run_name, norm_csvs, raw_csvs)
# norm_csvs / raw_csvs are dicts keyed by mask_type.
# Add a new mask_type key to each entry when the corresponding features are ready.
VERSIONS: list[tuple[str, str, dict[str, Path], dict[str, Path]]] = [
    (
        "v19", "v19_r1",
        {
            "roi_mask": DATA_ROOT / "synthetic_v19" / "roi_mask"
                        / "synthetic_v19_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v19" / "synthseg_mask_31"
                        / "synthetic_v19_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v19" / "synthseg_mask_31"
                        / "synthetic_v19_features_normalized_combined_wm_ratio_feat_selected.csv",
        },
        {
            "roi_mask": DATA_ROOT / "synthetic_v19" / "roi_mask" / "synthetic_v19_features.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v19" / "synthseg_mask_31" / "synthetic_v19_features.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v19" / "synthseg_mask_31" / "synthetic_v19_features.csv",
        },
    ),
    (
        "v19", "v19_b_r1",
        {
            "roi_mask": DATA_ROOT / "synthetic_v19_b" / "roi_mask"
                        / "synthetic_v19_b_features_normalized_feat_selected.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v19_b" / "synthseg_mask_31"
                        / "synthetic_v19_b_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v19_b" / "synthseg_mask_31"
                        / "synthetic_v19_b_features_normalized_combined_wm_ratio_feat_selected.csv",
        },
        {
            "roi_mask": DATA_ROOT / "synthetic_v19_b" / "roi_mask" / "synthetic_v19_b_features.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v19_b" / "synthseg_mask_31" / "synthetic_v19_b_features.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v19_b" / "synthseg_mask_31" / "synthetic_v19_b_features.csv",
        },
    ),
    (
        "v19", "v19_c_r1",
        {
            "roi_mask": DATA_ROOT / "synthetic_v19_c" / "roi_mask"
                        / "synthetic_v19_c_features_normalized_feat_selected.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v19_c" / "synthseg_mask_31"
                        / "synthetic_v19_c_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v19_c" / "synthseg_mask_31"
                        / "synthetic_v19_c_features_normalized_combined_wm_ratio_feat_selected.csv",
            "synthseg_mask_7": DATA_ROOT / "synthetic_v19_c" / "synthseg_mask_7"
                        / "synthetic_v19_c_features_normalized_combined_feat_selected.csv",
            "curia_embeddings": DATA_ROOT / "synthetic_v19_c" / "curia_embeddings"
                        / "synthetic_v19_c_features_normalized_combined_feat_selected.csv",
            "histogram_256": DATA_ROOT / "synthetic_v19_c" / "histogram_256"
                        / "synthetic_v19_c_features_normalized_combined_feat_selected.csv",
            "regional_hist_64": DATA_ROOT / "synthetic_v19_c" / "regional_hist_64"
                        / "synthetic_v19_c_features_normalized_combined_feat_selected.csv",
            "hog_972": DATA_ROOT / "synthetic_v19_c" / "hog_972"
                        / "synthetic_v19_c_features_normalized_combined_feat_selected.csv",
            "hog3d_512": DATA_ROOT / "synthetic_v19_c" / "hog3d_512"
                        / "synthetic_v19_c_features_normalized_combined_feat_selected.csv",
            "regional_hist_13_64": DATA_ROOT / "synthetic_v19_c" / "regional_hist_13_64"
                        / "synthetic_v19_c_features_normalized_combined_feat_selected.csv",
        },
        {
            "roi_mask": DATA_ROOT / "synthetic_v19_c" / "roi_mask" / "synthetic_v19_c_features.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v19_c" / "synthseg_mask_31" / "synthetic_v19_c_features.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v19_c" / "synthseg_mask_31" / "synthetic_v19_c_features.csv",
            "synthseg_mask_7": DATA_ROOT / "synthetic_v19_c" / "synthseg_mask_7" / "synthetic_v19_c_features.csv",
            "curia_embeddings": DATA_ROOT / "synthetic_v19_c" / "curia_embeddings" / "synthetic_v19_c_features.csv",
            "histogram_256": DATA_ROOT / "synthetic_v19_c" / "histogram_256" / "synthetic_v19_c_features.csv",
            "regional_hist_64": DATA_ROOT / "synthetic_v19_c" / "regional_hist_64" / "synthetic_v19_c_features.csv",
            "hog_972": DATA_ROOT / "synthetic_v19_c" / "hog_972" / "synthetic_v19_c_features.csv",
            "hog3d_512": DATA_ROOT / "synthetic_v19_c" / "hog3d_512" / "synthetic_v19_c_features.csv",
            "regional_hist_13_64": DATA_ROOT / "synthetic_v19_c" / "regional_hist_13_64" / "synthetic_v19_c_features.csv",
        },
    ),
    (
        "v20", "v20_r4",
        {
            "roi_mask": DATA_ROOT / "synthetic_v20" / "roi_mask"
                        / "synthetic_v20_features_normalized_feat_selected.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v20" / "synthseg_mask_31"
                        / "synthetic_v20_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v20" / "synthseg_mask_31"
                        / "synthetic_v20_features_normalized_combined_wm_ratio_feat_selected.csv",
        },
        {
            "roi_mask": DATA_ROOT / "synthetic_v20" / "roi_mask" / "synthetic_v20_features.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v20" / "synthseg_mask_31" / "synthetic_v20_features.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v20" / "synthseg_mask_31" / "synthetic_v20_features.csv",
        },
    ),
    (
        "v20", "v20_r5",
        {
            "roi_mask": DATA_ROOT / "synthetic_v20_r5" / "roi_mask"
                        / "synthetic_v20_r5_features_normalized_feat_selected.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v20_r5" / "synthseg_mask_31"
                        / "synthetic_v20_r5_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v20_r5" / "synthseg_mask_31"
                        / "synthetic_v20_r5_features_normalized_combined_wm_ratio_feat_selected.csv",
        },
        {
            "roi_mask": DATA_ROOT / "synthetic_v20_r5" / "roi_mask" / "synthetic_v20_r5_features.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v20_r5" / "synthseg_mask_31" / "synthetic_v20_r5_features.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v20_r5" / "synthseg_mask_31" / "synthetic_v20_r5_features.csv",
        },
    ),
    (
        "v20", "v20_r5_blur05",
        {
            "roi_mask": DATA_ROOT / "synthetic_v20_r5" / "roi_mask"
                        / "synthetic_v20_r5_blur05_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v20_r5" / "synthseg_mask_31"
                        / "synthetic_v20_r5_blur05_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v20_r5" / "synthseg_mask_31"
                        / "synthetic_v20_r5_blur05_features_normalized_combined_wm_ratio_feat_selected.csv",
        },
        {
            "roi_mask": DATA_ROOT / "synthetic_v20_r5" / "roi_mask" / "synthetic_v20_r5_blur05_features.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v20_r5" / "synthseg_mask_31" / "synthetic_v20_r5_blur05_features.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v20_r5" / "synthseg_mask_31" / "synthetic_v20_r5_blur05_features.csv",
        },
    ),
    (
        "v21", "v21_r2",
        {
            "roi_mask": DATA_ROOT / "synthetic_v21_r2" / "roi_mask"
                        / "synthetic_v21_r2_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v21_r2" / "synthseg_mask_31"
                        / "synthetic_v21_r2_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v21_r2" / "synthseg_mask_31"
                        / "synthetic_v21_r2_features_normalized_combined_wm_ratio_feat_selected.csv",
        },
        {
            "roi_mask": DATA_ROOT / "synthetic_v21_r2" / "roi_mask" / "synthetic_v21_r2_features.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v21_r2" / "synthseg_mask_31" / "synthetic_v21_r2_features.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v21_r2" / "synthseg_mask_31" / "synthetic_v21_r2_features.csv",
        },
    ),
    (
        "v21", "v21_1_r1",
        {
            "roi_mask": DATA_ROOT / "synthetic_v21_1_r1" / "roi_mask"
                        / "synthetic_v21_1_r1_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v21_1_r1" / "synthseg_mask_31"
                        / "synthetic_v21_1_r1_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v21_1_r1" / "synthseg_mask_31"
                        / "synthetic_v21_1_r1_features_normalized_combined_wm_ratio_feat_selected.csv",
        },
        {
            "roi_mask": DATA_ROOT / "synthetic_v21_1_r1" / "roi_mask" / "synthetic_v21_1_r1_features.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v21_1_r1" / "synthseg_mask_31" / "synthetic_v21_1_r1_features.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v21_1_r1" / "synthseg_mask_31" / "synthetic_v21_1_r1_features.csv",
        },
    ),
    (
        "v21", "v21_5_r3",
        {
            "roi_mask": DATA_ROOT / "synthetic_v21_5_r3" / "roi_mask"
                        / "synthetic_v21_5_r3_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v21_5_r3" / "synthseg_mask_31"
                        / "synthetic_v21_5_r3_features_normalized_combined_feat_selected.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v21_5_r3" / "synthseg_mask_31"
                        / "synthetic_v21_5_r3_features_normalized_combined_wm_ratio_feat_selected.csv",
        },
        {
            "roi_mask": DATA_ROOT / "synthetic_v21_5_r3" / "roi_mask" / "synthetic_v21_5_r3_features.csv",
            "synthseg_mask_31": DATA_ROOT / "synthetic_v21_5_r3" / "synthseg_mask_31" / "synthetic_v21_5_r3_features.csv",
            "synthseg_mask_31_WM_ratio": DATA_ROOT / "synthetic_v21_5_r3" / "synthseg_mask_31" / "synthetic_v21_5_r3_features.csv",
        },
    ),
    # ── LHC runs: v19_c + v22_1 with Sobol quasi-random parameter sampling ──
    # Optional 5th element: per-mask-type override for the normalized ORIGINAL CSV
    # (each normalize_combined run fits a new scaler on orig+synth, so we store
    #  the version-specific normalized original alongside the synthetic CSV).
    (
        "v19", "v19_c_lhc_r1",
        {
            "curia_embeddings": DATA_ROOT / "synthetic_v19_c_lhc" / "curia_embeddings"
                        / "synthetic_v19_c_lhc_features_normalized_combined_feat_selected.csv",
            "histogram_256": DATA_ROOT / "synthetic_v19_c_lhc" / "histogram_256"
                        / "synthetic_v19_c_lhc_features_normalized_combined_feat_selected.csv",
            "regional_hist_64": DATA_ROOT / "synthetic_v19_c_lhc" / "regional_hist_64"
                        / "synthetic_v19_c_lhc_features_normalized_combined_feat_selected.csv",
            "hog_972": DATA_ROOT / "synthetic_v19_c_lhc" / "hog_972"
                        / "synthetic_v19_c_lhc_features_normalized_combined_feat_selected.csv",
            "hog3d_512": DATA_ROOT / "synthetic_v19_c_lhc" / "hog3d_512"
                        / "synthetic_v19_c_lhc_features_normalized_combined_feat_selected.csv",
            "regional_hist_13_64": DATA_ROOT / "synthetic_v19_c_lhc" / "regional_hist_13_64"
                        / "synthetic_v19_c_lhc_features_normalized_combined_feat_selected.csv",
        },
        {
            "curia_embeddings": DATA_ROOT / "synthetic_v19_c_lhc" / "curia_embeddings"
                        / "synthetic_v19_c_lhc_features.csv",
            "histogram_256": DATA_ROOT / "synthetic_v19_c_lhc" / "histogram_256"
                        / "synthetic_v19_c_lhc_features.csv",
            "regional_hist_64": DATA_ROOT / "synthetic_v19_c_lhc" / "regional_hist_64"
                        / "synthetic_v19_c_lhc_features.csv",
            "hog_972": DATA_ROOT / "synthetic_v19_c_lhc" / "hog_972"
                        / "synthetic_v19_c_lhc_features.csv",
            "hog3d_512": DATA_ROOT / "synthetic_v19_c_lhc" / "hog3d_512"
                        / "synthetic_v19_c_lhc_features.csv",
            "regional_hist_13_64": DATA_ROOT / "synthetic_v19_c_lhc" / "regional_hist_13_64"
                        / "synthetic_v19_c_lhc_features.csv",
        },
        {
            "curia_embeddings": DATA_ROOT / "synthetic_v19_c_lhc" / "curia_embeddings"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "histogram_256": DATA_ROOT / "synthetic_v19_c_lhc" / "histogram_256"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "regional_hist_64": DATA_ROOT / "synthetic_v19_c_lhc" / "regional_hist_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "hog_972": DATA_ROOT / "synthetic_v19_c_lhc" / "hog_972"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "hog3d_512": DATA_ROOT / "synthetic_v19_c_lhc" / "hog3d_512"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "regional_hist_13_64": DATA_ROOT / "synthetic_v19_c_lhc" / "regional_hist_13_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        },
    ),
    (
        "v22", "v22_1_lhc_r1",
        {
            "curia_embeddings": DATA_ROOT / "synthetic_v22_1_lhc" / "curia_embeddings"
                        / "synthetic_v22_1_lhc_features_normalized_combined_feat_selected.csv",
            "histogram_256": DATA_ROOT / "synthetic_v22_1_lhc" / "histogram_256"
                        / "synthetic_v22_1_lhc_features_normalized_combined_feat_selected.csv",
            "regional_hist_64": DATA_ROOT / "synthetic_v22_1_lhc" / "regional_hist_64"
                        / "synthetic_v22_1_lhc_features_normalized_combined_feat_selected.csv",
            "hog_972": DATA_ROOT / "synthetic_v22_1_lhc" / "hog_972"
                        / "synthetic_v22_1_lhc_features_normalized_combined_feat_selected.csv",
            "hog3d_512": DATA_ROOT / "synthetic_v22_1_lhc" / "hog3d_512"
                        / "synthetic_v22_1_lhc_features_normalized_combined_feat_selected.csv",
            "regional_hist_13_64": DATA_ROOT / "synthetic_v22_1_lhc" / "regional_hist_13_64"
                        / "synthetic_v22_1_lhc_features_normalized_combined_feat_selected.csv",
        },
        {
            "curia_embeddings": DATA_ROOT / "synthetic_v22_1_lhc" / "curia_embeddings"
                        / "synthetic_v22_1_lhc_features.csv",
            "histogram_256": DATA_ROOT / "synthetic_v22_1_lhc" / "histogram_256"
                        / "synthetic_v22_1_lhc_features.csv",
            "regional_hist_64": DATA_ROOT / "synthetic_v22_1_lhc" / "regional_hist_64"
                        / "synthetic_v22_1_lhc_features.csv",
            "hog_972": DATA_ROOT / "synthetic_v22_1_lhc" / "hog_972"
                        / "synthetic_v22_1_lhc_features.csv",
            "hog3d_512": DATA_ROOT / "synthetic_v22_1_lhc" / "hog3d_512"
                        / "synthetic_v22_1_lhc_features.csv",
            "regional_hist_13_64": DATA_ROOT / "synthetic_v22_1_lhc" / "regional_hist_13_64"
                        / "synthetic_v22_1_lhc_features.csv",
        },
        {
            "curia_embeddings": DATA_ROOT / "synthetic_v22_1_lhc" / "curia_embeddings"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "histogram_256": DATA_ROOT / "synthetic_v22_1_lhc" / "histogram_256"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "regional_hist_64": DATA_ROOT / "synthetic_v22_1_lhc" / "regional_hist_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "hog_972": DATA_ROOT / "synthetic_v22_1_lhc" / "hog_972"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "hog3d_512": DATA_ROOT / "synthetic_v22_1_lhc" / "hog3d_512"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "regional_hist_13_64": DATA_ROOT / "synthetic_v22_1_lhc" / "regional_hist_13_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        },
    ),
    (
        "v22", "v22_2_lhc_r1",
        {
            "curia_embeddings": DATA_ROOT / "synthetic_v22_2_lhc" / "curia_embeddings"
                        / "synthetic_v22_2_lhc_features_normalized_combined_feat_selected.csv",
            "histogram_256": DATA_ROOT / "synthetic_v22_2_lhc" / "histogram_256"
                        / "synthetic_v22_2_lhc_features_normalized_combined_feat_selected.csv",
            "regional_hist_64": DATA_ROOT / "synthetic_v22_2_lhc" / "regional_hist_64"
                        / "synthetic_v22_2_lhc_features_normalized_combined_feat_selected.csv",
            "hog_972": DATA_ROOT / "synthetic_v22_2_lhc" / "hog_972"
                        / "synthetic_v22_2_lhc_features_normalized_combined_feat_selected.csv",
            "hog3d_512": DATA_ROOT / "synthetic_v22_2_lhc" / "hog3d_512"
                        / "synthetic_v22_2_lhc_features_normalized_combined_feat_selected.csv",
            "regional_hist_13_64": DATA_ROOT / "synthetic_v22_2_lhc" / "regional_hist_13_64"
                        / "synthetic_v22_2_lhc_features_normalized_combined_feat_selected.csv",
        },
        {
            "curia_embeddings": DATA_ROOT / "synthetic_v22_2_lhc" / "curia_embeddings"
                        / "synthetic_v22_2_lhc_features.csv",
            "histogram_256": DATA_ROOT / "synthetic_v22_2_lhc" / "histogram_256"
                        / "synthetic_v22_2_lhc_features.csv",
            "regional_hist_64": DATA_ROOT / "synthetic_v22_2_lhc" / "regional_hist_64"
                        / "synthetic_v22_2_lhc_features.csv",
            "hog_972": DATA_ROOT / "synthetic_v22_2_lhc" / "hog_972"
                        / "synthetic_v22_2_lhc_features.csv",
            "hog3d_512": DATA_ROOT / "synthetic_v22_2_lhc" / "hog3d_512"
                        / "synthetic_v22_2_lhc_features.csv",
            "regional_hist_13_64": DATA_ROOT / "synthetic_v22_2_lhc" / "regional_hist_13_64"
                        / "synthetic_v22_2_lhc_features.csv",
        },
        {
            "curia_embeddings": DATA_ROOT / "synthetic_v22_2_lhc" / "curia_embeddings"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "histogram_256": DATA_ROOT / "synthetic_v22_2_lhc" / "histogram_256"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "regional_hist_64": DATA_ROOT / "synthetic_v22_2_lhc" / "regional_hist_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "hog_972": DATA_ROOT / "synthetic_v22_2_lhc" / "hog_972"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "hog3d_512": DATA_ROOT / "synthetic_v22_2_lhc" / "hog3d_512"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "regional_hist_13_64": DATA_ROOT / "synthetic_v22_2_lhc" / "regional_hist_13_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        },
    ),
    (
        "v23", "v23_3_guidance_lhc_r1",
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v23_3_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v23_3_guidance_lhc_features_normalized_combined_feat_selected.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v23_3_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v23_3_guidance_lhc_features.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v23_3_guidance_lhc" / "regional_hist_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        },
    ),
    (
        "v23", "v23_4_guidance_lhc_r1",
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v23_4_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v23_4_guidance_lhc_features_normalized_combined_feat_selected.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v23_4_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v23_4_guidance_lhc_features.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v23_4_guidance_lhc" / "regional_hist_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        },
    ),
    (
        "v25", "v25_1_r1",
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v25_1_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v25_1_guidance_lhc_features_normalized_combined_feat_selected.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v25_1_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v25_1_guidance_lhc_features.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v25_1_guidance_lhc" / "regional_hist_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        },
    ),
    (
        "v25", "v25_2_r1",
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v25_2_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v25_2_guidance_lhc_features_normalized_combined_feat_selected.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v25_2_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v25_2_guidance_lhc_features.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v25_2_guidance_lhc" / "regional_hist_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        },
    ),
    (
        "v26", "v26_1_r1",
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_1_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v26_1_guidance_lhc_features_normalized_combined_feat_selected.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_1_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v26_1_guidance_lhc_features.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_1_guidance_lhc" / "regional_hist_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        },
    ),
    (
        "v26", "v26_2_r1",
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_2_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v26_2_guidance_lhc_features_normalized_combined_feat_selected.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_2_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v26_2_guidance_lhc_features.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_2_guidance_lhc" / "regional_hist_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        },
    ),
    (
        "v26", "v26_3_r1",
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_3_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v26_3_guidance_lhc_features_normalized_combined_feat_selected.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_3_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v26_3_guidance_lhc_features.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_3_guidance_lhc" / "regional_hist_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        },
    ),
    (
        "v26", "v26_4_r1",
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_4_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v26_4_guidance_lhc_features_normalized_combined_feat_selected.csv",
            "curia_embeddings": DATA_ROOT / "synthetic_v26_4_guidance_lhc" / "curia_embeddings"
                        / "synthetic_v26_4_guidance_lhc_features_normalized_combined_feat_selected.csv",
            "hog3d_512":        DATA_ROOT / "synthetic_v26_4_guidance_lhc" / "hog3d_512"
                        / "synthetic_v26_4_guidance_lhc_features_normalized_combined_feat_selected.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_4_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v26_4_guidance_lhc_features.csv",
            "curia_embeddings": DATA_ROOT / "synthetic_v26_4_guidance_lhc" / "curia_embeddings"
                        / "synthetic_v26_4_guidance_lhc_features.csv",
            "hog3d_512":        DATA_ROOT / "synthetic_v26_4_guidance_lhc" / "hog3d_512"
                        / "synthetic_v26_4_guidance_lhc_features.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_4_guidance_lhc" / "regional_hist_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "curia_embeddings": DATA_ROOT / "synthetic_v26_4_guidance_lhc" / "curia_embeddings"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
            "hog3d_512":        DATA_ROOT / "synthetic_v26_4_guidance_lhc" / "hog3d_512"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        },
    ),
    (
        "v26", "v26_2_r1",
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_2_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v26_2_guidance_lhc_features_normalized_combined_feat_selected.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_2_guidance_lhc" / "regional_hist_64"
                        / "synthetic_v26_2_guidance_lhc_features.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_2_guidance_lhc" / "regional_hist_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        },
    ),
    (
        "v26", "v26_4_x20_r1",
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_4_guidance_lhc_x20" / "regional_hist_64"
                        / "synthetic_v26_4_guidance_lhc_x20_features_normalized_combined_feat_selected.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_4_guidance_lhc_x20" / "regional_hist_64"
                        / "synthetic_v26_4_guidance_lhc_x20_features.csv",
        },
        {
            "regional_hist_64": DATA_ROOT / "synthetic_v26_4_guidance_lhc_x20" / "regional_hist_64"
                        / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
        },
    ),
    # ── v26_5: Polarized contrast (coherent cross-class mu polarity, 50% inverted) ─
    (
        "v26", "v26_5_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v26_5_guidance_lhc" / ft
             / f"synthetic_v26_5_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_5_guidance_lhc" / ft
             / f"synthetic_v26_5_guidance_lhc_features.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_5_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64",)},
    ),
    # ── v26_6: Signed alpha (allows within-region intensity inversion) ────────────
    (
        "v26", "v26_6_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v26_6_guidance_lhc" / ft
             / f"synthetic_v26_6_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64", "hog_972", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v26_6_guidance_lhc" / ft
             / f"synthetic_v26_6_guidance_lhc_features.csv"
         for ft in ("regional_hist_64", "hog_972", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v26_6_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64", "hog_972", "hog3d_512")},
    ),
    # ── v26_15: compound double remap (V26_6 applied twice) ──────────────────────
    (
        "v26", "v26_15_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v26_15_guidance_lhc" / ft
             / f"synthetic_v26_15_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_15_guidance_lhc" / ft
             / f"synthetic_v26_15_guidance_lhc_features.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_15_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64",)},
    ),
    # ── v28 family: HOG-space improvement ────────────────────────────────────────
    # ── v28_2: V26_6 + true resolution diversity at save time (50%: 2–4 mm) ──────
    (
        "v28", "v28_2_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v28_2_guidance_lhc" / ft
             / f"synthetic_v28_2_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v28_2_guidance_lhc" / ft
             / f"synthetic_v28_2_guidance_lhc_features.csv"
         for ft in ("regional_hist_64", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v28_2_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64", "hog3d_512")},
    ),
    # ── v28_3: V26_6 + susceptibility signal dropout (GRE pattern) ───────────────
    (
        "v28", "v28_3_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v28_3_guidance_lhc" / ft
             / f"synthetic_v28_3_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v28_3_guidance_lhc" / ft
             / f"synthetic_v28_3_guidance_lhc_features.csv"
         for ft in ("regional_hist_64", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v28_3_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64", "hog3d_512")},
    ),
    # ── v28_4: V28_3 + resolution diversity (combination) ────────────────────────
    (
        "v28", "v28_4_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v28_4_guidance_lhc" / ft
             / f"synthetic_v28_4_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v28_4_guidance_lhc" / ft
             / f"synthetic_v28_4_guidance_lhc_features.csv"
         for ft in ("regional_hist_64", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v28_4_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64", "hog3d_512")},
    ),
    # ── v28_1: V26_6 + Rician noise + aggressive resolution (zoom 0.20-1.0) ──────
    (
        "v28", "v28_1_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v28_1_guidance_lhc" / ft
             / f"synthetic_v28_1_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v28_1_guidance_lhc" / ft
             / f"synthetic_v28_1_guidance_lhc_features.csv"
         for ft in ("regional_hist_64", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v28_1_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64", "hog3d_512")},
    ),
    # ── v26_14: mixed affine + flat per K-means class ─────────────────────────────
    (
        "v26", "v26_14_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v26_14_guidance_lhc" / ft
             / f"synthetic_v26_14_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_14_guidance_lhc" / ft
             / f"synthetic_v26_14_guidance_lhc_features.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_14_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64",)},
    ),
    # ── v26_11: large-C (8–16) signed-alpha, no Voronoi ──────────────────────────
    (
        "v26", "v26_11_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v26_11_guidance_lhc" / ft
             / f"synthetic_v26_11_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_11_guidance_lhc" / ft
             / f"synthetic_v26_11_guidance_lhc_features.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_11_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64",)},
    ),
    # ── v26_12: standard C, stratified mu, signed alpha, Voronoi sub-parc ────────
    (
        "v26", "v26_12_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v26_12_guidance_lhc" / ft
             / f"synthetic_v26_12_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_12_guidance_lhc" / ft
             / f"synthetic_v26_12_guidance_lhc_features.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_12_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64",)},
    ),
    # ── v26_13: large C + stratified mu + signed alpha, no Voronoi ───────────────
    (
        "v26", "v26_13_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v26_13_guidance_lhc" / ft
             / f"synthetic_v26_13_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_13_guidance_lhc" / ft
             / f"synthetic_v26_13_guidance_lhc_features.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_13_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64",)},
    ),
    # ── v26_8: signed alpha + 50 % global inversion ──────────────────────────────
    (
        "v26", "v26_8_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v26_8_guidance_lhc" / ft
             / f"synthetic_v26_8_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_8_guidance_lhc" / ft
             / f"synthetic_v26_8_guidance_lhc_features.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_8_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64",)},
    ),
    # ── v26_9: signed alpha + log-uniform gamma tone ──────────────────────────────
    (
        "v26", "v26_9_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v26_9_guidance_lhc" / ft
             / f"synthetic_v26_9_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_9_guidance_lhc" / ft
             / f"synthetic_v26_9_guidance_lhc_features.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_9_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64",)},
    ),
    # ── v26_10: signed alpha + additive fractal noise ─────────────────────────────
    (
        "v26", "v26_10_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v26_10_guidance_lhc" / ft
             / f"synthetic_v26_10_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_10_guidance_lhc" / ft
             / f"synthetic_v26_10_guidance_lhc_features.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_10_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64",)},
    ),
    # ── v26_7: Flat-region constant intensity (our parcellation + SynthSeg-modeB remap) ──
    (
        "v26", "v26_7_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v26_7_guidance_lhc" / ft
             / f"synthetic_v26_7_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_7_guidance_lhc" / ft
             / f"synthetic_v26_7_guidance_lhc_features.csv"
         for ft in ("regional_hist_64",)},
        {ft: DATA_ROOT / "synthetic_v26_7_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64",)},
    ),
    # ── Our method: Mode A (label-conditioned per-region quantile chunking) ──────
    (
        "v27", "v27a_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v27a_guidance_lhc" / ft
             / f"synthetic_v27a_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64", "regional_hist_13_64", "histogram_256", "hog_972", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v27a_guidance_lhc" / ft
             / f"synthetic_v27a_guidance_lhc_features.csv"
         for ft in ("regional_hist_64", "regional_hist_13_64", "histogram_256", "hog_972", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v27a_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64", "regional_hist_13_64", "histogram_256", "hog_972", "hog3d_512")},
    ),
    # ── Our method: Mode A bis (global EM + per-selected-label refinement) ───────
    (
        "v27", "v27a_bis_guidance_lhc_r1",
        {ft: DATA_ROOT / "synthetic_v27a_bis_guidance_lhc" / ft
             / f"synthetic_v27a_bis_guidance_lhc_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64", "regional_hist_13_64", "histogram_256", "hog_972", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v27a_bis_guidance_lhc" / ft
             / f"synthetic_v27a_bis_guidance_lhc_features.csv"
         for ft in ("regional_hist_64", "regional_hist_13_64", "histogram_256", "hog_972", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_v27a_bis_guidance_lhc" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64", "regional_hist_13_64", "histogram_256", "hog_972", "hog3d_512")},
    ),
    # ── SynthSeg comparison: Mode A (SynthSeg segs → BrainGenerator) ─────────────
    (
        "synthseg", "synthseg_modeA_r1",
        {ft: DATA_ROOT / "synthetic_synthseg_modeA" / ft
             / f"synthetic_synthseg_modeA_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64", "regional_hist_13_64", "histogram_256", "hog_972", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_synthseg_modeA" / ft
             / f"synthetic_synthseg_modeA_features.csv"
         for ft in ("regional_hist_64", "regional_hist_13_64", "histogram_256", "hog_972", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_synthseg_modeA" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64", "regional_hist_13_64", "histogram_256", "hog_972", "hog3d_512")},
    ),
    # ── SynthSeg comparison: Mode B (EM label maps → BrainGenerator) ─────────────
    (
        "synthseg", "synthseg_modeB_em_r1",
        {ft: DATA_ROOT / "synthetic_synthseg_modeB_em" / ft
             / f"synthetic_synthseg_modeB_em_features_normalized_combined_feat_selected.csv"
         for ft in ("regional_hist_64", "regional_hist_13_64", "histogram_256", "hog_972", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_synthseg_modeB_em" / ft
             / f"synthetic_synthseg_modeB_em_features.csv"
         for ft in ("regional_hist_64", "regional_hist_13_64", "histogram_256", "hog_972", "hog3d_512")},
        {ft: DATA_ROOT / "synthetic_synthseg_modeB_em" / ft
             / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv"
         for ft in ("regional_hist_64", "regional_hist_13_64", "histogram_256", "hog_972", "hog3d_512")},
    ),
]


def out_dir(major: str, run: str, mask_type: str, analysis: str) -> Path:
    d = PLOTS_ROOT / major / run / mask_type / analysis
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_cmd(cmd: list[str], log_tag: str) -> bool:
    log.info("[%s] running: %s", log_tag, " ".join(str(c) for c in cmd))
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        log.error("[%s] FAILED (exit %d)", log_tag, result.returncode)
        return False
    log.info("[%s] done", log_tag)
    return True


def run_version(
    major: str,
    run: str,
    synth_norm: Path,
    synth_raw: Path,
    orig_norm: Path,
    orig_raw: Path,
    mask_type: str,
    umap_sup_weight: float = 0.0,
) -> None:
    tag = f"{major}/{run}/{mask_type}"

    if not synth_norm.exists():
        log.warning("[%s] norm CSV missing: %s — skipping", tag, synth_norm)
        return
    if not synth_raw.exists():
        log.warning("[%s] raw CSV missing: %s — skipping", tag, synth_raw)
        return
    if not orig_norm.exists():
        log.warning("[%s] orig norm CSV missing: %s — skipping", tag, orig_norm)
        return
    if not orig_raw.exists():
        log.warning("[%s] orig raw CSV missing: %s — skipping", tag, orig_raw)
        return

    python = sys.executable

    # 1. Feature divergence analysis
    run_cmd([
        python, SCRIPTS / "analyze_features.py",
        "--synthetic_csv",     synth_norm,
        "--synthetic_csv_raw", synth_raw,
        "--original_csv",      orig_norm,
        "--original_csv_raw",  orig_raw,
        "--output_dir",        out_dir(major, run, mask_type, "feature_analysis"),
    ], f"{tag}/feature_analysis")

    # 2. Contrast clustering (LDA)
    run_cmd([
        python, SCRIPTS / "analyze_contrast_clustering.py",
        "--synthetic_csv", synth_norm,
        "--original_csv",  orig_norm,
        "--output_dir",    out_dir(major, run, mask_type, "contrast_clustering"),
    ], f"{tag}/contrast_clustering")

    # 3. PCA only (fast)
    run_cmd([
        python, SCRIPTS / "plot_umap_joint.py",
        "--synthetic_csv", synth_norm,
        "--original_csv",  orig_norm,
        "--output_dir",    out_dir(major, run, mask_type, "pca"),
        "--plot_pca", "--plot_loadings",
        "--skip_umap",
    ], f"{tag}/pca")

    # 4. UMAP (slow)
    run_cmd([
        python, SCRIPTS / "plot_umap_joint.py",
        "--synthetic_csv", synth_norm,
        "--original_csv",  orig_norm,
        "--output_dir",    out_dir(major, run, mask_type, "umap"),
    ], f"{tag}/umap")

    # 5. Coverage analysis (PCA-based manifold coverage metrics)
    run_cmd([
        python, SCRIPTS / "plot_coverage.py",
        "--original_csv",  orig_norm,
        "--synthetic_csv", synth_norm,
        "--output_dir",    out_dir(major, run, mask_type, "coverage"),
    ], f"{tag}/coverage")

    # 6. PRDC + Vendi score per (contrast × scanner) cluster — two PCA variance levels
    for pca_var in ["0.90", "0.60"]:
        subdir = f"prdc_pca{int(float(pca_var)*100)}"
        run_cmd([
            python, SCRIPTS / "plot_prdc.py",
            "--original_csv",  orig_norm,
            "--synthetic_csv", synth_norm,
            "--output_dir",    out_dir(major, run, mask_type, f"pca/{subdir}"),
            "--pca-variance",  pca_var,
        ], f"{tag}/pca/{subdir}")

    # 7. Supervised UMAP (optional, skipped by default)
    if umap_sup_weight > 0.0:
        sup_dir = f"umap_sup_{umap_sup_weight}"
        run_cmd([
            python, SCRIPTS / "plot_umap_joint.py",
            "--synthetic_csv",  synth_norm,
            "--original_csv",   orig_norm,
            "--output_dir",     out_dir(major, run, mask_type, sup_dir),
            "--target_weight",  str(umap_sup_weight),
        ], f"{tag}/{sup_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", type=str, default=None,
                   help="Run only this run_name (e.g. v21_5_r3)")
    p.add_argument("--mask-type", type=str, default="roi_mask",
                   choices=sorted(ORIG_CSVS.keys()),
                   help="Which mask type to analyse (default: roi_mask)")
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument(
        "--umap-sup-weight", type=float, default=0.0,
        help="When > 0, also run a semi-supervised UMAP step with this target_weight "
             "and save results to umap_sup_{weight}/ (default: 0 = skip).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    orig_norm, orig_raw = ORIG_CSVS[args.mask_type]

    versions = VERSIONS
    if args.only:
        versions = [e for e in VERSIONS if e[1] == args.only]
        if not versions:
            log.error("Unknown run name: %s. Valid: %s", args.only,
                      [e[1] for e in VERSIONS])
            sys.exit(1)

    if args.world_size > 1:
        versions = versions[args.rank :: args.world_size]
        log.info("Rank %d/%d → %d versions: %s",
                 args.rank, args.world_size, len(versions),
                 [e[1] for e in versions])

    for entry in versions:
        major, run, norm_csvs, raw_csvs = entry[:4]
        orig_norm_overrides: dict[str, Path] = entry[4] if len(entry) > 4 else {}

        synth_norm = norm_csvs.get(args.mask_type)
        synth_raw  = raw_csvs.get(args.mask_type)
        if synth_norm is None or synth_raw is None:
            log.warning("=== %s/%s — no paths registered for mask_type=%s, skipping ===",
                        major, run, args.mask_type)
            continue
        version_orig_norm = orig_norm_overrides.get(args.mask_type) or orig_norm
        log.info("=== %s/%s  [%s] ===", major, run, args.mask_type)
        run_version(major, run, synth_norm, synth_raw, version_orig_norm, orig_raw,
                    args.mask_type, umap_sup_weight=args.umap_sup_weight)

    log.info("All done.")


if __name__ == "__main__":
    main()
