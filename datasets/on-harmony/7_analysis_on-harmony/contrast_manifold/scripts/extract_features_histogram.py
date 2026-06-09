#!/usr/bin/env python
"""
Histogram-based feature extraction for MRI contrast characterisation.

Produces three complementary feature blocks per volume:
  hist_0  … hist_255  – 256-bin intensity histogram (brain-masked, [0,1] normalised)
  grad_0  … grad_255  – 256-bin gradient-magnitude histogram (captures resolution/texture)
  stats_0 … stats_13  – 14 first-order statistics (moments, percentiles, entropy)

The two 256-bin histograms are model-agnostic, physics-motivated features:
  • Intensity histogram directly encodes tissue contrast (T1w bimodal WM/GM,
    T2w reversal, FLAIR CSF suppression, DWI noise floor, etc.)
  • Gradient histogram encodes sharpness / effective resolution (high-res T1w vs
    blurry DWI/EPI), complementing the resolution-resampler experiments (v22_x).
  • First-order stats capture distributional moments without binning artefacts.

CPU-only, parallel via ProcessPoolExecutor.  No GPU required.

Usage (original ON-Harmony, all 64 cores):
  set_slot 0 .venv/bin/python analysis/contrast_manifold/scripts/extract_features_histogram.py \\
    --mode original \\
    --output-csv analysis/contrast_manifold/outputs/data/original/histogram_256/on_harmony_features.csv \\
    --n-workers 56

Usage (synthetic, parallel ranks):
  for rank in 0 1 2 3; do
    set_slot 0 .venv/bin/python ... --mode synthetic --synth-root ... \\
      --rank $rank --world-size 4 --n-workers 14 > /tmp/hist_rank${rank}.log 2>&1 &
  done
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "extract_features_native",
    Path(__file__).parent / "extract_features_native.py",
)
_native = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_native)

discover_scans   = _native.discover_scans
parse_bids       = _native.parse_bids
make_modality_id = _native.make_modality_id
make_acq_tag     = _native.make_acq_tag
parse_scanner    = _native.parse_scanner

META_COLS = [
    "subject", "session", "modality_id", "acq_tag",
    "scanner_model", "scanner_vendor", "cohort_category",
    "image_path", "mask_path", "label_map_path",
]

N_BINS  = 256
N_STATS = 14  # see _stats_names below
_HIST_COLS  = [f"hist_{i}"  for i in range(N_BINS)]
_GRAD_COLS  = [f"grad_{i}"  for i in range(N_BINS)]
_STATS_COLS = [f"stats_{i}" for i in range(N_STATS)]

_STATS_NAMES = [
    "mean", "std", "skewness", "kurtosis",
    "p5", "p10", "p25", "p50", "p75", "p90", "p95",
    "entropy", "cov", "iqr",
]


# ─── Feature computation ──────────────────────────────────────────────────────

def _brain_mask(arr: np.ndarray) -> np.ndarray:
    """Return bool mask of non-background voxels.

    Uses a dual criterion: above a low absolute threshold AND within a
    smoothed (4mm FWHM ≈ σ≈1.7 vox) envelope to reject isolated hot pixels.
    """
    pos = arr[arr > 0]
    if pos.size < 1000:
        return arr > 0
    thresh = float(np.percentile(pos, 3))
    # Smooth envelope to include all brain-adjacent voxels
    smooth = uniform_filter(arr.astype(np.float32), size=5)
    return (arr > thresh) & (smooth > thresh * 0.5)


def _intensity_histogram(vals_norm: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(vals_norm, bins=N_BINS, range=(0.0, 1.0))
    s = h.sum()
    return (h / s).astype(np.float32) if s > 0 else np.zeros(N_BINS, np.float32)


def _gradient_histogram(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Finite differences along each axis → gradient magnitude
    gx = np.gradient(arr, axis=0)
    gy = np.gradient(arr, axis=1)
    gz = np.gradient(arr, axis=2)
    gmag = np.sqrt(gx**2 + gy**2 + gz**2)
    gvals = gmag[mask].astype(np.float64)
    p99 = float(np.percentile(gvals, 99))
    if p99 < 1e-8:
        return np.zeros(N_BINS, np.float32)
    gnorm = np.clip(gvals / p99, 0.0, 1.0)
    h, _ = np.histogram(gnorm, bins=N_BINS, range=(0.0, 1.0))
    s = h.sum()
    return (h / s).astype(np.float32) if s > 0 else np.zeros(N_BINS, np.float32)


def _first_order_stats(vals_norm: np.ndarray) -> np.ndarray:
    if vals_norm.size == 0:
        return np.zeros(N_STATS, np.float32)
    from scipy.stats import skew, kurtosis as kurt
    mean  = float(np.mean(vals_norm))
    std   = float(np.std(vals_norm))
    sk    = float(skew(vals_norm))
    ku    = float(kurt(vals_norm))
    p5, p10, p25, p50, p75, p90, p95 = (
        float(np.percentile(vals_norm, q)) for q in (5, 10, 25, 50, 75, 90, 95)
    )
    # Shannon entropy of the histogram (same bins as intensity histogram)
    h = _intensity_histogram(vals_norm)
    h_nz = h[h > 0]
    entropy = float(-np.sum(h_nz * np.log2(h_nz)))
    cov = std / mean if mean > 1e-8 else 0.0
    iqr = p75 - p25
    return np.array([mean, std, sk, ku, p5, p10, p25, p50, p75, p90, p95,
                     entropy, cov, iqr], dtype=np.float32)


def compute_features(nii_path: Path) -> np.ndarray | None:
    """Return (N_BINS + N_BINS + N_STATS,) feature vector or None on failure."""
    try:
        img = nib.as_closest_canonical(nib.load(str(nii_path)))
        arr = img.get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]

        mask = _brain_mask(arr)
        if mask.sum() < 500:
            return None

        vals = arr[mask]
        p1  = float(np.percentile(vals, 1))
        p99 = float(np.percentile(vals, 99))
        if p99 <= p1:
            return None
        vals_norm = np.clip((vals - p1) / (p99 - p1), 0.0, 1.0).astype(np.float64)

        hist  = _intensity_histogram(vals_norm)
        grad  = _gradient_histogram(arr, mask)
        stats = _first_order_stats(vals_norm)

        return np.concatenate([hist, grad, stats])
    except Exception as exc:
        log.warning("Feature computation failed for %s: %s", nii_path.name, exc)
        return None


# ─── Worker (top-level for pickling) ─────────────────────────────────────────

def _worker(args: tuple) -> tuple:
    """(scan_idx, nii_path) → (scan_idx, feature_vector | None)"""
    scan_idx, nii_path = args
    return scan_idx, compute_features(nii_path)


# ─── Scan discovery (synthetic) ───────────────────────────────────────────────

def discover_synthetic(synth_root: Path) -> list[dict]:
    records = []
    for nii in sorted(synth_root.rglob("*_syn-*.nii.gz")):
        entities = parse_bids(nii.name)
        sub = nii.parts[nii.parts.index(synth_root.name) + 1]
        ses_parts = [p for p in nii.parts if p.startswith("ses-")]
        ses = ses_parts[0] if ses_parts else "unknown"
        records.append({
            "path": nii,
            "entities": entities,
            "modality_id": (
                f"syn-{entities['syn']}_run-{entities['run']}"
                if "syn" in entities and "run" in entities
                else make_modality_id(entities)
            ),
            "acq_tag":  make_acq_tag(entities),
            "sub": sub,
            "ses": ses,
        })
    return records


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",       choices=["original", "synthetic"], required=True)
    p.add_argument("--bids-root",  type=Path, default=PROJECT_ROOT / "data" / "ON-Harmony")
    p.add_argument("--deriv-root", type=Path,
                   default=PROJECT_ROOT / "data" / "ON-Harmony" / "derivatives")
    p.add_argument("--synth-root", type=Path, default=None)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--n-workers",  type=int, default=56,
                   help="CPU workers for parallel extraction")
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    args = p.parse_args()

    # ── Discover scans ──
    if args.mode == "original":
        scans = discover_scans(args.bids_root, args.deriv_root, no_mask=True)
    else:
        if args.synth_root is None:
            log.error("--synth-root required for synthetic mode"); sys.exit(1)
        scans = discover_synthetic(args.synth_root)

    if args.world_size > 1:
        scans = scans[args.rank :: args.world_size]

    # ── Resume: skip already-done paths ──
    out_csv = args.output_csv
    if args.world_size > 1:
        out_csv = out_csv.with_name(out_csv.stem + f"_rank{args.rank}" + out_csv.suffix)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done_paths: set[str] = set()
    if out_csv.exists():
        try:
            done_paths = set(pd.read_csv(out_csv, usecols=["image_path"])["image_path"].astype(str))
            log.info("Resuming: %d already done", len(done_paths))
        except Exception:
            pass

    pending = [(i, s) for i, s in enumerate(scans)
               if str(s["path"]) not in done_paths]
    log.info("Processing %d / %d scans  (rank %d/%d, workers=%d)",
             len(pending), len(scans), args.rank, args.world_size, args.n_workers)

    if not pending:
        log.info("Nothing to do.")
        return

    # ── Feature columns ──
    feat_cols = _HIST_COLS + _GRAD_COLS + _STATS_COLS
    all_cols  = META_COLS + feat_cols

    write_header = not out_csv.exists()
    results_buf: list[dict] = []

    # ── Parallel extraction ──
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        tasks = {pool.submit(_worker, (orig_idx, s["path"])): (orig_idx, s)
                 for orig_idx, s in pending}
        done_count = 0
        for fut in as_completed(tasks):
            orig_idx, scan = tasks[fut]
            done_count += 1
            feat = fut.result()[1]
            if feat is None:
                log.warning("[%d/%d] SKIP %s", done_count, len(pending),
                            scan["path"].name)
                continue

            # Build metadata row
            if args.mode == "original":
                scanner_model, vendor = parse_scanner(scan.get("ses", ""))
                row = {
                    "subject":          scan.get("sub", ""),
                    "session":          scan.get("ses", ""),
                    "modality_id":      scan.get("modality_id", ""),
                    "acq_tag":          scan.get("acq_tag", ""),
                    "scanner_model":    scanner_model,
                    "scanner_vendor":   vendor,
                    "cohort_category":  scan.get("cohort_category", ""),
                    "image_path":       str(scan["path"]),
                    "mask_path":        "",
                    "label_map_path":   "",
                }
            else:
                ses = scan.get("ses", "")
                scanner_model, vendor = parse_scanner(ses)
                row = {
                    "subject":          scan.get("sub", ""),
                    "session":          ses,
                    "modality_id":      scan.get("modality_id", ""),
                    "acq_tag":          scan.get("acq_tag", ""),
                    "scanner_model":    scanner_model,
                    "scanner_vendor":   vendor,
                    "cohort_category":  "",
                    "image_path":       str(scan["path"]),
                    "mask_path":        "",
                    "label_map_path":   "",
                }

            for col, val in zip(feat_cols, feat):
                row[col] = float(val)

            results_buf.append(row)

            if done_count % 50 == 0 or done_count == len(pending):
                log.info("[%d/%d] flushing %d rows …", done_count, len(pending),
                         len(results_buf))
                df_chunk = pd.DataFrame(results_buf, columns=all_cols)
                df_chunk.to_csv(out_csv, mode="a", header=write_header, index=False)
                write_header = False
                results_buf.clear()

    # Flush remainder
    if results_buf:
        pd.DataFrame(results_buf, columns=all_cols).to_csv(
            out_csv, mode="a", header=write_header, index=False)

    log.info("Done → %s", out_csv)


if __name__ == "__main__":
    main()
