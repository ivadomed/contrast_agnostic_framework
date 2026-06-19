#!/usr/bin/env python
"""
HOG (Histogram of Oriented Gradients) feature extraction for MRI scans.

For each scan the SynthSeg label map (T1w of same sub/ses) is used to locate
the brain bounding box.  Three orthogonal centre slices (axial, coronal,
sagittal) are cropped, resized to SLICE_SIZE × SLICE_SIZE, and HOG features
are extracted independently.  The three HOG vectors are concatenated.

Default parameters (pixels_per_cell=16, cells_per_block=2, orientations=9,
SLICE_SIZE=64) produce 3 × 324 = 972-dimensional feature vectors.

HOG captures edge orientation patterns: acquisition noise structure, ringing
artefacts, tissue-boundary sharpness, and gradient anisotropy — all of which
differ between synthetic and real MRI in ways that pure intensity histograms
may miss.

Intensities are globally normalised by [p1, p99] over the whole brain before
HOG computation, keeping relative tissue brightness ordering intact.

CPU-only, parallel via ProcessPoolExecutor.

Usage (original ON-Harmony, all 56 workers):
  run_job --gpus 0 --slot 0 --wait -- .venv/bin/python \\
    analysis/contrast_manifold/scripts/extract_features_hog.py \\
    --mode original \\
    --output-csv analysis/contrast_manifold/outputs/data/original/hog_972/on_harmony_features.csv \\
    --n-workers 56

Usage (synthetic, 4 parallel ranks × 14 workers):
  for rank in 0 1 2 3; do
    run_job --gpus 0 --slot 0 --wait -- .venv/bin/python \\
      analysis/contrast_manifold/scripts/extract_features_hog.py \\
      --mode synthetic --synth-root <path> \\
      --output-csv <path>/features.csv \\
      --n-workers 14 --rank $rank --world-size 4 \\
     
  done
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from skimage.feature import hog
from skimage.transform import resize

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

SYNTHSEG_ROOT = PROJECT_ROOT / "data" / "ON-Harmony" / "derivatives" / "synthseg_masks"

META_COLS = [
    "subject", "session", "modality_id", "acq_tag",
    "scanner_model", "scanner_vendor", "cohort_category",
    "image_path", "mask_path", "label_map_path",
]

SLICE_SIZE   = 64    # each centre slice is resized to SLICE_SIZE × SLICE_SIZE
ORIENTATIONS = 9
PPC          = 16    # pixels_per_cell
CPB          = 2     # cells_per_block

# Feature size: 3 planes × HOG vector per plane
_HOG_PER_PLANE = hog(
    np.zeros((SLICE_SIZE, SLICE_SIZE)),
    orientations=ORIENTATIONS,
    pixels_per_cell=(PPC, PPC),
    cells_per_block=(CPB, CPB),
    visualize=False,
).size
N_FEATURES = 3 * _HOG_PER_PLANE   # = 972 with defaults


def _build_feature_cols() -> list[str]:
    planes = ["axial", "coronal", "sagittal"]
    cols = []
    for plane in planes:
        for i in range(_HOG_PER_PLANE):
            cols.append(f"hog_{plane}_{i:04d}")
    return cols


# ── SynthSeg mask lookup ──────────────────────────────────────────────────────

def _synthseg_path(sub: str, ses: str) -> Optional[Path]:
    sub_full = sub if sub.startswith("sub-") else f"sub-{sub}"
    ses_full = ses if ses.startswith("ses-") else f"ses-{ses}"
    p = SYNTHSEG_ROOT / sub_full / ses_full / "anat" / f"{sub_full}_{ses_full}_T1w_synthseg.nii.gz"
    return p if p.exists() else None


# ── Per-scan feature computation ──────────────────────────────────────────────

def compute_features(nii_path: Path, synthseg_path: Path) -> Optional[np.ndarray]:
    """Return (N_FEATURES,) HOG vector or None on failure."""
    try:
        img = nib.as_closest_canonical(nib.load(str(nii_path)))
        arr = img.get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        img3d = nib.Nifti1Image(arr, img.affine)

        seg_img = nib.as_closest_canonical(nib.load(str(synthseg_path)))
        if seg_img.shape[:3] != img3d.shape[:3] or not np.allclose(seg_img.affine, img3d.affine, atol=1e-3):
            seg_img = resample_from_to(seg_img, img3d, order=0)
        seg = np.round(seg_img.get_fdata()).astype(np.int32)

        brain_mask = seg > 0
        if brain_mask.sum() < 500:
            return None

        # Global p1-p99 normalisation over brain
        brain_vals = arr[brain_mask]
        p1  = float(np.percentile(brain_vals, 1))
        p99 = float(np.percentile(brain_vals, 99))
        if p99 <= p1:
            return None
        arr_norm = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0)

        # Brain bounding box
        coords = np.where(brain_mask)
        x0, x1 = int(coords[0].min()), int(coords[0].max()) + 1
        y0, y1 = int(coords[1].min()), int(coords[1].max()) + 1
        z0, z1 = int(coords[2].min()), int(coords[2].max()) + 1
        crop = arr_norm[x0:x1, y0:y1, z0:z1]
        sx, sy, sz = crop.shape

        # Three orthogonal centre slices
        slices_2d = [
            crop[:, :, sz // 2],   # axial
            crop[:, sy // 2, :],   # coronal
            crop[sx // 2, :, :],   # sagittal
        ]

        feats = []
        for sl in slices_2d:
            sl_r = resize(sl, (SLICE_SIZE, SLICE_SIZE), anti_aliasing=True, preserve_range=True)
            h = hog(
                sl_r,
                orientations=ORIENTATIONS,
                pixels_per_cell=(PPC, PPC),
                cells_per_block=(CPB, CPB),
                visualize=False,
            )
            feats.append(h.astype(np.float32))

        return np.concatenate(feats)

    except Exception as exc:
        log.warning("Feature computation failed for %s: %s", nii_path.name, exc)
        return None


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(args: tuple) -> tuple:
    scan_idx, nii_path, synthseg_path = args
    return scan_idx, compute_features(nii_path, synthseg_path)


# ── Scan discovery (synthetic mode) ──────────────────────────────────────────

def discover_synthetic(synth_root: Path) -> list[dict]:
    records = []
    for nii in sorted(synth_root.rglob("*_syn-*.nii.gz")):
        entities = parse_bids(nii.name)
        sub = nii.parts[nii.parts.index(synth_root.name) + 1]
        ses_parts = [p for p in nii.parts if p.startswith("ses-")]
        ses = ses_parts[0] if ses_parts else "unknown"
        seg = _synthseg_path(sub, ses)
        if seg is None:
            continue
        records.append({
            "path":        nii,
            "synthseg":    seg,
            "entities":    entities,
            "modality_id": (
                f"syn-{entities['syn']}_run-{entities['run']}"
                if "syn" in entities and "run" in entities
                else make_modality_id(entities)
            ),
            "acq_tag":     make_acq_tag(entities),
            "sub":         sub,
            "ses":         ses,
        })
    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",       choices=["original", "synthetic"], required=True)
    p.add_argument("--bids-root",  type=Path, default=PROJECT_ROOT / "data" / "ON-Harmony")
    p.add_argument("--deriv-root", type=Path,
                   default=PROJECT_ROOT / "data" / "ON-Harmony" / "derivatives")
    p.add_argument("--synth-root", type=Path, default=None)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--n-workers",  type=int, default=56)
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    args = p.parse_args()

    log.info("HOG feature extractor: %d features/scan (3 planes × %d)",
             N_FEATURES, _HOG_PER_PLANE)

    # ── Discover scans ──
    if args.mode == "original":
        raw_scans = discover_scans(args.bids_root, args.deriv_root, no_mask=True)
        scans = []
        for s in raw_scans:
            seg = _synthseg_path(s.get("sub", ""), s.get("ses", ""))
            if seg is None:
                continue
            s["synthseg"] = seg
            scans.append(s)
        log.info("Original scans with synthseg mask: %d / %d", len(scans), len(raw_scans))
    else:
        if args.synth_root is None:
            log.error("--synth-root required for synthetic mode"); sys.exit(1)
        scans = discover_synthetic(args.synth_root)

    if args.world_size > 1:
        scans = scans[args.rank :: args.world_size]

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

    feat_cols  = _build_feature_cols()
    all_cols   = META_COLS + feat_cols
    write_header = not out_csv.exists()
    results_buf: list[dict] = []

    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        tasks = {
            pool.submit(_worker, (i, s["path"], s["synthseg"])): (i, s)
            for i, s in pending
        }
        done_count = 0
        for fut in as_completed(tasks):
            orig_idx, scan = tasks[fut]
            done_count += 1
            feat = fut.result()[1]
            if feat is None:
                log.warning("[%d/%d] SKIP %s", done_count, len(pending), scan["path"].name)
                continue

            ses = scan.get("ses", "")
            scanner_model, vendor = parse_scanner(ses)
            row = {
                "subject":         scan.get("sub", ""),
                "session":         ses,
                "modality_id":     scan.get("modality_id", ""),
                "acq_tag":         scan.get("acq_tag", ""),
                "scanner_model":   scanner_model,
                "scanner_vendor":  vendor,
                "cohort_category": scan.get("cohort_category", "") if args.mode == "original" else "",
                "image_path":      str(scan["path"]),
                "mask_path":       "",
                "label_map_path":  str(scan["synthseg"]),
            }
            for col, val in zip(feat_cols, feat):
                row[col] = float(val)

            results_buf.append(row)

            if done_count % 50 == 0 or done_count == len(pending):
                log.info("[%d/%d] flushing %d rows …", done_count, len(pending), len(results_buf))
                df_chunk = pd.DataFrame(results_buf, columns=all_cols)
                df_chunk.to_csv(out_csv, mode="a", header=write_header, index=False)
                write_header = False
                results_buf.clear()

    if results_buf:
        pd.DataFrame(results_buf, columns=all_cols).to_csv(
            out_csv, mode="a", header=write_header, index=False)

    log.info("Done → %s", out_csv)


if __name__ == "__main__":
    main()
