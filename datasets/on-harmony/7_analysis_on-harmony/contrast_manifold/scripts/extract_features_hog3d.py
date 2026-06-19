#!/usr/bin/env python
"""
3D HOG feature extraction for MRI volumes — native-resolution, stride-subsampled.

For each scan the SynthSeg label map locates the brain bounding box.
3D gradients are computed on the raw brain crop at *native voxel pitch*
(no resize), then subsampled with an adaptive stride so that the effective
voxel count stays near TARGET_NVOX (≈250 k).  The subsampled volume is
divided into a CELLS_PER_DIM³ spatial grid; within each cell a
magnitude-weighted orientation histogram is built over N_ORIENT reference
directions (Fibonacci half-sphere, unsigned/antipodal gradients).  Each cell
histogram is L2-normalised independently.

Why native resolution instead of resizing to 64³:
  Resizing to a fixed voxel count normalises away the native spatial frequency
  content that distinguishes acquisition types:
    T1w  1 mm iso  →  ~180×230×170 crop  →  stride 3  →  ~60×77×57 subsampled
    bold 3 mm iso  →  ~50×60×45 crop     →  stride 1  →  full resolution
  After the stride the two end up at different effective resolutions and their
  HOG histograms genuinely differ — T1w has sharp fine-scale edges, bold/EPI
  has coarser, smoother gradients.

Parameters:
  CELLS_PER_DIM = 4  → 4×4×4 = 64 cells (proportional to crop, any shape)
  N_ORIENT      = 8  → 8 directions on upper half-sphere
  TARGET_NVOX   ≈ 250 000  → adaptive stride targets this voxel count
  Total features: 64 × 8 = 512

Speed: ~0.15 s/scan (CPU), ~10 s for 1650 files with 28 workers.

Usage (recommended — use a CPU-heavy job with extra workers):
  run_job --gpus 0 --cpus 32 --slot 0 --wait -- .venv/bin/python \\
    analysis/contrast_manifold/scripts/extract_features_hog3d.py \\
    --mode original \\
    --output-csv .../original/hog3d_512/on_harmony_features.csv \\
    --n-workers 224

  # Synthetic mode (same pattern):
  run_job --gpus 0 --cpus 32 --slot 0 --wait -- .venv/bin/python \\
    analysis/contrast_manifold/scripts/extract_features_hog3d.py \\
    --mode synthetic --synth-root <synth_dir> \\
    --output-csv <out>.csv \\
    --n-workers 224
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

# ── HOG 3D parameters ─────────────────────────────────────────────────────────
CELLS_PER_DIM   = 4          # spatial cell grid: CELLS_PER_DIM³ cells
N_ORIENT        = 8          # orientation bins (half-sphere, unsigned gradients)
TARGET_NVOX     = 250_000    # stride is chosen so subsampled count ≈ this

N_CELLS         = CELLS_PER_DIM ** 3               # 64
N_FEATURES      = N_CELLS * N_ORIENT               # 512


def _adaptive_stride(crop_shape: tuple) -> int:
    """Return stride so that subsampled voxel count ≈ TARGET_NVOX."""
    nvox = int(np.prod(crop_shape))
    return max(1, round((nvox / TARGET_NVOX) ** (1 / 3)))


def _gradient3d_f32(arr: np.ndarray):
    """Central-difference 3D gradient, float32 throughout (no float64 cast)."""
    gx = np.empty_like(arr)
    gx[1:-1] = (arr[2:] - arr[:-2]) * 0.5
    gx[0]    = arr[1]  - arr[0]
    gx[-1]   = arr[-1] - arr[-2]

    gy = np.empty_like(arr)
    gy[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) * 0.5
    gy[:, 0]    = arr[:, 1]  - arr[:, 0]
    gy[:, -1]   = arr[:, -1] - arr[:, -2]

    gz = np.empty_like(arr)
    gz[:, :, 1:-1] = (arr[:, :, 2:] - arr[:, :, :-2]) * 0.5
    gz[:, :, 0]    = arr[:, :, 1]  - arr[:, :, 0]
    gz[:, :, -1]   = arr[:, :, -1] - arr[:, :, -2]

    return gx, gy, gz


def _fibonacci_half_sphere(n: int) -> np.ndarray:
    """Return (n, 3) unit vectors uniformly distributed on the upper hemisphere."""
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(n, dtype=np.float64)
    # z runs from equator (0) to north pole (1)
    z   = i / max(n - 1, 1)
    phi = 2.0 * np.pi * i / golden
    r   = np.sqrt(np.maximum(1.0 - z ** 2, 0.0))
    dirs = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    return (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).astype(np.float32)


# Precomputed reference directions — shape (N_ORIENT, 3)
_ORIENT_REFS = _fibonacci_half_sphere(N_ORIENT)


def _build_feature_cols() -> list[str]:
    cols = []
    for ci in range(CELLS_PER_DIM):
        for cj in range(CELLS_PER_DIM):
            for ck in range(CELLS_PER_DIM):
                for o in range(N_ORIENT):
                    cols.append(f"hog3d_c{ci}{cj}{ck}_o{o}")
    return cols


# ── SynthSeg mask lookup ──────────────────────────────────────────────────────

def _synthseg_path(sub: str, ses: str) -> Optional[Path]:
    sub_full = sub if sub.startswith("sub-") else f"sub-{sub}"
    ses_full = ses if ses.startswith("ses-") else f"ses-{ses}"
    p = SYNTHSEG_ROOT / sub_full / ses_full / "anat" / f"{sub_full}_{ses_full}_T1w_synthseg.nii.gz"
    return p if p.exists() else None


# ── Per-scan feature computation ──────────────────────────────────────────────

def compute_features(nii_path: Path, synthseg_path: Path) -> Optional[np.ndarray]:
    """Return (N_FEATURES,) = (512,) 3D HOG vector or None on failure.

    Gradients are computed at native voxel pitch (no resize).  An adaptive
    stride subsamples the gradient arrays to ≈TARGET_NVOX voxels so that
    T1w (1 mm) and bold/EPI (3–4 mm) end up at genuinely different effective
    resolutions — preserving the spatial-frequency difference that is the
    primary HOG signal across acquisition types.
    """
    try:
        img = nib.as_closest_canonical(nib.load(str(nii_path)))
        arr = img.get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        img3d = nib.Nifti1Image(arr, img.affine)

        seg_img = nib.as_closest_canonical(nib.load(str(synthseg_path)))
        if seg_img.shape[:3] != img3d.shape[:3] or \
                not np.allclose(seg_img.affine, img3d.affine, atol=1e-3):
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
        arr_norm = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0).astype(np.float32)

        # Crop to brain bounding box
        coords = np.where(brain_mask)
        x0, x1 = int(coords[0].min()), int(coords[0].max()) + 1
        y0, y1 = int(coords[1].min()), int(coords[1].max()) + 1
        z0, z1 = int(coords[2].min()), int(coords[2].max()) + 1
        crop = arr_norm[x0:x1, y0:y1, z0:z1]   # native resolution, float32

        # 3D gradients at native pitch, float32 (avoids np.gradient's float64 cast)
        gx, gy, gz = _gradient3d_f32(crop)

        # Adaptive stride: subsample so effective voxel count ≈ TARGET_NVOX.
        # This keeps T1w (large, fine) and bold/EPI (small, coarse) at their
        # genuine effective resolutions rather than normalising them to the same grid.
        s = _adaptive_stride(crop.shape)
        if s > 1:
            gx = gx[::s, ::s, ::s]
            gy = gy[::s, ::s, ::s]
            gz = gz[::s, ::s, ::s]
        sx, sy, sz = gx.shape

        # Gradient magnitude + unit direction (all float32)
        mag    = np.sqrt(gx * gx + gy * gy + gz * gz)
        m_safe = np.maximum(mag, 1e-8)
        ux = (gx / m_safe).ravel()
        uy = (gy / m_safe).ravel()
        uz = (gz / m_safe).ravel()
        mag_flat = mag.ravel()

        # Orientation bin: argmax of |grad · ref_dir| over half-sphere refs
        dirs       = np.stack([ux, uy, uz], axis=1)       # (N, 3)
        dots       = np.abs(dirs @ _ORIENT_REFS.T)         # (N, N_ORIENT)
        orient_bin = np.argmax(dots, axis=1).astype(np.int32)

        # Cell assignment: proportional mapping from subsampled grid → CELLS_PER_DIM³
        # Works for any crop shape — no fixed CELL_SIZE constant needed.
        def _cell_idx(n: int) -> np.ndarray:
            return np.minimum(
                np.arange(n, dtype=np.int32) * CELLS_PER_DIM // n,
                CELLS_PER_DIM - 1,
            )

        xi3, yi3, zi3 = np.meshgrid(_cell_idx(sx), _cell_idx(sy), _cell_idx(sz),
                                     indexing="ij")
        cell_flat = (xi3 * CELLS_PER_DIM * CELLS_PER_DIM +
                     yi3 * CELLS_PER_DIM +
                     zi3).ravel().astype(np.int32)

        # Magnitude-weighted orientation histogram per cell
        combined = (cell_flat * N_ORIENT + orient_bin).astype(np.int32)
        hist = np.bincount(combined, weights=mag_flat.astype(np.float64),
                           minlength=N_CELLS * N_ORIENT).reshape(N_CELLS, N_ORIENT)

        # L2-normalise each cell independently
        norms = np.linalg.norm(hist, axis=1, keepdims=True)
        hist  = hist / np.maximum(norms, 1e-8)

        return hist.ravel().astype(np.float32)

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

    log.info("3D HOG extractor: %d features/scan (%d cells × %d orientations, native-res + adaptive stride → ~%dk voxels)",
             N_FEATURES, N_CELLS, N_ORIENT, TARGET_NVOX // 1000)

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

    feat_cols    = _build_feature_cols()
    all_cols     = META_COLS + feat_cols
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
                log.warning("[%d/%d] SKIP %s", done_count, len(pending),
                            scan["path"].name)
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
                log.info("[%d/%d] flushing %d rows …", done_count, len(pending),
                         len(results_buf))
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
