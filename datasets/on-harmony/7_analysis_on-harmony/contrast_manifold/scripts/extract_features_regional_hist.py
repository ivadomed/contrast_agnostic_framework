#!/usr/bin/env python
"""
Region-based histogram feature extraction using SynthSeg label maps.

For each scan, the full-brain SynthSeg segmentation (T1w of same sub/ses) is
resampled to the scan's voxel space; then a normalised intensity histogram is
computed independently for each of 7 macro-regions:

  Region 0 – white_matter        (FreeSurfer labels 2, 41)
  Region 1 – cortical_gm         (labels 3, 42)
  Region 2 – csf_ventricles      (labels 4, 5, 14, 15, 43, 44)
  Region 3 – subcortical_gm      (labels 10-13, 17-18, 26, 28, 49-54, 58, 60)
  Region 4 – cerebellum          (labels 7, 8, 46, 47)
  Region 5 – brainstem           (label 16)
  Region 6 – whole_brain_mask    (union of all labelled voxels, background-free)

Features per scan: 7 regions × N_BINS bins = 7*N_BINS floats.
Default N_BINS=64 → 448-dim vector.

Intensities are normalised globally by [p1, p99] of the whole brain BEFORE
computing per-region histograms, so the relative ordering between regions
(WM > GM for T1w; CSF > WM for T2w; CSF-suppressed for FLAIR) is preserved.

CPU-only, parallel via ProcessPoolExecutor.

Usage (original ON-Harmony, all 56 workers):
  set_slot 0 .venv/bin/python \\
    analysis/contrast_manifold/scripts/extract_features_regional_hist.py \\
    --mode original \\
    --output-csv analysis/contrast_manifold/outputs/data/original/regional_hist_64/on_harmony_features.csv \\
    --n-workers 56

Usage (synthetic, 4 parallel ranks × 14 workers):
  for rank in 0 1 2 3; do
    set_slot 0 .venv/bin/python \\
      analysis/contrast_manifold/scripts/extract_features_regional_hist.py \\
      --mode synthetic --synth-root <path> \\
      --output-csv <path>/features.csv \\
      --n-workers 14 --rank $rank --world-size 4 \\
      > /tmp/reghist_${VER}_r${rank}.log 2>&1 &
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

# ── Region definitions (FreeSurfer label → macro-region index) ────────────────
REGION_NAMES = [
    "white_matter",
    "cortical_gm",
    "csf_ventricles",
    "subcortical_gm",
    "cerebellum",
    "brainstem",
    "whole_brain",
]

# Maps each FreeSurfer label to a region index (0-5); 6 = whole_brain built separately
_LABEL_TO_REGION: dict[int, int] = {
    # white matter (0)
    2: 0, 41: 0,
    # cortical GM (1)
    3: 1, 42: 1,
    # CSF / ventricles (2)
    4: 2, 5: 2, 14: 2, 15: 2, 43: 2, 44: 2,
    # subcortical GM (3)
    10: 3, 11: 3, 12: 3, 13: 3, 17: 3, 18: 3, 26: 3, 28: 3,
    49: 3, 50: 3, 51: 3, 52: 3, 53: 3, 54: 3, 58: 3, 60: 3,
    # cerebellum (4)
    7: 4, 8: 4, 46: 4, 47: 4,
    # brainstem (5)
    16: 5,
}
N_REGIONS  = len(REGION_NAMES)  # 7
MIN_VOXELS = 50                  # skip region if fewer voxels after resampling


def _build_feature_cols(n_bins: int) -> list[str]:
    cols = []
    for region in REGION_NAMES:
        for b in range(n_bins):
            cols.append(f"{region}_hist_{b}")
    return cols


# ── SynthSeg mask lookup ──────────────────────────────────────────────────────

def _synthseg_path(sub: str, ses: str) -> Optional[Path]:
    """Return path to SynthSeg label map for this subject/session, or None.

    Accepts both 'sub-XXXX' (full) and 'XXXX' (bare) forms for sub and ses.
    """
    sub_full = sub if sub.startswith("sub-") else f"sub-{sub}"
    ses_full = ses if ses.startswith("ses-") else f"ses-{ses}"
    p = SYNTHSEG_ROOT / sub_full / ses_full / "anat" / f"{sub_full}_{ses_full}_T1w_synthseg.nii.gz"
    return p if p.exists() else None


# ── Per-scan feature computation ──────────────────────────────────────────────

def compute_features(nii_path: Path, synthseg_path: Path, n_bins: int) -> Optional[np.ndarray]:
    """
    Return (N_REGIONS * n_bins,) feature vector or None on failure.

    Normalisation is global (p1-p99 of whole brain) before per-region histograms,
    preserving the inter-region intensity ordering that defines MRI contrast.
    """
    try:
        # Load image in canonical orientation
        img = nib.as_closest_canonical(nib.load(str(nii_path)))
        arr = img.get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]

        # Build a 3D reference image (needed when the scan is 4D, e.g. DWI/bold).
        img3d = nib.Nifti1Image(arr, img.affine)

        # Load and resample label map to image space (nearest-neighbour).
        seg_img = nib.as_closest_canonical(nib.load(str(synthseg_path)))
        if seg_img.shape[:3] != img3d.shape[:3] or not np.allclose(seg_img.affine, img3d.affine, atol=1e-3):
            seg_img = resample_from_to(seg_img, img3d, order=0)
        seg = np.round(seg_img.get_fdata()).astype(np.int32)

        # Build region masks from label map
        region_masks = [np.zeros(arr.shape, dtype=bool) for _ in range(N_REGIONS)]
        for label, ridx in _LABEL_TO_REGION.items():
            region_masks[ridx] |= (seg == label)
        # whole_brain = union of all named labels
        brain_mask = np.zeros(arr.shape, dtype=bool)
        for m in region_masks[:N_REGIONS - 1]:
            brain_mask |= m
        region_masks[N_REGIONS - 1] = brain_mask

        # Global normalisation (p1-p99 over whole brain) to preserve ordering
        brain_vals = arr[brain_mask]
        if brain_vals.size < 500:
            return None
        p1  = float(np.percentile(brain_vals, 1))
        p99 = float(np.percentile(brain_vals, 99))
        if p99 <= p1:
            return None
        arr_norm = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0)

        # Per-region histograms
        feats = []
        for mask in region_masks:
            if mask.sum() < MIN_VOXELS:
                feats.append(np.zeros(n_bins, dtype=np.float32))
            else:
                vals = arr_norm[mask]
                h, _ = np.histogram(vals, bins=n_bins, range=(0.0, 1.0))
                s = h.sum()
                feats.append((h / s).astype(np.float32) if s > 0 else np.zeros(n_bins, np.float32))

        return np.concatenate(feats)

    except Exception as exc:
        log.warning("Feature computation failed for %s: %s", nii_path.name, exc)
        return None


# ── Worker (top-level for pickling) ──────────────────────────────────────────

def _worker(args: tuple) -> tuple:
    scan_idx, nii_path, synthseg_path, n_bins = args
    return scan_idx, compute_features(nii_path, synthseg_path, n_bins)


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
            continue  # no synthseg → skip
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
    p.add_argument("--n-bins",     type=int, default=64,
                   help="Histogram bins per region (default 64; 64×7=448 dims total)")
    p.add_argument("--n-workers",  type=int, default=56)
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    args = p.parse_args()

    # ── Discover scans ──
    if args.mode == "original":
        raw_scans = discover_scans(args.bids_root, args.deriv_root, no_mask=True)
        scans = []
        for s in raw_scans:
            sub = s.get("sub", "")
            ses = s.get("ses", "")
            seg = _synthseg_path(sub, ses)
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
    log.info("Processing %d / %d scans  (rank %d/%d, workers=%d, bins=%d)",
             len(pending), len(scans), args.rank, args.world_size,
             args.n_workers, args.n_bins)

    if not pending:
        log.info("Nothing to do.")
        return

    feat_cols = _build_feature_cols(args.n_bins)
    all_cols  = META_COLS + feat_cols
    write_header = not out_csv.exists()
    results_buf: list[dict] = []

    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        tasks = {
            pool.submit(_worker, (i, s["path"], s["synthseg"], args.n_bins)): (i, s)
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
