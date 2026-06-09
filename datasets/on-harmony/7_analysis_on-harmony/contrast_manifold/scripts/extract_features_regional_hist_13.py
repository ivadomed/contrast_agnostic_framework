#!/usr/bin/env python
"""
13-region histogram feature extraction using SynthSeg label maps.

Regions are designed to maximise modality discrimination:

  Region  0 – cortical_wm          (labels 2, 41)
  Region  1 – cortical_gm          (labels 3, 42)
  Region  2 – lateral_ventricles   (labels 4, 43)          ← FLAIR suppression signal
  Region  3 – deep_csf             (labels 5, 14, 15, 44)  ← 3rd/4th/inf ventricles
  Region  4 – thalamus             (labels 10, 49)         ← largest subcortical; strong T1/T2 contrast
  Region  5 – striatum             (labels 11, 12, 50, 51) ← caudate + putamen; iron-sensitive (T2*)
  Region  6 – globus_pallidus      (labels 13, 26, 52, 58) ← pallidum + accumbens; highest iron → T2* dark
  Region  7 – hippocampus_amygdala (labels 17, 18, 53, 54) ← medial temporal
  Region  8 – ventral_dc           (labels 28, 60)         ← subthalamic / hypothalamus
  Region  9 – cerebellum_wm        (labels 7, 46)          ← lower myelin density than cerebral WM
  Region 10 – cerebellum_gm        (labels 8, 47)          ← cerebellar cortex
  Region 11 – brainstem            (label 16)
  Region 12 – whole_brain          (union of all above)

Features per scan: 13 regions × N_BINS bins = 13*N_BINS floats.
Default N_BINS=64 → 832-dim vector.

Intensities are normalised globally by [p1, p99] of the whole brain BEFORE
computing per-region histograms (same convention as regional_hist_64).

CPU-only, parallel via ProcessPoolExecutor.

Usage (original ON-Harmony, 4 ranks × 56 workers):
  for rank in 0 1 2 3; do
    set_slot $rank .venv/bin/python \\
      analysis/contrast_manifold/scripts/extract_features_regional_hist_13.py \\
      --mode original \\
      --output-csv analysis/contrast_manifold/outputs/data/original/regional_hist_13_64/on_harmony_features.csv \\
      --n-workers 56 --rank $rank --world-size 4 \\
      > /tmp/rh13_orig_r${rank}.log 2>&1 &
  done

Usage (synthetic, 4 versions × 4 ranks):
  see scripts/run_regional_hist_13_pipeline.sh
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

# ── Region definitions ────────────────────────────────────────────────────────
REGION_NAMES = [
    "cortical_wm",           # 0
    "cortical_gm",           # 1
    "lateral_ventricles",    # 2  — key FLAIR discriminator
    "deep_csf",              # 3  — 3rd/4th/inf ventricles
    "thalamus",              # 4  — strong T1/T2 contrast
    "striatum",              # 5  — caudate + putamen, T2* iron-sensitive
    "globus_pallidus",       # 6  — highest iron, most T2*-hypointense
    "hippocampus_amygdala",  # 7  — medial temporal
    "ventral_dc",            # 8  — subthalamic / hypothalamus
    "cerebellum_wm",         # 9  — lower myelin than cerebral WM
    "cerebellum_gm",         # 10 — cerebellar cortex
    "brainstem",             # 11
    "whole_brain",           # 12 — union sentinel
]

_LABEL_TO_REGION: dict[int, int] = {
    # cortical WM (0)
    2: 0, 41: 0,
    # cortical GM (1)
    3: 1, 42: 1,
    # lateral ventricles (2)
    4: 2, 43: 2,
    # deep CSF: inf-lat + 3rd + 4th ventricles (3)
    5: 3, 14: 3, 15: 3, 44: 3,
    # thalamus (4)
    10: 4, 49: 4,
    # striatum: caudate + putamen (5)
    11: 5, 12: 5, 50: 5, 51: 5,
    # globus pallidus + accumbens (6)
    13: 6, 26: 6, 52: 6, 58: 6,
    # hippocampus + amygdala (7)
    17: 7, 18: 7, 53: 7, 54: 7,
    # ventral DC: subthalamic nucleus / hypothalamus (8)
    28: 8, 60: 8,
    # cerebellum WM (9)
    7: 9, 46: 9,
    # cerebellum GM (10)
    8: 10, 47: 10,
    # brainstem (11)
    16: 11,
}

N_REGIONS  = len(REGION_NAMES)   # 13
MIN_VOXELS = 20                   # smaller regions (e.g. pallidum) have fewer voxels


def _build_feature_cols(n_bins: int) -> list[str]:
    return [f"{region}_hist_{b}" for region in REGION_NAMES for b in range(n_bins)]


# ── SynthSeg mask lookup ──────────────────────────────────────────────────────

def _synthseg_path(sub: str, ses: str) -> Optional[Path]:
    sub_full = sub if sub.startswith("sub-") else f"sub-{sub}"
    ses_full = ses if ses.startswith("ses-") else f"ses-{ses}"
    p = SYNTHSEG_ROOT / sub_full / ses_full / "anat" / f"{sub_full}_{ses_full}_T1w_synthseg.nii.gz"
    return p if p.exists() else None


# ── Per-scan feature computation ──────────────────────────────────────────────

def compute_features(nii_path: Path, synthseg_path: Path, n_bins: int) -> Optional[np.ndarray]:
    """Return (N_REGIONS * n_bins,) feature vector or None on failure."""
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

        # Build region masks
        region_masks = [np.zeros(arr.shape, dtype=bool) for _ in range(N_REGIONS)]
        for label, ridx in _LABEL_TO_REGION.items():
            region_masks[ridx] |= (seg == label)

        # whole_brain = union of all named regions
        brain_mask = np.zeros(arr.shape, dtype=bool)
        for m in region_masks[:N_REGIONS - 1]:
            brain_mask |= m
        region_masks[N_REGIONS - 1] = brain_mask

        # Global p1-p99 normalisation over whole brain
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


# ── Worker ────────────────────────────────────────────────────────────────────

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
    p.add_argument("--n-bins",     type=int, default=64,
                   help="Histogram bins per region (default 64; 64×13=832 dims total)")
    p.add_argument("--n-workers",  type=int, default=56)
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    args = p.parse_args()

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
    log.info("Processing %d / %d scans  (rank %d/%d, workers=%d, bins=%d)",
             len(pending), len(scans), args.rank, args.world_size,
             args.n_workers, args.n_bins)

    if not pending:
        log.info("Nothing to do.")
        return

    feat_cols    = _build_feature_cols(args.n_bins)
    all_cols     = META_COLS + feat_cols
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
