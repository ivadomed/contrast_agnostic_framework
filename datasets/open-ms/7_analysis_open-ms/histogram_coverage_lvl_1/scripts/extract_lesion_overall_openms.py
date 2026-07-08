#!/usr/bin/env python
"""
Stage 2 (open-ms Pillar-2): extract the open-ms coverage feature = [MS-lesion histogram,
overall (whole-image) histogram] = 2 regions × 64 bins = 128-dim.

open-ms provides ONLY sparse binary MS lesion labels. Its brainmask is a computed brain-extraction,
NOT a provided annotation, so we deliberately do NOT use it — we use only the labels open-ms
actually annotates:
  region 0 = LESION   (FLAIR `dseg` mask; binary; co-registered → applies to every contrast)
  region 1 = OVERALL  (whole image, no mask; the overall intensity distribution)

This is the honest, annotation-faithful open-ms analog of on-harmony's 31-region histograms
(2 regions, by design). One global p1–p99 normalisation over the whole image preserves where lesion
intensities sit relative to the rest (bright FLAIR lesions vs dark T1w lesions).

  --mode real            : open-ms FLAIR/T1w/T2w scans (co-registered → shared lesion mask).
  --mode synth --source X: generated volumes (key ends _X); lesion mask from the source subject.

Usage:
  run_job --gpus 0 --cpus 16 --mem 16G --wait -- .venv/bin/python extract_lesion_overall_openms.py \\
      --mode real  --output-csv outputs/real_lesion_overall.csv
  run_job --gpus 0 --cpus 16 --mem 16G --wait -- .venv/bin/python extract_lesion_overall_openms.py \\
      --mode synth --source FLAIR --output-csv outputs/synth_flair_lesion_overall.csv
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

THIS = Path(__file__).resolve()
ANALYSIS = THIS.parents[1]
REPO = THIS.parents[5]
BIDS = REPO / "datasets/open-ms/1_BIDS_open-ms/open-ms-brain"
LESION_DIR = BIDS / "derivatives" / "manual_masks"
GENERATED = ANALYSIS.parent / "data" / "generated"    # shared with Pillar-1 texture_analysis_lvl_1
METHODS = ["palette", "synthseg_em", "synthseg_noem", "auglab_default"]
CONTRASTS = ("FLAIR", "T1w", "T2w")
SCANNER = "openms"
REGIONS = ["lesion", "overall"]
MIN_VOXELS = 50


def feature_cols(n_bins: int):
    return [f"{r}_hist_{b}" for r in REGIONS for b in range(n_bins)]


def lesion_path(sub: str) -> Optional[Path]:
    p = LESION_DIR / sub / "anat" / f"{sub}_FLAIR_dseg.nii.gz"   # only annotation open-ms provides
    return p if p.exists() else None


def compute_features(img_path, lesion_p, n_bins) -> Optional[np.ndarray]:
    try:
        img_nii = nib.as_closest_canonical(nib.load(str(img_path)))
        arr = img_nii.get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        ref = nib.Nifti1Image(arr, img_nii.affine)
        les_nii = nib.as_closest_canonical(nib.load(str(lesion_p)))
        if les_nii.shape[:3] != ref.shape[:3] or not np.allclose(les_nii.affine, ref.affine, atol=1e-3):
            les_nii = resample_from_to(les_nii, ref, order=0)
        lesion = np.round(les_nii.get_fdata()).astype(np.int32) > 0

        p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)     # over the whole image (no mask)
        if p99 <= p1:
            return None
        norm = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0)
        feats = []
        for m in (lesion, np.ones_like(lesion, dtype=bool)):        # lesion, then OVERALL (all voxels)
            if m.sum() < MIN_VOXELS:
                feats.append(np.zeros(n_bins, np.float32))
            else:
                h, _ = np.histogram(norm[m], bins=n_bins, range=(0.0, 1.0))
                s = h.sum()
                feats.append((h / s).astype(np.float32) if s > 0 else np.zeros(n_bins, np.float32))
        return np.concatenate(feats)
    except Exception as exc:
        log.warning("features failed for %s: %s", Path(img_path).name, exc)
        return None


def discover_real():
    tasks, skip = [], 0
    for c in CONTRASTS:
        for img in sorted(BIDS.glob(f"sub-*/anat/*_{c}.nii.gz")):
            sub = img.name.replace(f"_{c}.nii.gz", "")
            les = lesion_path(sub)
            if les is None:
                skip += 1; continue
            tasks.append({"path": img, "lesion": les, "subject": sub,
                          "modality_id": c, "image_path": str(img)})
    log.info("Real scans with lesion mask: %d (skipped %d)", len(tasks), skip)
    return tasks


def discover_synth(source, methods):
    tasks, miss = [], set()
    for method in methods:
        mdir = GENERATED / method
        if not mdir.is_dir():
            log.warning("Method dir absent: %s", mdir); continue
        for case in sorted(p for p in mdir.iterdir() if p.is_dir() and p.name.endswith(f"_{source}")):
            key = case.name
            sub = key.rsplit("_", 1)[0]
            les = lesion_path(sub)
            if les is None:
                miss.add(key); continue
            for nii in sorted(case.glob(f"{key}_run-*.nii.gz")):
                run = nii.stem.split("_run-")[-1].replace(".nii", "")
                tasks.append({"path": nii, "lesion": les, "method": method,
                              "subject": sub, "session": source, "key": key, "run": run})
    if miss:
        log.warning("No lesion mask for %d keys: %s", len(miss), sorted(miss)[:3])
    return tasks


def _worker(a):
    i, img, les, nb = a
    return i, compute_features(img, les, nb)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["real", "synth"], required=True)
    p.add_argument("--source", choices=list(CONTRASTS), default=None)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--n-bins", type=int, default=64)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--methods", nargs="+", default=METHODS)
    args = p.parse_args()
    if args.mode == "synth" and not args.source:
        raise SystemExit("--source required for --mode synth")

    tasks = discover_real() if args.mode == "real" else discover_synth(args.source, args.methods)
    log.info("Discovered %d volumes (mode=%s%s)", len(tasks), args.mode,
             f", source={args.source}" if args.source else "")
    if not tasks:
        log.error("Nothing to extract."); sys.exit(1)

    fcols = feature_cols(args.n_bins)
    rows: list[Optional[dict]] = [None] * len(tasks)
    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futs = {pool.submit(_worker, (i, str(t["path"]), str(t["lesion"]), args.n_bins)): i
                for i, t in enumerate(tasks)}
        for done, fut in enumerate(as_completed(futs), 1):
            i, feat = fut.result()
            t = tasks[i]
            if args.mode == "real":
                base = {"subject": t["subject"], "session": "", "modality_id": t["modality_id"],
                        "acq_tag": "", "scanner_model": SCANNER, "scanner_vendor": SCANNER,
                        "image_path": t["image_path"], "label_map_path": str(t["lesion"])}
            else:
                base = {"method": t["method"], "subject": t["subject"], "session": t["session"],
                        "key": t["key"], "run": t["run"]}
            rows[i] = {**base, **({c: float("nan") for c in fcols} if feat is None
                                   else dict(zip(fcols, feat.tolist())))}
            if feat is None:
                n_fail += 1
            if done % 500 == 0:
                log.info("  %d / %d (%d failed)", done, len(tasks), n_fail)

    df = pd.DataFrame([r for r in rows if r is not None])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    log.info("Saved %d rows × %d cols → %s (%d failed)", len(df), df.shape[1], args.output_csv, n_fail)
    log.info("%s", (df["modality_id"] if args.mode == "real" else df["method"]).value_counts().to_string())


if __name__ == "__main__":
    main()
