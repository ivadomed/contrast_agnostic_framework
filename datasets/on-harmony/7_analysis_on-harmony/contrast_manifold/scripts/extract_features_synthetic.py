#!/usr/bin/env python
"""
Extract PyRadiomics features from synthetic_v19 ON-Harmony volumes.

Re-uses the helper functions from extract_features_native.py verbatim.
Brain masks come from the original derivatives/masks/ tree (same anatomy).

Output:
  analysis/contrast_manifold/outputs/data/synthetic_v19/synthetic_v19_features.csv

Usage:
  set_slot 1 .venv/bin/python analysis/contrast_manifold/scripts/extract_features_synthetic.py \\
      --n_workers 6

  # Dev / dry-run (1 subject):
  set_slot 1 .venv/bin/python analysis/contrast_manifold/scripts/extract_features_synthetic.py \\
      --limit 1
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

for _lg in ("radiomics", "radiomics.glcm", "pykwalify", "pykwalify.core"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

# Import helpers from the native extraction script
_NATIVE_SCRIPT = Path(__file__).parent / "extract_features_native.py"
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("extract_features_native", _NATIVE_SCRIPT)
_native = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_native)

extract_one   = _native.extract_one
build_extractor = _native.build_extractor
parse_bids    = _native.parse_bids
make_modality_id = _native.make_modality_id
make_acq_tag  = _native.make_acq_tag

SYNTHETIC_COHORT = "synthetic_v19"

# ─── Scan discovery ───────────────────────────────────────────────────────────


def discover_synthetic_scans(
    synth_root: Path,
    masks_root: Path,
) -> list[dict]:
    """
    Pair each synthetic NIfTI with the original T1w brain mask.

    Synthetic files are named:
      sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_run-{i:02d}_syn-T1w.nii.gz

    The corresponding mask lives at:
      masks_root/sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_T1w_mask.nii.gz
    """
    records: list[dict] = []
    for nii in sorted(synth_root.rglob("*_syn-T1w.nii.gz")):
        parts = nii.parts
        # sub-X is 2 levels up from file (sub-X/ses-Y/file)
        if len(parts) < 3:
            continue
        sub_dir = nii.parent.parent.name   # sub-XXXXX
        ses_dir = nii.parent.name          # ses-XXXXX
        sub = sub_dir.replace("sub-", "")
        ses = ses_dir.replace("ses-", "")

        mask_name = f"sub-{sub}_ses-{ses}_T1w_mask.nii.gz"
        mask_path = masks_root / f"sub-{sub}" / f"ses-{ses}" / "anat" / mask_name
        if not mask_path.exists():
            log.debug("No mask for %s — skipping", nii.name)
            continue

        entities = parse_bids(nii.name)
        records.append({
            "path":        nii,
            "mask_path":   mask_path,
            "entities":    entities,
            "suffix":      entities.get("suffix", "unknown"),
            "modality_id": f"syn-T1w_run-{entities.get('run', '??')}",
            "acq_tag":     f"synthetic_v19_run-{entities.get('run', '??')}",
            "sub":         f"sub-{sub}",
            "ses":         f"ses-{ses}",
        })
    return records


# ─── Per-scan worker ──────────────────────────────────────────────────────────


def _blur_nii(nii_path: Path, sigma_mm: float) -> Path:
    """
    Apply a Gaussian blur (sigma in mm) via SmoothingRecursiveGaussian — a fast
    1D recursive IIR filter applied along each axis; much cheaper than DiscreteGaussian.
    Returns a temp path. Caller is responsible for cleanup.
    """
    import tempfile
    img = sitk.ReadImage(str(nii_path), sitk.sitkFloat32)
    img_blur = sitk.SmoothingRecursiveGaussian(img, sigma=sigma_mm)
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    tmp.close()
    sitk.WriteImage(img_blur, tmp.name)
    return Path(tmp.name)


def _extract_synthetic_one(
    nii_path: Path,
    mask_path: Path,
    entities: dict,
    sub: str,
    ses: str,
    modality_id: str,
    acq_tag: str,
    config_path: Path,
    blur_sigma: float = 0.0,
) -> dict | None:
    """Thin wrapper: calls extract_one with synthetic-specific cohort metadata."""
    for _lg in ("radiomics", "radiomics.glcm", "pykwalify", "pykwalify.core"):
        logging.getLogger(_lg).setLevel(logging.ERROR)

    tmp_path = None
    if blur_sigma > 0:
        tmp_path = _blur_nii(nii_path, blur_sigma)
        extract_path = tmp_path
    else:
        extract_path = nii_path

    try:
        result = extract_one(
            nii_path=extract_path,
            mask_path=mask_path,
            entities=entities,
            sub=sub,
            ses=ses,
            modality_id=modality_id,
            acq_tag=acq_tag,
            config_path=config_path,
            no_mask=False,
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    if result is not None:
        result["cohort_category"] = SYNTHETIC_COHORT
        result["scanner_model"]  = "v19_generator"
        result["scanner_vendor"] = "synthetic"
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synth-root", type=Path,
                   default=PROJECT_ROOT / "data" / "ON-Harmony" / "derivatives" / "synthetic_v19")
    p.add_argument("--masks-root", type=Path,
                   default=PROJECT_ROOT / "data" / "ON-Harmony" / "derivatives" / "masks")
    p.add_argument("--output-csv", type=Path,
                   default=PROJECT_ROOT / "analysis" / "contrast_manifold" / "outputs"
                            / "data" / "synthetic_v19" / "roi_mask" / "synthetic_v19_features.csv")
    p.add_argument("--config", type=Path,
                   default=PROJECT_ROOT / "analysis" / "contrast_manifold" / "config"
                            / "radiomics_config.yaml")
    p.add_argument("--n_workers", type=int, default=256)
    p.add_argument("--append", action="store_true",
                   help="Append to existing CSV instead of overwriting")
    p.add_argument("--limit", type=int, default=None,
                   help="Dev mode: process only first N scans")
    p.add_argument("--blur-sigma", type=float, default=0.0,
                   help="Gaussian blur sigma in mm applied before feature extraction (0 = no blur)")
    p.add_argument("--rank", type=int, default=0,
                   help="Slot index for multi-process splitting (0-based)")
    p.add_argument("--world-size", type=int, default=1,
                   help="Total number of slots; each slot processes 1/world-size of scans")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"PyRadiomics config not found: {args.config}")
    if not args.synth_root.exists():
        raise FileNotFoundError(
            f"Synthetic root not found: {args.synth_root}\n"
            "Run scripts/generate_synthetic_v19.py first."
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Each rank writes to its own shard to avoid concurrent-write races.
    # When world_size == 1 (single process or merge mode), write directly to output_csv.
    if args.world_size > 1:
        shard_csv = args.output_csv.with_name(
            args.output_csv.stem + f"_rank{args.rank}" + args.output_csv.suffix
        )
    else:
        shard_csv = args.output_csv
    log.info("Output → %s", shard_csv)

    log.info("Discovering synthetic scans …")
    scans = discover_synthetic_scans(args.synth_root, args.masks_root)
    log.info("%d synthetic scans found", len(scans))

    # Rank sharding first so dedup only checks this rank's slice
    if args.world_size > 1:
        scans = scans[args.rank :: args.world_size]
        log.info("Rank %d/%d → %d scans", args.rank, args.world_size, len(scans))

    if args.limit:
        scans = scans[: args.limit]

    # Deduplication when appending
    done_keys: set[str] = set()
    if args.append and shard_csv.exists():
        existing = pd.read_csv(shard_csv, usecols=["image_path"])
        done_keys = set(existing["image_path"].tolist())
        scans = [s for s in scans if str(s["path"]) not in done_keys]
        log.info("%d scans remaining after dedup", len(scans))

    log.info("Submitting %d jobs (%d workers) …", len(scans), args.n_workers)

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {
            pool.submit(
                _extract_synthetic_one,
                s["path"], s["mask_path"], s["entities"],
                s["sub"], s["ses"], s["modality_id"], s["acq_tag"],
                args.config, args.blur_sigma,
            ): s["path"]
            for s in scans
        }
        with tqdm(as_completed(futures), total=len(futures),
                  desc="Extracting features", unit="scan",
                  dynamic_ncols=True) as pbar:
            for fut in pbar:
                src = futures[fut]
                try:
                    rec = fut.result()
                except Exception as exc:
                    pbar.write(f"ERROR {src.name}: {exc}")
                    rec = None
                if rec is not None:
                    results.append(rec)
                else:
                    pbar.write(f"WARN  no features: {src.name}")
                pbar.set_postfix(valid=len(results))

    if not results:
        log.error("No features extracted.")
        return

    META_COLS = [
        "subject", "session", "modality_id", "acq_tag",
        "scanner_model", "scanner_vendor", "cohort_category",
        "image_path", "mask_path",
    ]
    df_new = pd.DataFrame(results)
    feat_cols = [c for c in df_new.columns if c not in META_COLS]
    df_new = df_new[META_COLS + feat_cols]

    if args.append and shard_csv.exists():
        df_existing = pd.read_csv(shard_csv)
        df_out = pd.concat([df_existing, df_new], axis=0, ignore_index=True, sort=False)
    else:
        df_out = df_new

    df_out.to_csv(shard_csv, index=False)
    log.info("Saved %d rows × %d columns → %s", len(df_out), len(df_out.columns), shard_csv)

    # Merge shards into final CSV if this is a multi-rank run
    if args.world_size > 1:
        shards = sorted(args.output_csv.parent.glob(
            args.output_csv.stem + "_rank*" + args.output_csv.suffix
        ))
        if len(shards) == args.world_size:
            log.info("All %d shards present — merging …", args.world_size)
            df_merged = pd.concat([pd.read_csv(s) for s in shards],
                                  axis=0, ignore_index=True, sort=False)
            df_merged.to_csv(args.output_csv, index=False)
            log.info("Merged → %d rows × %d columns → %s",
                     len(df_merged), len(df_merged.columns), args.output_csv)
        else:
            log.info("Shard %d/%d done (waiting for others before merge)",
                     args.rank + 1, args.world_size)


if __name__ == "__main__":
    main()
