#!/usr/bin/env python
"""
Run SynthSeg on all original T1w volumes in ON-Harmony.
Outputs one label map per scan under:
  data/ON-Harmony/derivatives/synthseg_masks/sub-X/ses-Y/anat/
    sub-X_ses-Y_T1w_synthseg.nii.gz

Each rank writes a pair of text files (input list / output list) and calls
SynthSeg's predict() ONCE per rank so the model is loaded only once.
QC is applied after all segmentations in the rank are done.

Model selection: uses SynthSeg 2.0 if SynthSeg/models/synthseg_2.0.h5 exists,
otherwise falls back to SynthSeg 1.0 with a warning.

Usage (4 parallel slots — recommended):
  for rank in 0 1 2 3; do
    run_job --gpus 1 --slot $rank --wait -- .venv/bin/python analysis/contrast_manifold/scripts/run_synthseg.py \\
      --rank $rank --world-size 4 > /tmp/synthseg_rank${rank}.log 2>&1 < /dev/null &
  done
  wait && echo "All done."

Usage (single slot):
  run_job --gpus 1 --slot 0 --wait -- .venv/bin/python analysis/contrast_manifold/scripts/run_synthseg.py

Dev / dry-run (2 volumes):
  run_job --gpus 1 --slot 0 --wait -- .venv/bin/python analysis/contrast_manifold/scripts/run_synthseg.py --limit 2
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
SYNTHSEG_HOME = PROJECT_ROOT / "SynthSeg"
MODELS_DIR    = SYNTHSEG_HOME / "models"
LABELS_DIR    = SYNTHSEG_HOME / "data" / "labels_classes_priors"

MIN_STRUCTURE_VOXELS = 10_000

QC_LABELS = {
    "lh_wm":     2,
    "rh_wm":    41,
    "lh_cortex": 3,
    "rh_cortex": 42,
}


# ─── Model selection ──────────────────────────────────────────────────────────

def _pick_model() -> tuple[Path, bool]:
    v2 = MODELS_DIR / "synthseg_2.0.h5"
    v1 = MODELS_DIR / "synthseg_1.0.h5"
    if v2.exists():
        log.info("Using SynthSeg 2.0: %s", v2)
        return v2, True
    if v1.exists():
        log.warning(
            "SynthSeg 2.0 model not found (%s). Falling back to SynthSeg 1.0.", v2
        )
        return v1, False
    raise FileNotFoundError(
        f"No SynthSeg model found in {MODELS_DIR}. "
        "Expected synthseg_1.0.h5 or synthseg_2.0.h5."
    )


# ─── Batch SynthSeg inference (model loaded once for all scans in rank) ───────

def run_synthseg_batch(
    scan_pairs: list[tuple[Path, Path]],   # [(input, output), ...]
    model_path: Path,
    is_v2: bool,
    rank: int,
) -> None:
    """
    Run SynthSeg on a list of (input, output) pairs in a single predict() call
    so the model is loaded exactly once for the entire batch.
    """
    sys.path.insert(0, str(SYNTHSEG_HOME))
    import tensorflow as tf
    # Each slot owns 64 CPU cores — allow TF to use them for pre/postprocessing
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    from SynthSeg.predict_synthseg import predict  # noqa: PLC0415

    # Write input / output text files for this rank
    tmp_in  = Path(f"/tmp/synthseg_inputs_rank{rank}.txt")
    tmp_out = Path(f"/tmp/synthseg_outputs_rank{rank}.txt")
    with open(tmp_in, "w")  as f: f.write("\n".join(str(p[0]) for p in scan_pairs))
    with open(tmp_out, "w") as f: f.write("\n".join(str(p[1]) for p in scan_pairs))

    suffix = "_2.0.npy" if is_v2 else ".npy"
    predict(
        path_images=str(tmp_in),
        path_segmentations=str(tmp_out),
        path_model_segmentation=str(model_path),
        labels_segmentation=str(LABELS_DIR / f"synthseg_segmentation_labels{suffix}"),
        robust=False,
        fast=True,
        v1=not is_v2,
        n_neutral_labels=19 if is_v2 else 18,
        labels_denoiser=str(LABELS_DIR / "synthseg_denoiser_labels_2.0.npy"),
        path_posteriors=None,
        path_resampled=None,
        path_volumes=None,
        do_parcellation=False,
        path_model_parcellation=None,
        labels_parcellation=str(LABELS_DIR / "synthseg_parcellation_labels.npy"),
        path_qc_scores=None,
        path_model_qc=None,
        labels_qc=str(LABELS_DIR / f"synthseg_qc_labels{suffix}"),
        cropping=None,
        names_segmentation=str(LABELS_DIR / f"synthseg_segmentation_names{suffix}"),
        topology_classes=str(LABELS_DIR / f"synthseg_topological_classes{suffix}"),
    )


# ─── QC ───────────────────────────────────────────────────────────────────────

def qc_segmentation(seg_path: Path) -> dict:
    img  = nib.load(str(seg_path))
    data = np.asarray(img.dataobj, dtype=np.int32)
    result: dict = {"seg_path": str(seg_path), "pass": True}
    for name, label_id in QC_LABELS.items():
        count = int((data == label_id).sum())
        result[f"voxels_{name}"] = count
        if count < MIN_STRUCTURE_VOXELS:
            result["pass"] = False
    return result


# ─── T1w discovery ────────────────────────────────────────────────────────────

def discover_t1w(bids_root: Path) -> list[dict]:
    records = []
    for nii in sorted(bids_root.rglob("*_T1w.nii.gz")):
        if "derivatives" in nii.parts:
            continue
        if "defacemask" in nii.name:
            continue
        sub = nii.parent.parent.parent.name   # sub-X
        ses = nii.parent.parent.name          # ses-Y
        records.append({"path": nii, "sub": sub, "ses": ses})
    return records


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-root", type=Path,
                   default=PROJECT_ROOT / "data" / "ON-Harmony")
    p.add_argument("--output-root", type=Path,
                   default=PROJECT_ROOT / "data" / "ON-Harmony"
                           / "derivatives" / "synthseg_masks")
    p.add_argument("--qc-csv", type=Path, default=None)
    p.add_argument("--rank",        type=int, default=0)
    p.add_argument("--world-size",  type=int, default=1)
    p.add_argument("--limit",       type=int, default=None)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.qc_csv is None:
        args.qc_csv = args.output_root / f"synthseg_qc_rank{args.rank}.csv"

    model_path, is_v2 = _pick_model()

    log.info("Discovering T1w scans …")
    scans = discover_t1w(args.bids_root)
    log.info("%d T1w scans found", len(scans))

    if args.world_size > 1:
        scans = scans[args.rank :: args.world_size]
        log.info("Rank %d/%d → %d scans", args.rank, args.world_size, len(scans))

    if args.limit:
        scans = scans[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Build list of (input, output) pairs, skipping already-done scans
    pairs: list[tuple[Path, Path]] = []
    for scan in scans:
        nii     = scan["path"]
        sub     = scan["sub"]
        ses     = scan["ses"]
        stem    = nii.name.replace(".nii.gz", "")
        out_dir = args.output_root / sub / ses / "anat"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_synthseg.nii.gz"

        if args.skip_existing and out_path.exists():
            log.debug("Skipping (exists): %s", out_path.name)
            continue
        pairs.append((nii, out_path))

    log.info("%d scans to segment (skipped %d existing)", len(pairs), len(scans) - len(pairs))

    if pairs:
        log.info("Running SynthSeg batch (model loaded once) …")
        run_synthseg_batch(pairs, model_path, is_v2, args.rank)
    else:
        log.info("Nothing to do.")

    # QC all outputs (including those already existed + newly created)
    log.info("Running QC …")
    qc_records: list[dict] = []
    n_ok = n_fail = 0
    for scan in scans:
        nii      = scan["path"]
        sub      = scan["sub"]
        ses      = scan["ses"]
        stem     = nii.name.replace(".nii.gz", "")
        out_path = args.output_root / sub / ses / "anat" / f"{stem}_synthseg.nii.gz"

        if not out_path.exists():
            log.warning("Missing output (SynthSeg likely failed): %s", out_path)
            qc_records.append({"seg_path": str(out_path), "pass": False,
                                **{f"voxels_{k}": -1 for k in QC_LABELS}})
            n_fail += 1
            continue

        qc = qc_segmentation(out_path)
        qc_records.append(qc)
        if qc["pass"]:
            n_ok += 1
            log.info("[%s/%s] QC OK  LH-WM=%d  RH-WM=%d",
                     sub, ses, qc["voxels_lh_wm"], qc["voxels_rh_wm"])
        else:
            n_fail += 1
            log.warning("[%s/%s] QC FAIL  LH-WM=%d  RH-WM=%d  LH-CTX=%d  RH-CTX=%d",
                        sub, ses,
                        qc["voxels_lh_wm"], qc["voxels_rh_wm"],
                        qc["voxels_lh_cortex"], qc["voxels_rh_cortex"])

    if qc_records:
        args.qc_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.qc_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(qc_records[0].keys()))
            w.writeheader()
            w.writerows(qc_records)

    log.info("Done. OK=%d  FAIL/QC=%d  QC→%s", n_ok, n_fail, args.qc_csv)


if __name__ == "__main__":
    main()
