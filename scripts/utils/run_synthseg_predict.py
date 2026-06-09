#!/usr/bin/env python
"""
Run SynthSeg v1.0 segmentation on all ON-Harmony T1w volumes.

Output structure (mirrors BIDS):
  data/ON-Harmony/derivatives/synthseg_segs/sub-*/ses-*/<t1w_name>.nii.gz
  (same filename as the source T1w — generate_synthetic_guidance.py with
  --seg-root looks up segs as: seg_root/sub/ses/t1w_name.nii.gz)

Calls SynthSeg predict() once per rank (not per subject) so the TF model
is loaded only once, making throughput much faster.

Usage (4 GPUs, one rank each):
  set_slot 0 .venv/bin/python scripts/run_synthseg_predict.py --rank 0 --world-size 4 > /tmp/ss_r0.log 2>&1 &
  set_slot 1 .venv/bin/python scripts/run_synthseg_predict.py --rank 1 --world-size 4 > /tmp/ss_r1.log 2>&1 &
  set_slot 2 .venv/bin/python scripts/run_synthseg_predict.py --rank 2 --world-size 4 > /tmp/ss_r2.log 2>&1 &
  set_slot 3 .venv/bin/python scripts/run_synthseg_predict.py --rank 3 --world-size 4 > /tmp/ss_r3.log 2>&1 &
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
SYNTHSEG_HOME = PROJECT_ROOT / "SynthSeg"
OUT_ROOT      = PROJECT_ROOT / "data" / "ON-Harmony" / "derivatives" / "synthseg_segs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-root", type=Path,
                   default=PROJECT_ROOT / "data" / "ON-Harmony")
    p.add_argument("--output-root", type=Path, default=OUT_ROOT)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    import os, ctypes, glob
    # set_slot does NOT set CUDA_VISIBLE_DEVICES — restrict this rank to its GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.rank)
    # sudo strips LD_LIBRARY_PATH; load venv CUDA libs via ctypes so TF finds them.
    for _lib in sorted(glob.glob(str(
            PROJECT_ROOT / ".venv/lib/python3.12/site-packages/nvidia/*/lib/*.so*"))):
        try: ctypes.CDLL(_lib)
        except OSError: pass
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    sys.path.insert(0, str(SYNTHSEG_HOME))
    from SynthSeg.predict_synthseg import predict

    labels_dir = SYNTHSEG_HOME / "data" / "labels_classes_priors"
    model_path  = SYNTHSEG_HOME / "models" / "synthseg_1.0.h5"

    if not model_path.exists():
        raise FileNotFoundError(f"SynthSeg v1.0 model not found: {model_path}")

    # Discover all T1w files and build output paths
    t1w_files = sorted(args.bids_root.glob("sub-*/ses-*/anat/*_T1w.nii.gz"))
    log.info("Found %d T1w volumes", len(t1w_files))
    if args.limit:
        t1w_files = t1w_files[:args.limit]

    def _out_path(p: Path) -> Path:
        sub, ses = p.parts[-4], p.parts[-3]
        return args.output_root / sub / ses / p.name

    # Filter to pending (not yet done)
    pending_pairs = [(p, _out_path(p)) for p in t1w_files if not _out_path(p).exists()]
    log.info("%d pending / %d total", len(pending_pairs), len(t1w_files))

    if args.world_size > 1:
        pending_pairs = pending_pairs[args.rank::args.world_size]
        log.info("Rank %d/%d: %d volumes", args.rank, args.world_size, len(pending_pairs))

    if not pending_pairs:
        log.info("Nothing to do.")
        return

    # Create output directories
    for _, out in pending_pairs:
        out.parent.mkdir(parents=True, exist_ok=True)

    # Write path lists to temp files and call predict() once (model loaded once)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f_in, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f_out:
        for t1w, out in pending_pairs:
            f_in.write(str(t1w) + "\n")
            f_out.write(str(out) + "\n")
        in_list_path  = f_in.name
        out_list_path = f_out.name

    log.info("Starting SynthSeg v1.0 on %d volumes (model loaded once) …", len(pending_pairs))
    try:
        predict(
            path_images=in_list_path,
            path_segmentations=out_list_path,
            path_model_segmentation=str(model_path),
            labels_segmentation=str(labels_dir / "synthseg_segmentation_labels.npy"),
            robust=False,
            fast=True,
            v1=True,
            do_parcellation=False,
            n_neutral_labels=18,
            names_segmentation=str(labels_dir / "synthseg_segmentation_names.npy"),
            labels_denoiser=str(labels_dir / "synthseg_denoiser_labels_2.0.npy"),
            topology_classes=str(labels_dir / "synthseg_topological_classes.npy"),
            path_posteriors=None,
            path_resampled=None,
            path_volumes=None,
            path_model_parcellation=None,
            labels_parcellation=None,
            names_parcellation=None,
            path_model_qc=None,
            labels_qc=None,
            path_qc_scores=None,
            names_qc=None,
            cropping=None,
            ct=False,
        )
    finally:
        import os
        os.unlink(in_list_path)
        os.unlink(out_list_path)

    n_done = sum(1 for _, out in pending_pairs if out.exists())
    log.info("Done. %d/%d segs written to %s", n_done, len(pending_pairs), args.output_root)


if __name__ == "__main__":
    main()
