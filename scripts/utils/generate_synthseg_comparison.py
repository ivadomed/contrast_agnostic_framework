#!/usr/bin/env python
"""
Generate synthetic images using SynthSeg's BrainGenerator for manifold comparison.

Two modes:
  --mode modeA   Dense SynthSeg segmentation → BrainGenerator (uniform GMM priors).
                 Requires segs in derivatives/synthseg_segs/ (run run_synthseg_predict.py first).
  --mode modeB   EM-clustered intensity label map → BrainGenerator (uniform GMM priors).
                 No pre-computed segs needed; runs sklearn GMM on each T1w directly.

Output:
  data/ON-Harmony/derivatives/synthseg_modeA/sub-*/ses-*/sub-*_ses-*_run-NN_syn-T1w.nii.gz
  data/ON-Harmony/derivatives/synthseg_modeB_em/sub-*/ses-*/sub-*_ses-*_run-NN_syn-T1w.nii.gz

Spatial augmentations are disabled so outputs are in the same space as the input T1w
(required for feature extraction with atlas-based region masks).

Usage (4 GPUs):
  for rank in 0 1 2 3; do
    set_slot $rank .venv/bin/python scripts/generate_synthseg_comparison.py \\
      --mode modeA --rank $rank --world-size 4 > /tmp/ss_modeA_r${rank}.log 2>&1 &
  done
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import ctypes, glob, os as _os
# sudo strips LD_LIBRARY_PATH; load venv CUDA libs with ctypes so TF finds the GPU.
for _lib in sorted(glob.glob(str(
        Path(__file__).resolve().parents[1] /
        ".venv/lib/python3.12/site-packages/nvidia/*/lib/*.so*"))):
    try: ctypes.CDLL(_lib)
    except OSError: pass
_os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
SYNTHSEG_HOME = PROJECT_ROOT / "SynthSeg"
BIDS_ROOT     = PROJECT_ROOT / "data" / "ON-Harmony"
SEG_ROOT      = BIDS_ROOT / "derivatives" / "synthseg_segs"
LABELS_DIR    = SYNTHSEG_HOME / "data" / "labels_classes_priors"

# Standard FreeSurfer-compatible label IDs produced by SynthSeg v1.0
SYNTHSEG_V1_LABELS = str(LABELS_DIR / "synthseg_segmentation_labels.npy")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["modeA", "modeB"], required=True,
                   help="modeA: SynthSeg segs → BrainGenerator; modeB: EM labels → BrainGenerator")
    p.add_argument("--n-variants", type=int, default=10,
                   help="Synthetic variants per subject (default 10)")
    p.add_argument("--bids-root", type=Path, default=BIDS_ROOT)
    p.add_argument("--seg-root", type=Path, default=SEG_ROOT,
                   help="Root of SynthSeg segs (modeA only)")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--em-components", type=int, default=None,
                   help="Fixed number of EM clusters for modeB. "
                        "Default: random in [3, 10] per subject.")
    return p.parse_args()


# ─── EM label map creation (Mode B) ──────────────────────────────────────────

def _make_em_label_map(t1w_data: np.ndarray, n_components: int | None) -> np.ndarray:
    """
    Cluster T1w foreground intensities with a Gaussian Mixture Model.
    Returns an integer label map (0=background, 1..K=EM clusters).
    """
    from sklearn.mixture import GaussianMixture

    if n_components is None:
        n_components = int(np.random.randint(3, 11))  # [3, 10] inclusive

    fg_mask = t1w_data > t1w_data[t1w_data > 0].mean() * 0.05  # rough brain mask
    fg_vals = t1w_data[fg_mask].reshape(-1, 1)

    # Subsample if very large (GMM is O(n·K²))
    max_samples = 50_000
    if len(fg_vals) > max_samples:
        idx = np.random.choice(len(fg_vals), max_samples, replace=False)
        fg_sub = fg_vals[idx]
    else:
        fg_sub = fg_vals

    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          max_iter=100, random_state=42)
    gmm.fit(fg_sub)

    label_map = np.zeros_like(t1w_data, dtype=np.int32)
    if fg_mask.any():
        label_map[fg_mask] = gmm.predict(t1w_data[fg_mask].reshape(-1, 1)) + 1  # 1-indexed

    return label_map


# ─── BrainGenerator wrapper ──────────────────────────────────────────────────

def _synthseg_generate(
    label_map: np.ndarray,
    affine: np.ndarray,
    generation_labels: str | None,
    n_variants: int,
    tmp_seg_path: Path,
) -> list[np.ndarray]:
    """
    Instantiate BrainGenerator for one label map, generate n_variants images.

    Spatial augmentations are disabled to keep outputs aligned with the input T1w
    (required for atlas-based regional_hist_64 feature extraction).
    """
    sys.path.insert(0, str(SYNTHSEG_HOME))

    # Keras 3 removed get_shape() from KerasTensor; patch it back for SynthSeg compatibility.
    try:
        from keras.src.backend.common.keras_tensor import KerasTensor as _KT
        if not hasattr(_KT, "get_shape"):
            class _SW:
                def __init__(self, s): self._s = s
                def as_list(self): return list(self._s)
            _KT.get_shape = lambda self: _SW(self.shape)
    except Exception:
        pass

    from SynthSeg.brain_generator import BrainGenerator

    # Save label map to a temp NIfTI (BrainGenerator reads from disk)
    tmp_seg_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(label_map.astype(np.int32), affine), str(tmp_seg_path))

    gen = BrainGenerator(
        labels_dir=str(tmp_seg_path),
        generation_labels=generation_labels,
        prior_distributions="uniform",
        # Disable all spatial augmentations → output stays in T1w space
        flipping=False,
        scaling_bounds=0,
        rotation_bounds=0,
        shearing_bounds=0,
        translation_bounds=False,
        nonlin_std=0,
        randomise_res=False,
        bias_field_std=0,
        return_gradients=False,
    )

    images = []
    for _ in range(n_variants):
        img, lbl = gen.generate_brain()
        img = img.astype(np.float32)
        img[lbl == 0] = 0.0  # zero non-brain background (BrainGenerator assigns random intensities to label 0)
        images.append(img)

    return images


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    out_name = "synthseg_modeA" if args.mode == "modeA" else "synthseg_modeB_em"
    out_root = args.bids_root / "derivatives" / out_name

    t1w_files = sorted(args.bids_root.glob("sub-*/ses-*/anat/*_T1w.nii.gz"))
    log.info("Found %d T1w volumes", len(t1w_files))
    if args.limit:
        t1w_files = t1w_files[:args.limit]

    def _is_complete(p: Path) -> bool:
        sub, ses = p.parts[-4], p.parts[-3]
        d = out_root / sub / ses
        return all(
            (d / f"{sub}_{ses}_run-{i:02d}_syn-T1w.nii.gz").exists()
            for i in range(args.n_variants)
        )

    pending = [p for p in t1w_files if not _is_complete(p)]
    log.info("%d pending / %d total", len(pending), len(t1w_files))

    import os, ctypes, glob
    # set_slot controls cgroups only — set CUDA_VISIBLE_DEVICES explicitly per rank.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.rank)
    for _lib in sorted(glob.glob(str(
            PROJECT_ROOT / ".venv/lib/python3.12/site-packages/nvidia/*/lib/*.so*"))):
        try: ctypes.CDLL(_lib)
        except OSError: pass
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    if args.world_size > 1:
        pending = pending[args.rank::args.world_size]
        log.info("Rank %d/%d: %d volumes", args.rank, args.world_size, len(pending))

    if not pending:
        log.info("Nothing to do.")
        return

    import tempfile, os
    tmp_dir = Path(tempfile.mkdtemp(prefix="synthseg_gen_"))

    for vol_idx, t1w_path in enumerate(pending):
        sub, ses = t1w_path.parts[-4], t1w_path.parts[-3]
        out_dir = out_root / sub / ses
        out_dir.mkdir(parents=True, exist_ok=True)

        missing = [i for i in range(args.n_variants)
                   if not (out_dir / f"{sub}_{ses}_run-{i:02d}_syn-T1w.nii.gz").exists()]
        if not missing:
            continue

        log.info("[%d/%d] %s %s — generating runs %s",
                 vol_idx + 1, len(pending), sub, ses,
                 [f"{i:02d}" for i in missing])

        # Load T1w for Mode B (EM) or Mode A affine recovery
        t1w_nii = nib.load(str(t1w_path))
        t1w_nii = nib.as_closest_canonical(t1w_nii)
        t1w_data = t1w_nii.get_fdata(dtype=np.float32)
        affine = t1w_nii.affine

        if args.mode == "modeA":
            seg_path = args.seg_root / sub / ses / t1w_path.name
            if not seg_path.exists():
                log.warning("Seg not found, skipping: %s", seg_path)
                continue
            seg_nii = nib.load(str(seg_path))
            seg_nii = nib.as_closest_canonical(seg_nii)
            label_map = np.round(seg_nii.get_fdata()).astype(np.int32)
            affine = seg_nii.affine
            gen_labels = SYNTHSEG_V1_LABELS
        else:
            # Mode B: derive label map via EM
            label_map = _make_em_label_map(t1w_data, args.em_components)
            gen_labels = None  # BrainGenerator auto-detects from the label map

        tmp_seg = tmp_dir / f"{sub}_{ses}_tmp_labels.nii.gz"

        try:
            images = _synthseg_generate(
                label_map, affine, gen_labels,
                n_variants=len(missing),
                tmp_seg_path=tmp_seg,
            )
        except Exception as e:
            log.error("Generation failed for %s/%s: %s", sub, ses, e)
            continue
        finally:
            if tmp_seg.exists():
                tmp_seg.unlink()

        for run_idx, img in zip(missing, images):
            out_path = out_dir / f"{sub}_{ses}_run-{run_idx:02d}_syn-T1w.nii.gz"
            out_nii = nib.Nifti1Image(img, affine)
            out_nii.header.set_data_dtype(np.float32)
            nib.save(out_nii, str(out_path))

        log.info("  → saved %d files", len(missing))

    # Cleanup temp dir
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    log.info("Done. Outputs in %s", out_root)


if __name__ == "__main__":
    main()
