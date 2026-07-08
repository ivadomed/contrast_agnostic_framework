#!/usr/bin/env python3
"""
Stage 1 (open-ms Pillar-2): generate source-aligned augmented copies of the open-ms scans with
the EXACT AugLab GPU training operators (spatial OFF), for the 4 methods
(palette, synthseg_em, synthseg_noem, auglab_default). Mirrors the on-harmony texture generator.

open-ms provides ONLY sparse lesion labels (its brainmask is a computed brain-extraction, not an
annotation, so we do not use it). The seg passed to the transforms is therefore the lesion mask only:
    background = 0,  lesion = 1
SynthSeg-EM estimates its per-label GMM from the REAL image intensities, so it still fills the whole
image sensibly from a coarse label map; SynthSeg-noEM (pure parametric) is expected to be poor on
these sparse labels — that is the point of testing on a sparse-annotation dataset. PALETTE's
label-free core is unaffected.

Sources: FLAIR and T1w (co-registered → same masks). Output (key encodes contrast):
  <out-root>/<method>/<key>/<key>_run-NN.nii.gz     key = sub-patientNN_<CONTRAST>

Usage (romane, 4 GPUs):
  for R in 0 1 2 3; do
    set_slot $R .venv/bin/python generate_openms_volumes.py --rank $R --world-size 4 & done; wait
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

REPO = Path(__file__).resolve().parents[5]
ANALYSIS = Path(__file__).resolve().parents[1]
AUGLAB = REPO / "sub-workspaces/auglab_workspace/AugLab"
BIDS = REPO / "datasets/open-ms/1_BIDS_open-ms/open-ms-brain"
RAW = REPO / "datasets/open-ms/0_raw_open-ms"
LESION_DIR = BIDS / "derivatives" / "manual_masks"
DEFAULT_CONFIGS = REPO / "datasets/on-harmony/7_analysis_on-harmony/texture_analysis_lvl_1/configs"
METHODS = ["palette", "synthseg_em", "synthseg_noem", "auglab_default", "v26_6_2_noisefill_v2"]
SOURCES = ["FLAIR", "T1w"]


def zscore(image):
    image = image.astype(np.float32)
    return (image - image.mean()) / max(image.std(), 1e-8)


def _mask_to(mask_nii, ref_nii):
    if mask_nii.shape[:3] != ref_nii.shape[:3] or not np.allclose(mask_nii.affine, ref_nii.affine, atol=1e-3):
        mask_nii = resample_from_to(mask_nii, ref_nii, order=0)
    return np.round(mask_nii.get_fdata()).astype(np.int32)


def list_sources(contrasts):
    """[(key, image, lesion)] for scans that have a lesion mask."""
    items = []
    for c in contrasts:
        for img in sorted(BIDS.glob(f"sub-*/anat/*_{c}.nii.gz")):
            sub = img.name.replace(f"_{c}.nii.gz", "")
            les = LESION_DIR / sub / "anat" / f"{sub}_FLAIR_dseg.nii.gz"
            if not les.exists():
                print(f"MISSING lesion mask for {img.name}, skipping", flush=True); continue
            items.append((img.name.replace(".nii.gz", ""), img, les))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--n-variants", type=int, default=10)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    ap.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIGS)
    # shared by both open-ms Pillar-1 (texture_analysis_lvl_1) and Pillar-2 (this coverage
    # analysis) — lives one level up, at 7_analysis_open-ms/data/, not under this pillar.
    ap.add_argument("--out-root", type=Path, default=REPO / "datasets/open-ms/7_analysis_open-ms/data/generated")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.rank))
    sys.path.insert(0, str(AUGLAB))
    import torch
    from auglab.transforms.gpu.transforms import AugTransformsGPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = list_sources(args.sources)[args.rank::args.world_size]
    if args.limit is not None:
        items = items[:args.limit]
    print(f"[rank {args.rank}] device={device} {len(items)} sources, methods={args.methods}, "
          f"sources={args.sources}, N={args.n_variants}", flush=True)

    transforms = {}
    for m in args.methods:
        transforms[m] = AugTransformsGPU(json_path=str(args.config_dir / f"{m}.json")).to(device).eval()
        print(f"[rank {args.rank}] built '{m}'", flush=True)

    counts = {m: 0 for m in args.methods}
    for ii, (key, img_path, les_path) in enumerate(items):
        img_nii = nib.load(str(img_path))
        src_affine, src_shape = img_nii.affine, img_nii.shape
        img = img_nii.get_fdata(dtype=np.float32)

        lesion = _mask_to(nib.load(str(les_path)), img_nii) > 0
        seg = np.zeros(img.shape, dtype=np.int64)            # lesion-only: background=0, lesion=1
        seg[lesion] = 1

        img_t = torch.from_numpy(zscore(img))[None, None].to(device)
        lbl_t = torch.from_numpy(seg)[None, None].to(device)

        for m in args.methods:
            out_dir = args.out_root / m / key
            out_dir.mkdir(parents=True, exist_ok=True)
            for run in range(args.n_variants):
                out_path = out_dir / f"{key}_run-{run:02d}.nii.gz"
                if out_path.exists() and not args.overwrite:
                    counts[m] += 1; continue
                gvi = args.rank + ii * args.world_size
                seed = (args.seed_base + gvi * 100000 + (METHODS.index(m) + 1) * 1000 + run) & 0x7FFFFFFF
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
                try:
                    with torch.no_grad():
                        out_img, _ = transforms[m](img_t.clone(), lbl_t.clone())
                except Exception:
                    print(f"[rank {args.rank}] ERROR {m}/{key} run{run}:\n{traceback.format_exc()}", flush=True)
                    continue
                out_np = out_img[0, 0].detach().cpu().numpy().astype(np.float32)
                assert out_np.shape == tuple(src_shape), f"shape drift ({m}/{key})"
                nib.save(nib.Nifti1Image(out_np, src_affine), str(out_path))
                counts[m] += 1
        if (ii + 1) % 5 == 0 or ii == len(items) - 1:
            print(f"[rank {args.rank}] {ii+1}/{len(items)}; counts={counts}", flush=True)

    print(f"[rank {args.rank}] DONE counts={counts}", flush=True)


if __name__ == "__main__":
    main()
