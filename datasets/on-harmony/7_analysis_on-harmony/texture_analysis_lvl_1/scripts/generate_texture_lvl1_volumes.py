#!/usr/bin/env python3
"""Generate source-aligned augmented copies of the 84 on-harmony T1w volumes using
the EXACT AugLab GPU training operators (spatial OFF), for texture_analysis_lvl_1.

For each (method, volume) we run the method's ComposeTransforms N times with a fresh
RNG seed and save each output voxel-aligned to the source T1w (identical shape+affine).

Methods (driven by the derived configs in ../configs/):
  palette, synthseg_em, synthseg_noem, auglab_default

Run (per GPU, rank-sharded over the 84 volumes):
  set_slot R .venv/bin/python generate_texture_lvl1_volumes.py --rank R --world-size 4

See GENERATION_PROMPT.md. Normalization matches nnUNetPlans for Dataset031
(ZScoreNormalization, use_mask_for_norm=False -> whole-image z-score).
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import nibabel as nib

REPO = Path(__file__).resolve().parents[5]
ANALYSIS = Path(__file__).resolve().parents[1]
AUGLAB = REPO / "sub-workspaces/auglab_workspace/AugLab"
IMAGES_TR = REPO / ("datasets/on-harmony/2_nnUNet_on-harmony/raw/"
                    "Dataset031_OnHarmonyT1w31/imagesTr")
LABELS_TR = REPO / ("datasets/on-harmony/2_nnUNet_on-harmony/raw/"
                    "Dataset031_OnHarmonyT1w31/labelsTr")

METHODS = ["palette", "synthseg_em", "synthseg_noem", "auglab_default"]


def zscore_wholeimage(image: np.ndarray) -> np.ndarray:
    """nnUNet ZScoreNormalization with use_mask_for_norm=False (Dataset031 plans)."""
    image = image.astype(np.float32)
    mean = image.mean()
    std = image.std()
    return (image - mean) / max(std, 1e-8)


def list_volumes():
    vols = sorted(IMAGES_TR.glob("*_0000.nii.gz"))
    if not vols:
        raise FileNotFoundError(f"No *_0000.nii.gz in {IMAGES_TR}")
    return vols


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--n-variants", type=int, default=10)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--config-dir", type=Path, default=ANALYSIS / "configs")
    ap.add_argument("--out-root", type=Path,
                    default=ANALYSIS / "data/generated")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap #volumes (after sharding) for smoke tests")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output file already exists")
    args = ap.parse_args()

    # set_slot does not set CUDA_VISIBLE_DEVICES; pin one GPU per rank.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.rank))

    sys.path.insert(0, str(AUGLAB))
    import torch
    from auglab.transforms.gpu.transforms import AugTransformsGPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rank {args.rank}] device={device} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    vols = list_volumes()
    vols = vols[args.rank::args.world_size]
    if args.limit is not None:
        vols = vols[:args.limit]
    print(f"[rank {args.rank}] {len(vols)} volumes, methods={args.methods}, "
          f"N={args.n_variants}", flush=True)

    # Build one transform per method (reused across all volumes).
    transforms = {}
    for m in args.methods:
        cfg = args.config_dir / f"{m}.json"
        transforms[m] = AugTransformsGPU(json_path=str(cfg)).to(device).eval()
        print(f"[rank {args.rank}] built transform '{m}' from {cfg.name}", flush=True)

    counts = {m: 0 for m in args.methods}
    for vi, vpath in enumerate(vols):
        stem = vpath.name.replace("_0000.nii.gz", "")           # sub-XXXX_ses-YYYY_T1w
        subj = stem                                              # canonical KEY (incl. _T1w)
        lpath = LABELS_TR / f"{stem}.nii.gz"
        if not lpath.exists():
            print(f"[rank {args.rank}] MISSING LABEL for {stem}, skipping", flush=True)
            continue

        t1w_nii = nib.load(str(vpath))
        src_affine = t1w_nii.affine
        src_shape = t1w_nii.shape
        t1w = t1w_nii.get_fdata(dtype=np.float32)
        lbl = np.round(nib.load(str(lpath)).get_fdata()).astype(np.int64)
        assert lbl.shape == t1w.shape, f"label/img shape mismatch for {stem}"

        img_norm = zscore_wholeimage(t1w)
        img_t = torch.from_numpy(img_norm)[None, None].to(device)      # (1,1,D,H,W)
        lbl_t = torch.from_numpy(lbl)[None, None].to(device)           # (1,1,D,H,W)

        for mi, m in enumerate(args.methods):
            out_dir = args.out_root / m / subj
            out_dir.mkdir(parents=True, exist_ok=True)
            for run in range(args.n_variants):
                out_path = out_dir / f"{subj}_run-{run:02d}.nii.gz"
                if out_path.exists() and not args.overwrite:
                    counts[m] += 1
                    continue
                # Fresh, deterministic-per-(volume,method,run) seed.
                # Deterministic per (global volume index, method, run).
                gvi = args.rank + vi * args.world_size
                seed = (args.seed_base
                        + gvi * 100000
                        + (METHODS.index(m) + 1) * 1000
                        + run) & 0x7FFFFFFF
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                try:
                    with torch.no_grad():
                        out_img, _ = transforms[m](img_t.clone(), lbl_t.clone())
                except Exception:
                    print(f"[rank {args.rank}] ERROR {m}/{subj} run{run}:\n"
                          f"{traceback.format_exc()}", flush=True)
                    continue
                out_np = out_img[0, 0].detach().cpu().numpy().astype(np.float32)

                # ---- alignment sanity asserts (per write) ----
                assert out_np.shape == tuple(src_shape), \
                    f"shape drift {out_np.shape} vs {src_shape} ({m}/{subj})"
                out_nii = nib.Nifti1Image(out_np, src_affine)
                assert np.allclose(out_nii.affine, src_affine), \
                    f"affine drift ({m}/{subj})"
                nib.save(out_nii, str(out_path))
                counts[m] += 1
        if (vi + 1) % 5 == 0 or vi == len(vols) - 1:
            print(f"[rank {args.rank}] {vi+1}/{len(vols)} volumes done; "
                  f"counts={counts}", flush=True)

    print(f"[rank {args.rank}] DONE counts={counts}", flush=True)


if __name__ == "__main__":
    main()
