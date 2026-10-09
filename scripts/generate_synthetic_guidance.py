#!/usr/bin/env python
"""
Generate synthetic contrasts directly from guidance maps (no U-Net forward pass).

Pipeline per variant:
  1. Run target_generator to get the raw guidance map (piecewise histogram remap)
  2. Upsample 2× with trilinear interpolation
  3. Apply separable 1D Gaussian blur (σ=1.0 px in upsampled space ≈ 0.5 px native)
  4. Downsample back to original resolution with trilinear interpolation
  5. Zero background voxels (< dark_threshold)

Skips the U-Net entirely — much faster than full synthesis.
The upsample→blur→downsample pipeline smooths chunk-boundary
discontinuities in the piecewise intensity remap.

With --lhc, (mu, alpha) draws use the same Sobol sequence as the standard
generator script, so guidance variants are directly comparable.

Output:
  data/ON-Harmony/derivatives/synthetic_{generator}_guidance[_lhc]/sub-*/ses-*/
  sub-{sub}_ses-{ses}_run-{i:02d}_syn-T1w.nii.gz

Usage (4 GPUs, parallelised over volumes):
  for rank in 0 1 2 3; do
    run_job --gpus 1 --slot $rank --wait --log /tmp/guidance_v23_1_r${rank}.log -- \
      .venv/bin/python scripts/generate_synthetic_guidance.py \
      --generator v23_1 --lhc --rank $rank --world-size 4 --device cuda:0 &
  done; wait
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityd, EnsureTyped,
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

UPSAMPLE_FACTOR  = 2
GAUSSIAN_SIGMA   = 1.0   # px in upsampled space (≈ 0.5 px at native resolution)
GAUSSIAN_TRUNCATE = 3.0  # kernel radius = truncate × sigma

# ─── Preprocessing ────────────────────────────────────────────────────────────

_PREPROCESS = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    EnsureTyped(keys=["image"], data_type="tensor"),
])

# ─── Separable 1D Gaussian blur ───────────────────────────────────────────────

def _gaussian_kernel_1d(sigma: float, truncate: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(truncate * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _gaussian_blur_3d_separable(x: torch.Tensor, sigma: float, truncate: float = GAUSSIAN_TRUNCATE) -> torch.Tensor:
    """Separable 1D Gaussian blur along each spatial axis. x: B×C×D×H×W."""
    k1d = _gaussian_kernel_1d(sigma, truncate, x.device)
    pad  = len(k1d) // 2
    B, C, D, H, W = x.shape
    y = x.view(B * C, 1, D, H, W)
    y = F.conv3d(y, k1d.view(1, 1, -1, 1, 1), padding=(pad, 0, 0))
    y = F.conv3d(y, k1d.view(1, 1, 1, -1, 1), padding=(0, pad, 0))
    y = F.conv3d(y, k1d.view(1, 1, 1, 1, -1), padding=(0, 0, pad))
    return y.view(B, C, D, H, W)

# ─── Guidance synthesis pipeline ──────────────────────────────────────────────

@torch.no_grad()
def _synthesize_guidance(
    wrapper, hist_module, x: torch.Tensor,
    num_bins: int, num_chunks: int, dark_threshold: float,
    mu_lhc: torch.Tensor | None = None,
    alpha_lhc_raw: torch.Tensor | None = None,
    blur_sigma_range: tuple = (GAUSSIAN_SIGMA, GAUSSIAN_SIGMA),
    resolution_zoom_range: tuple = (1.0, 1.0),
    native_blur: bool = False,
    labels: torch.Tensor | None = None,
) -> list[np.ndarray]:
    """
    Run target_generator to get guidance maps, then apply blur + optional resolution sim.

    native_blur=True: blur at native resolution (fast, fine for smooth generators like V23/V25).
    native_blur=False: upsample 2× → blur → downsample (for label-based generators with hard edges).
    """
    import random
    _, _, guidance_map = wrapper.target_generator(
        input_images=x, num_bins=num_bins, num_chunks=num_chunks,
        dark_threshold=dark_threshold, hist_module=hist_module,
        return_guidance_map=True, labels=labels,
        mu_lhc=mu_lhc, alpha_lhc_raw=alpha_lhc_raw,
    )
    g = guidance_map.float()           # B×1×D×H×W, [0, 1]
    orig_size = g.shape[2:]

    # Log-skewed blur: 30% no blur, 70% log-uniform in [sigma_min, sigma_max].
    # Produces many low-blur samples while still allowing heavy blur occasionally.
    import math
    if isinstance(blur_sigma_range, list):
        sigma = random.choice(blur_sigma_range)
    else:
        lo, hi = blur_sigma_range
        if lo == hi:
            sigma = lo
        elif random.random() < 0.30:
            sigma = 0.0
        else:
            sigma = math.exp(random.uniform(math.log(lo), math.log(hi)))

    if native_blur:
        # Blur directly at native resolution — 8× cheaper than upsample+blur+downsample.
        # Safe for chunk/intensity-based generators that produce no hard voxel-edge artifacts.
        zoom = random.uniform(*resolution_zoom_range)
        if zoom < 0.99:
            low_size = tuple(max(1, int(s * zoom)) for s in orig_size)
            g = F.interpolate(g, size=low_size, mode="trilinear", align_corners=False)
        if sigma > 0.0:
            g = _gaussian_blur_3d_separable(g, sigma=sigma)
        if zoom < 0.99:
            g = F.interpolate(g, size=orig_size, mode="trilinear", align_corners=False)
        out = g
    else:
        # Legacy path: upsample 2× → blur → downsample (smooths hard label boundaries).
        zoom = random.uniform(*resolution_zoom_range)
        if zoom < 0.99:
            low_size = tuple(max(1, int(s * zoom)) for s in orig_size)
            g = F.interpolate(g, size=low_size, mode="trilinear", align_corners=False)
        up = F.interpolate(g, scale_factor=UPSAMPLE_FACTOR, mode="trilinear", align_corners=False)
        if sigma > 0.0:
            up = _gaussian_blur_3d_separable(up, sigma=sigma)
        out = F.interpolate(up, size=orig_size, mode="trilinear", align_corners=False)

    out = out.clamp(0.0, 1.0)
    out = torch.where(x[:, :1] < dark_threshold, torch.zeros_like(out), out)

    return [out[b].squeeze().cpu().numpy() for b in range(out.shape[0])]

# ─── Seg loading helper ───────────────────────────────────────────────────────

def _load_seg_tensor(
    seg_path: Path, target_shape: tuple, device: torch.device
) -> torch.Tensor | None:
    """Load a SynthSeg segmentation, reorient to RAS, resize to target_shape.
    Returns a 1×1×D×H×W long tensor, or None if the file is missing."""
    if not seg_path.exists():
        log.warning("Seg file not found: %s", seg_path)
        return None
    seg_nii = nib.load(str(seg_path))
    seg_nii = nib.as_closest_canonical(seg_nii)          # → RAS
    seg_data = np.round(seg_nii.get_fdata()).astype(np.int32)
    seg_t = torch.from_numpy(seg_data).float().unsqueeze(0).unsqueeze(0)  # 1×1×D×H×W
    if tuple(seg_t.shape[2:]) != tuple(target_shape):
        seg_t = F.interpolate(seg_t, size=target_shape, mode="nearest")
    return seg_t.long().to(device)


# ─── Checkpoint discovery ─────────────────────────────────────────────────────

def find_latest_ckpt(base: Path) -> Path:
    run_pattern = re.compile(r"^run(\d+)$")
    runs = [(int(m.group(1)), child)
            for child in sorted(base.iterdir())
            if (m := run_pattern.match(child.name))]
    if not runs:
        raise FileNotFoundError(f"No run directories found under {base}")
    _, latest = max(runs, key=lambda t: t[0])
    candidates = sorted(latest.glob("best_loss*.ckpt")) or list(latest.glob("last.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found in {latest}")
    ckpt = candidates[-1]
    log.info("Using checkpoint: %s", ckpt)
    return ckpt


def _ckpt_base(generator: str) -> Path:
    if generator == "v19":
        return PROJECT_ROOT / "checkpoints" / "on_harmony" / "on_harmony_v19" / "generator" / "t1w"
    if generator in ("v23_4",) or generator.startswith("v24") or generator.startswith("v25") or generator.startswith("v26") or generator.startswith("v27") or generator.startswith("v28"):
        # These variants borrow the v23_3 checkpoint (same architecture, different synthesis params).
        return PROJECT_ROOT / "checkpoints" / "on_harmony" / "generator" / "v23_3" / "t1w"
    return PROJECT_ROOT / "checkpoints" / "on_harmony" / "generator" / generator / "t1w"

# ─── Model loading ────────────────────────────────────────────────────────────

def load_module(ckpt_path: Path, device: torch.device, generator: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from src.training.lightning_modules import MRISynthesisLightning

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"), version_base=None):
        cfg = compose("config", overrides=["task=generator", f"generator={generator}", "data=on_harmony"])

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    module = MRISynthesisLightning(cfg)
    module.load_state_dict(ckpt["state_dict"], strict=True)
    module.eval().to(device)
    log.info("Model loaded to %s", device)
    return module

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generator", type=str, default="v23_1",
                   choices=["v19", "v19_c", "v22_1", "v22_2", "v23_1", "v23_2", "v23_3", "v23_4",
                            "v26_4", "v26_5", "v26_6", "v26_7", "v26_8", "v26_9", "v26_10",
                            "v26_11", "v26_12", "v26_13", "v26_14", "v26_15",
                            "v28_1", "v28_2", "v28_3", "v28_4",
                            "v24", "v24_t2w", "v24_desc", "v24_inv", "v24_pdw", "v24_nm", "v24_nm2", "v24_nm3",
                            "v24_t2w_label", "v25_1", "v25_2", "v26_1", "v26_2", "v26_3",
                            "v27a", "v27a_bis"])
    p.add_argument("--lhc", action="store_true",
                   help="Use Sobol quasi-random (mu, alpha) sampling")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--bids-root", type=Path, default=PROJECT_ROOT / "data" / "ON-Harmony")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--n-variants", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--variant-batch-size", type=int, default=10,
                   help="Variants per target_generator call (no U-Net, so memory is cheap)")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--seg-root", type=Path, default=None,
                   help="Root of SynthSeg segmentation derivatives "
                        "(sub-*/ses-*/<t1w_name>.nii.gz). Required for v27a/v27a_bis.")
    p.add_argument("--resolution-diversity", action="store_true",
                   help="v28_2/v28_4: randomly save 50%% of subjects at 2–4 mm voxel size "
                        "(all 10 variants of a subject share the same resolution). "
                        "Improves HOG3D coverage for bold/DWI/EPI without in-pipeline zoom.")
    return p.parse_args()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.output_dir is None:
        lhc_suffix = "_lhc" if args.lhc else ""
        args.output_dir = (PROJECT_ROOT / "data" / "ON-Harmony" / "derivatives"
                           / f"synthetic_{args.generator}_guidance{lhc_suffix}")

    if args.ckpt is None:
        args.ckpt = find_latest_ckpt(_ckpt_base(args.generator))

    module = load_module(args.ckpt, device, generator=args.generator)

    cfg_g = module.cfg.model.generator
    num_bins       = int(cfg_g.num_bins)
    num_chunks     = int(cfg_g.num_chunks)
    dark_threshold = float(cfg_g.dark_threshold)
    wrapper = module.compiled_wrapper
    if hasattr(wrapper, "_orig_mod"):
        wrapper = wrapper._orig_mod

    # Per-generator blur / resolution diversity settings.
    # native_blur=True: blur at native res (fast, safe for smooth intensity-based generators).
    _BLUR_SIGMA_RANGES = {
        "v23_3": (0.3, 3.0),
        "v23_4": (0.3, 3.0),  # log-skewed: 30% no blur, 70% log-uniform
        "v25_1": (0.3, 3.0),
        "v25_2": (0.3, 3.0),  # same blur schedule as v23_4
        "v26_1": (0.3, 3.0),
        "v26_2": [0.0, 0.5, 1.0, 1.5],
        "v26_3": [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],  # weighted toward no/low blur, max 0.8
        "v26_4":    [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        "v26_5":    [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        "v26_6":    [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        "v26_7":    [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        "v26_8":    [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        "v26_9":    [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        "v26_10":   [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        "v26_11":   [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        "v26_12":   [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        "v26_13":   [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        "v26_14":   [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        "v26_15":   [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        # v28 family — HOG-space improvement
        "v28_1":    (0.3, 3.0),   # wide blur range (v28_1 only: aggressive zoom)
        "v28_2":    [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],  # same blur as v26_6; resolution via --resolution-diversity
        "v28_3":    [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],  # same; susceptibility handled by target generator
        "v28_4":    [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],  # v28_2 + v28_3
        "v27a":     [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        "v27a_bis": [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
    }
    _ZOOM_RANGES = {
        "v23_3": (0.4, 1.0),
        "v23_4": (0.4, 1.0),
        "v25_1": (0.4, 1.0),
        "v25_2": (0.4, 1.0),
        "v26_1": (0.4, 1.0),
        "v26_2": (0.4, 1.0),
        "v26_3": (0.4, 1.0),
        "v26_4":    (0.4, 1.0),
        "v26_5":    (0.4, 1.0),
        "v26_6":    (0.4, 1.0),
        "v26_7":    (0.4, 1.0),
        "v26_8":    (0.4, 1.0),
        "v26_9":    (0.4, 1.0),
        "v26_10":   (0.4, 1.0),
        "v26_11":   (0.4, 1.0),
        "v26_12":   (0.4, 1.0),
        "v26_13":   (0.4, 1.0),
        "v26_14":   (0.4, 1.0),
        "v26_15":   (0.4, 1.0),
        "v28_1":    (0.20, 1.0),  # zoom down to 0.20 → 5 mm effective, covers bold/EPI regime
        "v28_2":    (1.0, 1.0),   # no in-pipeline zoom; true resolution via --resolution-diversity flag
        "v28_3":    (0.4, 1.0),   # same as v26_6
        "v28_4":    (1.0, 1.0),   # no in-pipeline zoom; true resolution via --resolution-diversity flag
        "v27a":     (0.4, 1.0),
        "v27a_bis": (0.4, 1.0),
    }
    _NATIVE_BLUR = {"v23_3", "v23_4", "v25_1", "v25_2", "v26_1", "v26_2", "v26_3", "v26_4",
                    "v26_5", "v26_6", "v26_7", "v26_8", "v26_9", "v26_10",
                    "v26_11", "v26_12", "v26_13", "v26_14", "v26_15",
                    "v28_1", "v28_2", "v28_3", "v28_4", "v27a", "v27a_bis"}
    blur_sigma_range      = _BLUR_SIGMA_RANGES.get(args.generator, (GAUSSIAN_SIGMA, GAUSSIAN_SIGMA))
    resolution_zoom_range = _ZOOM_RANGES.get(args.generator, (1.0, 1.0))
    native_blur           = args.generator in _NATIVE_BLUR

    if isinstance(blur_sigma_range, list):
        log.info(f"Guidance pipeline: using blur sigma from list {blur_sigma_range}")
    else:
        log.info("Guidance pipeline: upsample %d× → Gaussian σ∈[%.1f,%.1f]px → downsample (trilinear)",
                 UPSAMPLE_FACTOR, *blur_sigma_range)

    # ── Discover volumes ──
    t1w_files = sorted(args.bids_root.glob("sub-*/ses-*/anat/*_T1w.nii.gz"))
    log.info("Found %d T1w volumes", len(t1w_files))
    if args.dry_run:
        t1w_files, args.n_variants = t1w_files[:1], 1
    elif args.limit:
        t1w_files = t1w_files[: args.limit]

    def _is_complete(p: Path) -> bool:
        sub, ses = p.parts[-4], p.parts[-3]
        d = args.output_dir / sub / ses
        return all((d / f"{sub}_{ses}_run-{i:02d}_syn-T1w.nii.gz").exists()
                   for i in range(args.n_variants))

    pending = [p for p in t1w_files if not _is_complete(p)]
    log.info("%d pending / %d total  (%d already complete)",
             len(pending), len(t1w_files), len(t1w_files) - len(pending))

    if args.world_size > 1:
        pending = pending[args.rank :: args.world_size]
        log.info("Rank %d/%d: assigned %d volumes", args.rank, args.world_size, len(pending))

    if not pending:
        log.info("Nothing to do.")
        return

    # ── Sobol LHC samples ──
    lhc_samples = None
    if args.lhc:
        from scipy.stats.qmc import Sobol
        n_total = len(pending) * args.n_variants
        m = max(1, math.ceil(math.log2(max(n_total, 2))))
        sobol = Sobol(d=16, scramble=True, seed=42)
        lhc_samples = sobol.random_base2(m=m)
        log.info("LHC: Sobol sequence — %d samples (dim=16)", len(lhc_samples))

    # ── Cache preprocessing ──
    log.info("Building preprocessing cache (%d volumes, %d workers) …",
             len(pending), args.num_workers)
    dataset = CacheDataset(
        data=[{"image": str(p)} for p in pending],
        transform=_PREPROCESS,
        cache_rate=1.0,
        num_workers=args.num_workers,
    )
    log.info("Cache ready. Starting guidance synthesis …")

    generated = 0
    for idx in tqdm(range(len(dataset)), desc="Volumes", unit="vol"):
        nii_path = pending[idx]
        sub, ses = nii_path.parts[-4], nii_path.parts[-3]
        out_dir = args.output_dir / sub / ses
        out_dir.mkdir(parents=True, exist_ok=True)

        missing = [i for i in range(args.n_variants)
                   if not (out_dir / f"{sub}_{ses}_run-{i:02d}_syn-T1w.nii.gz").exists()]
        if not missing:
            continue

        img_meta = dataset[idx]["image"]
        affine = np.array(img_meta.affine) if hasattr(img_meta, "affine") else np.eye(4)
        x = (img_meta.as_tensor() if hasattr(img_meta, "as_tensor") else img_meta
             ).float().unsqueeze(0).to(device)

        # Resolution diversity (v28_2, v28_4): sample one target voxel size per subject.
        # All 10 variants of this subject are saved at the same resolution so the HOG
        # extractor sees a volume at a genuinely different native pitch (not just blurred 1mm).
        # Weights: 50% native 1mm, 17% 2mm (DWI-like), 25% 3mm (bold-like), 8% 4mm (EPI-like).
        save_voxel_mm = 1.0
        if args.resolution_diversity:
            import random as _random
            save_voxel_mm = float(_random.choices(
                [1.0, 2.0, 3.0, 4.0], weights=[50, 17, 25, 8]
            )[0])

        # Load corresponding segmentation (required for v27a / v27a_bis)
        seg_tensor = None
        if args.seg_root is not None:
            seg_path = args.seg_root / sub / ses / nii_path.name
            seg_tensor = _load_seg_tensor(seg_path, x.shape[2:], device)

        vbs = args.variant_batch_size
        for chunk_start in range(0, len(missing), vbs):
            chunk = missing[chunk_start : chunk_start + vbs]

            mu_batch = alpha_batch = None
            if lhc_samples is not None:
                mu_list, alpha_list = [], []
                for i in chunk:
                    sample_idx = (idx * args.n_variants + i) % len(lhc_samples)
                    params = lhc_samples[sample_idx]
                    mu_list.append(torch.from_numpy(params[:8].astype(np.float32)))
                    alpha_list.append(torch.from_numpy(params[8:].astype(np.float32)))
                mu_batch    = torch.stack(mu_list)
                alpha_batch = torch.stack(alpha_list)

            x_batch = x.expand(len(chunk), -1, -1, -1, -1).contiguous()
            seg_batch = (seg_tensor.expand(len(chunk), -1, -1, -1, -1).contiguous()
                         if seg_tensor is not None else None)

            try:
                synths = _synthesize_guidance(
                    wrapper, module.histogram_module, x_batch,
                    num_bins, num_chunks, dark_threshold,
                    mu_lhc=mu_batch, alpha_lhc_raw=alpha_batch,
                    blur_sigma_range=blur_sigma_range,
                    resolution_zoom_range=resolution_zoom_range,
                    native_blur=native_blur,
                    labels=seg_batch,
                )
                def _save(args_tuple):
                    i, synth = args_tuple
                    out_path = out_dir / f"{sub}_{ses}_run-{i:02d}_syn-T1w.nii.gz"
                    save_arr = synth.astype(np.float32)
                    save_aff = affine

                    # True resolution downsampling: resample to target voxel size so the
                    # HOG extractor sees a genuinely coarse-resolution volume (not a blurred 1mm).
                    if save_voxel_mm > 1.01:
                        native_vox = float(np.sqrt((affine[:3, 0]**2).sum()))  # voxel size from affine
                        scale = save_voxel_mm / max(native_vox, 0.5)
                        new_shape = tuple(max(1, round(s / scale)) for s in save_arr.shape)
                        t = torch.from_numpy(save_arr)[None, None]
                        save_arr = F.interpolate(
                            t.float(), size=new_shape, mode="trilinear", align_corners=False
                        )[0, 0].numpy()
                        save_aff = affine.copy()
                        save_aff[:3, :3] *= scale   # expand voxel-to-mm mapping

                    out_nii = nib.Nifti1Image(save_arr, save_aff)
                    out_nii.header.set_data_dtype(np.float32)
                    nib.save(out_nii, str(out_path))

                with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
                    list(pool.map(_save, zip(chunk, synths)))
                generated += len(chunk)
            except Exception as exc:
                log.error("Guidance synthesis failed %s runs %s: %s",
                          nii_path.name, [f"{i:02d}" for i in chunk], exc)

    log.info("Done. Generated %d files → %s", generated, args.output_dir)


if __name__ == "__main__":
    main()
