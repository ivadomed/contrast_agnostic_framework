#!/usr/bin/env python
"""
Level-1 texture / structure-preservation metrics for open-ms (Pillar 1), network-free.

For each generated volume and each of 2 ROIs, measure how much of the SOURCE's texture
survives in the generated volume, using contrast-AND-inversion-invariant census metrics
(rank/census math copied verbatim from on-harmony's texture_analysis_lvl_1; see
datasets/on-harmony/7_analysis_on-harmony/texture_analysis_lvl_1/LITERATURE_REVIEW.md):

  * census_r1  = |corr( rank_transform(source,1), rank_transform(synth,1) )| over the whole ROI.
                 PRIMARY for the texture-vs-texture-blind axis (image-driven ≫ SynthSeg floor).
  * census_r2  = radius-2 robustness variant.
  * census_local{LOCAL_BLOCK} = the same |corr| but computed WITHIN {LOCAL_BLOCK}³ blocks and
                 averaged — cancellation-robust (whole-ROI |corr| cancels PALETTE's signed
                 per-region α across regions; see local_abscorr). Use this for palette-vs-auglab.

NMI (Studholme MI) is deliberately NOT reported here: it is spatially blind (operates on the
1-D set of ROI intensity values, ignoring arrangement) so it is not a texture metric. It
remains in the shared on-harmony pipeline; the shared aggregate/plot scripts auto-detect which
metric columns are present, so its absence from the open-ms CSVs simply drops it from open-ms
tables/plots without affecting on-harmony.

open-ms constraint (no anatomical parcellation — only sparse binary MS lesion labels; its
brainmask is a computed extraction, not an annotation, so it is not used — same choice as
the open-ms histogram-coverage analysis, see extract_lesion_overall_openms.py). ROIs:

  * lesion     = the FLAIR dseg mask (co-registered → applies to every source contrast).
  * foreground = intensity > 10% of the volume's 99th percentile. NOT the whole image (unlike
                 the coverage histogram) — census on a mostly-background image is confounded
                 by background-background correlation — and NOT the computed brainmask (not
                 an annotation). A plain intensity threshold is a modest stand-in for "the
                 brain", not a ROI open-ms itself provides; flag this when reporting.

The ONLY change from on-harmony's compute_texture_metrics.py is the ROI mechanism (named
boolean masks {lesion, foreground} instead of `range(1, 32)` anatomical label ids) and the
source/lesion discovery (open-ms BIDS FLAIR/T1w scans + derivatives/manual_masks dseg, mirrors
generate_openms_volumes.py's list_sources()). rank_transform / abscorr / nmi_1d / erode and the
--sanity self-test are unchanged.

Run per source contrast (mirrors run_coverage_openms.sh): --source FLAIR|T1w, --set-label
flair|t1w — reuses aggregate_texture_metrics.py / plot_texture_metrics.py UNCHANGED (they are
already dataset-agnostic; "set" here holds the source contrast, not blur/noblur).

Usage
-----
  python compute_texture_metrics_openms.py --sanity                      # metric self-test
  python compute_texture_metrics_openms.py --device cuda --rank 0 --world-size 4 \
      --source FLAIR --set-label flair --output-csv .../metrics_flair_rank0.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

THIS_DIR      = Path(__file__).resolve().parent
ANALYSIS_ROOT = THIS_DIR.parent                                       # texture_analysis_lvl_1/
# parents[0]=lvl1, [1]=7_analysis, [2]=open-ms, [3]=datasets, [4]=repo root.
PROJECT_ROOT  = THIS_DIR.parents[4]

BIDS       = PROJECT_ROOT / "datasets/open-ms/1_BIDS_open-ms/open-ms-brain"
LESION_DIR = BIDS / "derivatives" / "manual_masks"
DEFAULT_GENERATED = PROJECT_ROOT / "datasets/open-ms/7_analysis_open-ms/data/generated"
DEFAULT_OUT       = ANALYSIS_ROOT / "outputs" / "data" / "texture_metrics.csv"

CONTRASTS         = ("FLAIR", "T1w")
GENERATED_METHODS = ["palette", "synthseg_em", "synthseg_noem", "auglab_default"]
CONTROL_METHODS   = ["gamma", "histeq"]          # synthesised inline from source
GAMMA_VALUES      = [0.5, 1.5, 2.5]
ROI_NAMES         = ["lesion", "foreground"]
FOREGROUND_FRAC   = 0.10       # foreground = intensity > FOREGROUND_FRAC * p99(volume)

MIN_VOX    = 50        # ROI smaller than this (after erosion) → skip; report n regardless
EPS        = 1e-7
RANK_RADII = [1, 2]    # census window radii reported (robustness grid)
LOCAL_BLOCK    = 8     # census_local8: block edge for the cancellation-robust local census
MIN_BLOCK_VOX  = 64    # a block needs this many in-ROI voxels to contribute


# ─────────────────────────── metric core (torch) — UNCHANGED from on-harmony ────────────────
def rank_transform(x: torch.Tensor, r: int) -> torch.Tensor:
    """Census/rank transform: fraction of (2r+1)^3−1 neighbours that x exceeds. x,→ (D,H,W).
    Strict `>` (ties not counted) so flat regions → near-constant field → ~0 correlation."""
    out = torch.zeros_like(x)
    cnt = 0
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dz == dy == dx == 0:
                    continue
                out += (x > torch.roll(x, (dz, dy, dx), (0, 1, 2))).float()
                cnt += 1
    return out / cnt


def abscorr(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> float:
    """|Pearson correlation| of a,b over boolean mask m. |·| → inversion-invariant."""
    av, bv = a[m], b[m]
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = av.norm() * bv.norm()
    if denom < EPS:
        return float("nan")
    return abs(float((av * bv).sum() / denom))


def local_abscorr(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor,
                  block: int = LOCAL_BLOCK, min_vox: int = MIN_BLOCK_VOX) -> float:
    """Cancellation-robust LOCAL census: tile (D,H,W) into non-overlapping `block`³ cubes, take
    |corr(a,b)| within each cube's masked voxels, return the voxel-count-weighted mean over cubes
    with ≥ min_vox in-ROI voxels. Whole-ROI abscorr sums signed products across the ENTIRE ROI
    before |·|, so PALETTE's signed per-region α (fromSeg.py:444) makes +corr and −corr regions
    cancel — this restricts the correlation to local cubes (≈ single region) so it doesn't. Fully
    vectorised (reshape + masked moments per cube) — no Python loop over cubes. Returns NaN if no
    cube qualifies."""
    D, H, W = a.shape
    pD, pH, pW = (-D) % block, (-H) % block, (-W) % block
    if pD or pH or pW:
        pad = (0, pW, 0, pH, 0, pD)                       # F.pad fills last dim first
        a = F.pad(a[None, None], pad)[0, 0]
        b = F.pad(b[None, None], pad)[0, 0]
        m = F.pad(m[None, None].float(), pad)[0, 0] > 0.5
    Dp, Hp, Wp = a.shape
    nD, nH, nW = Dp // block, Hp // block, Wp // block

    def cubes(t):                                          # (nCubes, block³)
        return (t.reshape(nD, block, nH, block, nW, block)
                 .permute(0, 2, 4, 1, 3, 5).reshape(nD * nH * nW, block ** 3))

    ca, cb, cm = cubes(a), cubes(b), cubes(m.float())
    n = cm.sum(1)
    keep = n >= min_vox
    if not bool(keep.any()):
        return float("nan")
    ca, cb, cm, n = ca[keep], cb[keep], cm[keep], n[keep]
    ma, mb = (ca * cm).sum(1) / n, (cb * cm).sum(1) / n    # masked per-cube means
    cov = (ca * cb * cm).sum(1) / n - ma * mb
    va  = (ca * ca * cm).sum(1) / n - ma * ma
    vb  = (cb * cb * cm).sum(1) / n - mb * mb
    denom = (va * vb).clamp_min(0).sqrt()
    good = denom > EPS
    if not bool(good.any()):
        return float("nan")
    corr = (cov[good] / denom[good]).abs()
    w = n[good]
    return float((corr * w).sum() / w.sum())


def nmi_1d(xv: torch.Tensor, yv: torch.Tensor, bins: int) -> float:
    """Studholme NMI = (H(X)+H(Y))/H(X,Y) on ROI voxel values xv, yv (1-D)."""
    def quant(v):
        vmin, vmax = v.min(), v.max()
        if (vmax - vmin) < EPS:
            return None
        return ((v - vmin) / (vmax - vmin) * (bins - 1)).round().long().clamp(0, bins - 1)

    xi, yi = quant(xv), quant(yv)
    if xi is None or yi is None:
        return float("nan")
    joint = torch.bincount(xi * bins + yi, minlength=bins * bins).float().reshape(bins, bins)
    joint /= joint.sum()
    px, py = joint.sum(1), joint.sum(0)
    hx  = -(px[px > 0] * px[px > 0].log()).sum()
    hy  = -(py[py > 0] * py[py > 0].log()).sum()
    hxy = -(joint[joint > 0] * joint[joint > 0].log()).sum()
    if hxy < EPS:
        return float("nan")
    return float(((hx + hy) / hxy).item())


def erode(mask: torch.Tensor, iters: int) -> torch.Tensor:
    """Binary erosion via min-pool. mask: (D,H,W) bool → bool."""
    if iters <= 0:
        return mask
    m = mask.float()[None, None]
    for _ in range(iters):
        m = -F.max_pool3d(-m, kernel_size=3, stride=1, padding=1)
    return m[0, 0] > 0.5


# ─────────────────────────── per-volume computation (ROI mechanism changed) ─────────────────
def foreground_mask(source: torch.Tensor) -> torch.Tensor:
    """Simple intensity threshold — NOT the computed brainmask (not an open-ms annotation)."""
    p99 = torch.quantile(source.flatten(), 0.99)
    return source > (FOREGROUND_FRAC * p99)


def metrics_for_pair(source, synth, masks: dict, bins, erode_iters, src_ranks=None):
    """source/synth: (D,H,W) on device. masks: {roi_name: bool (D,H,W)}. src_ranks: cached
    {r: rank_transform(source,r)} across a source's variants (computed here if None)."""
    if src_ranks is None:
        src_ranks = {r: rank_transform(source, r) for r in RANK_RADII}
    syn_ranks = {r: rank_transform(synth, r) for r in RANK_RADII}
    rows = []
    for roi_name, base_mask in masks.items():
        m = erode(base_mask, erode_iters)
        n = int(m.sum().item())
        if n < MIN_VOX:
            continue
        # NMI intentionally dropped for open-ms: it is spatially blind (operates on the 1-D set
        # of ROI intensity values, ignoring their arrangement) so it is not a texture metric —
        # kept only in the shared on-harmony pipeline. Texture here = census_r1/r2 (whole-ROI
        # ordinal) + census_local{LOCAL_BLOCK} (cancellation-robust local; see local_abscorr).
        row = dict(roi_id=roi_name, n_vox=n)
        for r in RANK_RADII:
            row[f"census_r{r}"] = abscorr(src_ranks[r], syn_ranks[r], m)
        row[f"census_local{LOCAL_BLOCK}"] = local_abscorr(src_ranks[1], syn_ranks[1], m)
        rows.append(row)
    return rows


# ─────────────────────────── I/O helpers ────────────────────────────────────
def _load(path: Path, device) -> torch.Tensor:
    import nibabel as nib
    arr = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[..., 0]
    return torch.from_numpy(arr).to(device)


def _load_lesion(path: Path, ref_shape, device) -> torch.Tensor:
    """Lesion dseg, resampled onto the source grid if needed (mirrors generate_openms_volumes's
    _mask_to; sources here are always the co-registered scan the mask was built for, but be
    defensive about shape drift)."""
    import nibabel as nib
    nii = nib.load(str(path))
    arr = np.round(nii.get_fdata()).astype(np.int32)
    if arr.shape != tuple(ref_shape):
        from nibabel.processing import resample_from_to
        ref = nib.Nifti1Image(np.zeros(ref_shape, dtype=np.float32), nii.affine)
        arr = np.round(resample_from_to(nii, ref, order=0).get_fdata()).astype(np.int32)
    return torch.from_numpy(arr > 0).to(device)


def _foreground_norm01(x, fg):
    vals = x[fg]
    lo, hi = torch.quantile(vals, 0.01), torch.quantile(vals, 0.99)
    return ((x - lo) / (hi - lo + EPS)).clamp(0, 1)


def make_control(source, fg_mask, kind, param):
    """Inline positive control synthesised from the source (monotone remap → census→1)."""
    s01 = _foreground_norm01(source, fg_mask)
    if kind == "gamma":
        return s01.clamp_min(0) ** float(param)
    if kind == "histeq":
        from skimage.exposure import equalize_hist
        out = equalize_hist(s01.cpu().numpy(), mask=fg_mask.cpu().numpy())
        return torch.from_numpy(out.astype(np.float32)).to(source.device)
    raise ValueError(kind)


# ─────────────────────────── task enumeration ───────────────────────────────
def list_source_keys(contrast: str):
    """{key: (image_path, lesion_path)} for open-ms scans with a lesion mask, one contrast.
    Mirrors generate_openms_volumes.py's list_sources(). key = sub-patientNN_<CONTRAST>."""
    out = {}
    for img in sorted(BIDS.glob(f"sub-*/anat/*_{contrast}.nii.gz")):
        sub = img.name.replace(f"_{contrast}.nii.gz", "")
        les = LESION_DIR / sub / "anat" / f"{sub}_FLAIR_dseg.nii.gz"
        if not les.exists():
            continue
        key = img.name.replace(".nii.gz", "")   # sub-patientNN_<CONTRAST>
        out[key] = (img, les)
    return out


def build_tasks(generated_root: Path, source_keys: dict, methods):
    """Each task = (method, key, variant, synth_path|None)."""
    tasks = []
    for method in methods:
        if method in CONTROL_METHODS:
            for key in source_keys:
                for v in (GAMMA_VALUES if method == "gamma" else [0]):
                    tasks.append((method, key, v, None))
            continue
        mdir = generated_root / method
        for key in source_keys:
            for synth in sorted((mdir / key).glob(f"{key}_run-*.nii.gz")):
                variant = int(synth.stem.split("run-")[-1].split(".")[0])
                tasks.append((method, key, variant, synth))
    return tasks


# ─────────────────────────── sanity self-test ───────────────────────────────
def run_sanity(device, bins, erode_iters) -> int:
    torch.manual_seed(0)
    D = 44

    def boxf(x, w):
        return F.avg_pool3d(x[None, None], w, 1, w // 2, count_include_pad=False)[0, 0]

    source = boxf(torch.randn(D, D, D, device=device), 5)
    source = (source - source.min()) / (source.max() - source.min() + EPS)
    lesion_mask = torch.zeros(D, D, D, dtype=torch.bool, device=device)
    lesion_mask[15:25, 15:25, 15:25] = True                       # small "lesion" ROI
    fg_mask = torch.zeros(D, D, D, dtype=torch.bool, device=device)
    fg_mask[5:39, 5:39, 5:39] = True                               # large "foreground" ROI
    masks = {"lesion": lesion_mask, "foreground": fg_mask}

    def palette_like(x, nb=6):
        centers = torch.rand(nb, 3, device=device) * D
        coords  = torch.stack(torch.meshgrid(*[torch.arange(D, device=device)] * 3, indexing="ij"), -1).float()
        lab = ((coords[..., None, :] - centers) ** 2).sum(-1).argmin(-1)
        out = x.clone()
        for b in range(nb):
            mb = lab == b
            if mb.sum() < 10:
                continue
            mu = torch.rand(1, device=device).item()
            alpha = (torch.rand(1, device=device).item() * 1.5 + 0.5) * (1 if torch.rand(1) > .5 else -1)
            out[mb] = mu + alpha * (x[mb] - x[mb].mean())
        return out

    # palette-like is high-variance on a phantom (few random blocks) — average several draws so
    # the sanity tests the METRIC (piecewise-monotone remap separates from noise), not a fragile
    # single-draw absolute. Real PALETTE numbers come from real volumes, not this.
    palette_draws = [palette_like(source) for _ in range(4)]
    cases = {
        "identity":     source.clone(),
        "gamma":        source.clamp_min(0) ** 2.0,
        "inverted":     1.0 - source,
        "noise":        torch.rand(D, D, D, device=device),
    }
    lkey = f"census_local{LOCAL_BLOCK}"
    print(f"{'case':16s} {'census_r1':>10s} {'census_r2':>10s} {lkey:>13s}")
    res = {}
    for name, synth in cases.items():
        rows = metrics_for_pair(source, synth, masks, bins, erode_iters)
        res[name] = (float(np.nanmean([r["census_r1"] for r in rows])),
                     float(np.nanmean([r["census_r2"] for r in rows])),
                     float(np.nanmean([r[lkey] for r in rows])))
        print(f"{name:16s} {res[name][0]:10.3f} {res[name][1]:10.3f} {res[name][2]:13.3f}")
    pal = float(np.mean([np.nanmean([r["census_r1"] for r in
                 metrics_for_pair(source, d, masks, bins, erode_iters)]) for d in palette_draws]))
    print(f"{'palette-like(avg4)':16s} {pal:10.3f}")

    checks = [
        ("identity census≈1",         res["identity"][0] > 0.95),
        ("gamma census≈1 (monotone)", res["gamma"][0]    > 0.95),
        ("inverted census≈1 (|·|)",   res["inverted"][0] > 0.95),
        ("noise census≈0 floor",      res["noise"][0]    < 0.15),
        ("palette-like ≫ noise floor", pal > res["noise"][0] + 0.15),
        (f"identity {lkey}≈1",         res["identity"][2] > 0.95),
        (f"noise {lkey}≈0 floor",      res["noise"][2]    < 0.20),
    ]
    print()
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}"); ok &= passed
    print(f"\nSANITY {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


# ─────────────────────────── main ───────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED)
    p.add_argument("--source", type=str, choices=list(CONTRASTS), required=False)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUT)
    p.add_argument("--methods", type=str, default=",".join(GENERATED_METHODS))
    p.add_argument("--set-label", type=str, default="flair", help="tag rows: flair / t1w / ref")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mi-bins", type=int, default=64)
    # lesions are small/sparse — erosion (min-pool over a 3^3 window) can erase them entirely,
    # unlike on-harmony's large anatomical ROIs. Default 0 (no erosion) for both ROIs.
    p.add_argument("--erode-iters", type=int, default=0)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sanity", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    if args.sanity:
        sys.exit(run_sanity(device, args.mi_bins, args.erode_iters))

    if not args.source:
        sys.exit("--source FLAIR|T1w is required (unless --sanity)")

    import pandas as pd
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    source_keys = list_source_keys(args.source)
    if not source_keys:
        log.error("No source scans with lesion mask for contrast=%s", args.source); sys.exit(1)
    log.info("set=%s  source=%s  methods=%s  sources=%d",
              args.set_label, args.source, methods, len(source_keys))

    tasks = build_tasks(args.generated_root, source_keys, methods)
    tasks = tasks[args.rank::args.world_size]
    if args.limit:
        tasks = tasks[:args.limit]
    log.info("Rank %d/%d: %d tasks", args.rank, args.world_size, len(tasks))

    source_cache, mask_cache, rank_cache = {}, {}, {}
    out_rows = []
    n_skipped_shape = 0
    for i, (method, key, variant, synth_path) in enumerate(tasks):
        try:
            if key not in source_cache:
                img_path, les_path = source_keys[key]
                src = _load(img_path, device)
                lesion = _load_lesion(les_path, src.shape, device)
                fg = foreground_mask(src)
                source_cache[key] = src
                mask_cache[key] = {"lesion": lesion, "foreground": fg}
                rank_cache[key] = {r: rank_transform(src, r) for r in RANK_RADII}
            source, masks = source_cache[key], mask_cache[key]

            if method in CONTROL_METHODS:
                synth = make_control(source, masks["foreground"], method, variant)
            else:
                synth = _load(synth_path, device)
                if synth.shape != source.shape:
                    log.warning("SHAPE MISMATCH %s %s: %s vs %s — skip",
                                method, key, tuple(synth.shape), tuple(source.shape))
                    n_skipped_shape += 1
                    continue

            for r in metrics_for_pair(source, synth, masks, args.mi_bins,
                                      args.erode_iters, src_ranks=rank_cache[key]):
                # session="single" (not ""): on-harmony's aggregate/plot scripts groupby
                # ["set","method","subject","session","roi_id"], and pandas' groupby drops
                # NaN groups by default — an empty CSV field round-trips as NaN and would
                # silently zero every row. open-ms has no sessions, so use a non-empty constant.
                out_rows.append(dict(set=args.set_label, method=method, subject=key.split("_")[0],
                                     session="single", variant=variant, **r))
        except Exception as e:                                        # noqa: BLE001
            log.warning("FAILED %s/%s/%s: %s", method, key, variant, e)
        if (i + 1) % 200 == 0:
            log.info("  %d/%d done", i + 1, len(tasks))

    if n_skipped_shape:
        log.warning("Skipped %d volumes for shape mismatch", n_skipped_shape)

    df = pd.DataFrame(out_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    log.info("Wrote %d rows → %s", len(df), args.output_csv)
    if len(df):
        log.info("n per (method, roi_id):\n%s",
                  df.groupby(["method", "roi_id"])["n_vox"].count().to_string())


if __name__ == "__main__":
    main()
