#!/usr/bin/env python
"""
Region-vs-surroundings texture fingerprints (Paul's hypothesis, 2026-09-24).

Hypothesis: whether real-fill (keeps each sub-region's real internal texture under a random
affine remap incl. sign flip) helps or hurts on an eval contrast depends on how the region's
texture relates to its SURROUNDINGS, not only to the same region in the training contrast:
  (a) if the region's texture resembles its surroundings in the eval image, texture can't help;
  (b) if the texture learned for the region in the TRAINING contrast resembles the eval image's
      surroundings more than the eval region, the learned cue points to the wrong place -> failure.
NGF / correlation ratio can't express this: they are voxel-paired (same location in two
images), while region and surroundings are different voxels. So each region gets a texture
FINGERPRINT that can be compared between any two regions.

Fingerprint = normalized spatial autocorrelation: Pearson correlation between intensities of
voxel pairs (v, v + k*e_axis), both inside the mask, for k = 1..4 voxels on each of the 3 array
axes (12 numbers). Pearson is exactly invariant to x -> a*x + b for any a != 0 (sign flips
included), i.e. to what real-fill randomizes; no gray-level discretization; captures fine vs
coarse texture and direction (incl. the thick-slice S-I blur of t2w/t2f).

Masks per patient (from region_masks; brain = healthy | whole_tumor):
  part="region" : the region eroded by 1 voxel (EDT > 1), to avoid partial-volume borders.
  part="ring"   : voxels OUTSIDE the region at Euclidean distance (GAP, d_out] from it, inside
                  brain — all labels (what the model must separate the region from).
  part="ring_healthy": ring restricted to healthy tissue.
  d_out in {3, 5, 8} (primary 5), GAP = 1 (a 1-voxel moat against partial volume).
Variants: "raw" and "highpass" = x - mask-normalized Gaussian(sigma=2) within brain (removes
slow shading / bias field so it can't masquerade as texture). A mask with < MIN_VOX voxels or a
lag with < MIN_PAIRS voxel pairs gives NaN.

Parallel/resumable like the other scripts (--rank/--world-size, per-rank shard appended per
patient). Patients = the 70 eval cases in outputs/data/patient_region_deltas.csv.

  python compute_region_surround_texture.py --sanity
  python compute_region_surround_texture.py --rank R --world-size W
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from compute_cross_contrast_ngf import CONTRAST_SUFFIX, load_patient, region_masks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = THIS_DIR.parent / "outputs" / "data"
REGIONS = ("SNFH", "RC", "ET", "NCR")
LAGS = (1, 2, 3, 4)
D_OUTS = (3, 5, 8)
GAP = 1.0
HP_SIGMA = 2.0
MIN_VOX = 200
MIN_PAIRS = 100
ACF_COLS = [f"acf_ax{a}_lag{k}" for a in range(3) for k in LAGS]
COLS = ["patient", "contrast", "region", "part", "d_out", "variant", "n_vox"] + ACF_COLS


def _pairs(mask: np.ndarray, axis: int, k: int):
    n = mask.shape[axis]
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    lo[axis], hi[axis] = slice(0, n - k), slice(k, n)
    both = mask[tuple(lo)] & mask[tuple(hi)]
    return tuple(lo), tuple(hi), both


def acf_vector(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(len(ACF_COLS), np.nan)
    if int(mask.sum()) < MIN_VOX:
        return out
    i = 0
    for axis in range(3):
        for k in LAGS:
            lo, hi, both = _pairs(mask, axis, k)
            if int(both.sum()) >= MIN_PAIRS:
                x, y = img[lo][both], img[hi][both]
                sx, sy = x.std(), y.std()
                if sx > 0 and sy > 0:
                    out[i] = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
            i += 1
    return out


def region_parts(region: np.ndarray, brain: np.ndarray, healthy: np.ndarray):
    parts = {("region", np.nan): distance_transform_edt(region) > 1.0}
    dist_out = distance_transform_edt(~region)
    for d in D_OUTS:
        ring = (dist_out > GAP) & (dist_out <= d) & brain & ~region
        parts[("ring", d)] = ring
        parts[("ring_healthy", d)] = ring & healthy
    return parts


def highpass(img: np.ndarray, brain: np.ndarray) -> np.ndarray:
    b = brain.astype(np.float64)
    smooth = gaussian_filter(img * b, HP_SIGMA) / np.maximum(gaussian_filter(b, HP_SIGMA), 1e-6)
    return np.where(brain, img - smooth, 0.0)


def run_sanity() -> int:
    rng = np.random.default_rng(0)
    shape = (60, 60, 60)
    white = rng.normal(size=shape)
    coarse = gaussian_filter(rng.normal(size=shape), 2.0)
    aniso = gaussian_filter(rng.normal(size=shape), (0, 0, 3.0))
    cube = np.zeros(shape, bool)
    cube[10:50, 10:50, 10:50] = True
    checks = []

    a0, a1 = acf_vector(coarse, cube), acf_vector(-2.7 * coarse + 11.0, cube)
    checks.append(("invariant to a*x+b with a<0 (max|diff| < 1e-9)", np.nanmax(np.abs(a0 - a1)) < 1e-9))
    aw = acf_vector(white, cube)
    checks.append(("white noise: |acf| < 0.03 at all lags", np.nanmax(np.abs(aw)) < 0.03))
    lag1 = [a0[ax * len(LAGS)] for ax in range(3)]
    checks.append(("coarse texture: lag-1 acf > 0.8 on every axis", min(lag1) > 0.8))
    dec = all(np.all(np.diff(a0[ax * len(LAGS):(ax + 1) * len(LAGS)]) < 0) for ax in range(3))
    checks.append(("coarse texture: acf decreases with lag", dec))
    aa = acf_vector(aniso, cube)
    checks.append(("axis-2 blur: axis-2 lag-1 acf > axes 0/1 lag-1 acf",
                   aa[2 * len(LAGS)] > max(aa[0], aa[len(LAGS)]) + 0.3))

    region = np.zeros(shape, bool)
    region[25:35, 25:35, 25:35] = True
    brain = np.ones(shape, bool)
    parts = region_parts(region, brain, brain)
    ring = parts[("ring", 5)]
    d = distance_transform_edt(~region)
    checks.append(("ring disjoint from region and within (1,5] voxels",
                   not (ring & region).any() and d[ring].min() > 1 and d[ring].max() <= 5))
    checks.append(("eroded region is a strict subset", parts[("region", np.nan)].sum() < region.sum()))

    def dist(u, v):
        return float(np.linalg.norm(u - v))
    train_region = a0                                                        # coarse texture
    other_coarse = acf_vector(gaussian_filter(rng.normal(size=shape), 2.0), cube)
    other_fine = acf_vector(rng.normal(size=shape), cube)
    # good case: eval region coarse (like training), eval surround fine
    m_good = dist(train_region, other_fine) - dist(train_region, other_coarse)
    # bad case: eval region fine, eval surround coarse (looks like the learned region texture)
    m_bad = dist(train_region, other_coarse) - dist(train_region, other_fine)
    checks.append(("margin > 0 when eval region matches the training texture", m_good > 0.5))
    checks.append(("margin < 0 when eval surround matches it instead", m_bad < -0.5))

    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= bool(passed)
    print(f"\nSANITY {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def patient_rows(pid: str):
    vols_t, label_t = load_patient(pid, "cpu")
    masks = {k: m.numpy() for k, m in region_masks(label_t, vols_t["t1n"]).items()}
    brain, healthy = masks["healthy"] | masks["whole_tumor"], masks["healthy"]
    raw = {c: vols_t[c].numpy().astype(np.float64) for c in CONTRAST_SUFFIX}
    variants = {"raw": raw, "highpass": {c: highpass(v, brain) for c, v in raw.items()}}
    rows = []
    for region in REGIONS:
        if int(masks[region].sum()) == 0:
            continue
        for (part, d_out), m in region_parts(masks[region], brain, healthy).items():
            n = int(m.sum())
            for vname, imgs in variants.items():
                for c in CONTRAST_SUFFIX:
                    rows.append([pid, c, region, part, d_out, vname, n, *acf_vector(imgs[c], m)])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit-patients", type=int, default=None)
    p.add_argument("--shard-prefix", default="region_surround_texture_shard")
    p.add_argument("--sanity", action="store_true")
    args = p.parse_args()
    if args.sanity:
        sys.exit(run_sanity())

    patients = sorted(pd.read_csv(DATA_DIR / "patient_region_deltas.csv")["case"].unique())
    if args.limit_patients:
        patients = patients[: args.limit_patients]
    mine = patients[args.rank::args.world_size]
    shard = DATA_DIR / f"{args.shard_prefix}{args.rank}.csv"
    done = set(pd.read_csv(shard)["patient"].unique()) if shard.exists() else set()
    todo = [q for q in mine if q not in done]
    log.info("rank %d/%d: %d assigned, %d done, %d to do", args.rank, args.world_size, len(mine), len(done), len(todo))
    header = not shard.exists()
    for i, pid in enumerate(todo):
        try:
            rows = patient_rows(pid)
        except (FileNotFoundError, ValueError) as e:
            log.warning("SKIP %s: %s", pid, e)
            continue
        pd.DataFrame(rows, columns=COLS).to_csv(shard, mode="a", header=header, index=False)
        header = False
        if (i + 1) % 5 == 0:
            log.info("  rank %d: %d/%d", args.rank, i + 1, len(todo))
    log.info("rank %d finished", args.rank)


if __name__ == "__main__":
    main()
