#!/usr/bin/env python
"""
H7 — border-transition profiles (Paul's hypothesis, 2026-09-24).

Hypothesis H7: the border transition matters. Noise-fill (K-means+Voronoi partition, each
region filled with a constant random mean + iid noise, borders always SHARP, optionally
blurred sigma<=0.8vox) trains a model that expects sharp region borders. Real-fill (same
partition, but each region filled with the real image under a random per-region affine map,
scale 0.5-2, random sign) trains a model that expects the TRAINING contrast's own real border
profile. This script computes, per patient x contrast x region, the intensity profile across
the region boundary; the companion `boundary_vs_fill_swap_summary.py` turns that into the
transition-width (W) and step-strength (STEP) statistics and tests the pre-registered
predictions against real-fill/noise-fill Dice deltas (P1/P2/P3, see that script's docstring —
written and committed to disk BEFORE any of this script's shard output was inspected).

Boundary/bin convention (resolved here, before looking at outcomes, because the spec's
"distance to the GT boundary, bins of 1 voxel" is ambiguous at d=0):
  - Inside distance: scipy.ndimage.distance_transform_edt(region_mask) gives, for every voxel
    IN the region, its integer voxel-distance to the nearest non-region voxel (1 for a
    boundary-adjacent region voxel, 2 for the next ring in, ...). Bin = -that distance.
  - Outside distance: distance_transform_edt(~region_mask) gives, for every voxel NOT in the
    region, its distance to the nearest region voxel. Bin = +that distance, but the voxel must
    ALSO be in the "healthy" mask (fg & label==0) — this is what "excluding other tumor labels"
    means: the surrounding tissue must be normal brain, not another tumor sub-region.
  - There is deliberately NO d=0 bin (a voxel is either in the region or not; the EDT convention
    above never produces d=0 for either side) — bins are the integers {-8,...,-1,1,...,+8}.
    This avoids the empty/ambiguous zero-bin problem a half-voxel-shift convention would create.
  - "brain" for z-scoring = healthy | whole_tumor (from region_masks in compute_cross_contrast_ngf).
    z(v) = (img(v) - mean_brain) / std_brain, per patient x contrast.

Each shard stores per-bin SUFFICIENT STATISTICS ONLY (n, sum_z, sum_z^2), not the profile
itself — this lets the summary script compute every windowing variant (primary ±6, robustness
±4/±8, alternate I_in/I_out windows) and the pooled-variance STEP statistic without re-touching
imaging data. Resumable/parallel like compute_region_surround_texture.py: --rank/--world-size,
one CSV per rank, appended per patient, skips patients already present on rerun.

The core W/STEP math (crossing_width, step_stat, profile_distance) lives HERE, is unit-tested
by --sanity below, and is imported unchanged by boundary_vs_fill_swap_summary.py — so the
summary never reimplements or subtly diverges from what --sanity verified.

Usage (Slurm CPU job only, see run_boundary_profiles.sh):
  .venv/bin/python compute_boundary_profiles.py --sanity
  .venv/bin/python compute_boundary_profiles.py --rank R --world-size W [--limit-patients N]
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
MAX_BIN = 8                       # shard stores -8..-1, 1..8 so the summary can trim to +-4/6/8
BINS = tuple(range(-MAX_BIN, 0)) + tuple(range(1, MAX_BIN + 1))
MIN_BIN_VOX = 20                  # pre-registered: a bin with fewer voxels is NaN, not noisy
SHARD_COLS = ["patient", "contrast", "region", "bin", "n", "sum_z", "sum_z2"]


# ───────────────────────── core math (imported by the summary script; tested by --sanity) ──────
def bin_stats_to_mean_var(n: np.ndarray, sum_z: np.ndarray, sum_z2: np.ndarray):
    """Per-bin mean and (population) variance from pooled sufficient stats. NaN where n==0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum_z / n
        var = sum_z2 / n - mean ** 2
    mean = np.where(n >= MIN_BIN_VOX, mean, np.nan)
    var = np.where(n >= MIN_BIN_VOX, var, np.nan)
    return mean, var


def pooled_mean_std(n: np.ndarray, sum_z: np.ndarray, sum_z2: np.ndarray, sel: np.ndarray):
    """Pools raw voxel sufficient stats across the bins where sel is True into one sample mean
    and (sample, ddof=1) std. Returns (nan, nan) if fewer than 2 pooled voxels."""
    N = int(n[sel].sum())
    if N < 2:
        return float("nan"), float("nan")
    S1, S2 = float(sum_z[sel].sum()), float(sum_z2[sel].sum())
    mean = S1 / N
    var = max((S2 - N * mean ** 2) / (N - 1), 0.0)
    return mean, float(np.sqrt(var))


def normalize_profile(centers: np.ndarray, mean_z: np.ndarray, in_window, out_window):
    """p(d) = (I(d)-I_out)/(I_in-I_out). in_window/out_window are (lo, hi) inclusive bin bounds.
    Returns (p, i_in, i_out) — p is NaN wherever mean_z is NaN or normalization degenerates."""
    in_sel = (centers >= in_window[0]) & (centers <= in_window[1])
    out_sel = (centers >= out_window[0]) & (centers <= out_window[1])
    i_in = np.nanmean(mean_z[in_sel]) if in_sel.any() else float("nan")
    i_out = np.nanmean(mean_z[out_sel]) if out_sel.any() else float("nan")
    denom = i_in - i_out
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        return np.full_like(mean_z, np.nan, dtype=float), i_in, i_out
    p = (mean_z - i_out) / denom
    return p, i_in, i_out


def crossing_width(centers: np.ndarray, p: np.ndarray, hi=0.8, lo=0.2):
    """W = distance between the first (most-negative-d) downward crossing of `hi` and the next
    downward crossing of `lo` scanning inside(-)->outside(+), linear interpolation between bin
    centers. NaN if p has gaps (NaN) inside the scanned span, or either crossing is missing."""
    order = np.argsort(centers)
    c, v = centers[order], p[order]
    if np.isnan(v).any():
        # only fail if a NaN sits strictly between the first >=hi and last <=lo sample
        finite = np.isfinite(v)
        if not finite.any():
            return float("nan")
        first_fin, last_fin = np.argmax(finite), len(finite) - 1 - np.argmax(finite[::-1])
        if np.isnan(v[first_fin:last_fin + 1]).any():
            return float("nan")

    def find_crossing(start_idx, thresh):
        for i in range(start_idx, len(c) - 1):
            v0, v1 = v[i], v[i + 1]
            if np.isnan(v0) or np.isnan(v1):
                continue
            if v0 >= thresh >= v1 and v0 != v1:
                frac = (v0 - thresh) / (v0 - v1)
                return c[i] + frac * (c[i + 1] - c[i]), i
        return None, None

    d_hi, i_hi = find_crossing(0, hi)
    if d_hi is None:
        return float("nan")
    d_lo, _ = find_crossing(i_hi, lo)
    if d_lo is None:
        return float("nan")
    w = d_lo - d_hi
    return w if w >= 0 else float("nan")


def step_stat(i_in: float, i_out: float, pooled_std: float):
    if not np.isfinite(i_in) or not np.isfinite(i_out) or not np.isfinite(pooled_std) or pooled_std <= 1e-9:
        return float("nan")
    return abs(i_in - i_out) / pooled_std


def profile_distance(centers: np.ndarray, p_a: np.ndarray, p_b: np.ndarray):
    """RMS(p_a - p_b) over bins where both are finite. NaN if fewer than 4 shared bins."""
    both = np.isfinite(p_a) & np.isfinite(p_b)
    if int(both.sum()) < 4:
        return float("nan")
    return float(np.sqrt(np.mean((p_a[both] - p_b[both]) ** 2)))


# ───────────────────────── sanity ─────────────────────────
def run_sanity() -> int:
    checks = []
    shape = (60, 60, 60)
    region = np.zeros(shape, bool)
    region[15:45, 15:45, 15:45] = True
    healthy = ~region  # whole volume outside the cube counts as "healthy" for this synthetic test
    dt_in = distance_transform_edt(region)
    dt_out = distance_transform_edt(~region)
    checks.append(("no zero-distance voxel on either side (bin convention has no d=0)",
                   (dt_in[region] > 0).all() and (dt_out[~region] > 0).all()))
    checks.append(("boundary-adjacent inside voxel has dt_in==1",
                   dt_in[region].min() == 1))
    checks.append(("no empty bin within +-6 for a 30-voxel-wide cube (coverage sanity)",
                   all(int((region & (dt_in == k)).sum()) >= MIN_BIN_VOX for k in range(1, 7))
                   and all(int((healthy & (dt_out == k)).sum()) >= MIN_BIN_VOX for k in range(1, 7))))

    def bin_it(img):
        centers, means, varz = [], [], []
        for k in range(1, MAX_BIN + 1):
            for sign, mask in ((-1, region & (dt_in == k)), (1, healthy & (dt_out == k))):
                n = int(mask.sum())
                z = img[mask]
                centers.append(sign * k)
                means.append(z.mean() if n >= MIN_BIN_VOX else np.nan)
                varz.append(z.var() if n >= MIN_BIN_VOX else np.nan)
        order = np.argsort(centers)
        return np.array(centers)[order], np.array(means)[order], np.array(varz)[order]

    # sharp step: inside=1, outside=0 (+tiny noise so variance isn't exactly 0)
    rng = np.random.default_rng(0)
    step_img = np.where(region, 1.0, 0.0) + rng.normal(0, 0.01, shape)
    centers, mean_step, _ = bin_it(step_img)
    p_step, i_in, i_out = normalize_profile(centers, mean_step, (-6, -4), (4, 6))
    w_step = crossing_width(centers, p_step)
    checks.append(("sharp step: W is small (<1.5 vox)", np.isfinite(w_step) and w_step < 1.5))

    # blurred step: sigma=3 Gaussian blur of the same step -> W ~= 1.68*sigma (Gaussian 0.2-0.8
    # quantile spread = 2*0.8416*sigma = 1.683*sigma for a true infinite Gaussian edge)
    blur_img = gaussian_filter(np.where(region, 1.0, 0.0), sigma=3.0) + rng.normal(0, 0.01, shape)
    centers_b, mean_blur, _ = bin_it(blur_img)
    p_blur, _, _ = normalize_profile(centers_b, mean_blur, (-6, -4), (4, 6))
    w_blur = crossing_width(centers_b, p_blur)
    expected = 1.683 * 3.0
    checks.append((f"blurred (sigma=3) step: W within 40% of expected {expected:.2f}",
                   np.isfinite(w_blur) and abs(w_blur - expected) < 0.4 * expected))
    checks.append(("blur widens the transition vs the sharp step", w_blur > w_step))

    # invariance to x -> -2x+7 (what real-fill's random per-region affine, incl. sign, does)
    aff_img = -2.0 * step_img + 7.0
    centers_a, mean_aff, _ = bin_it(aff_img)
    p_aff, _, _ = normalize_profile(centers_a, mean_aff, (-6, -4), (4, 6))
    w_aff = crossing_width(centers_a, p_aff)
    checks.append(("W invariant to x -> -2x+7 (|diff| < 1e-6)",
                   np.isfinite(w_aff) and abs(w_aff - w_step) < 1e-6))
    p_diff = np.nanmax(np.abs(p_aff - p_step))
    checks.append(("p(d) itself invariant to x -> -2x+7 (max|diff| < 1e-6)", p_diff < 1e-6))

    # profile_distance: identical profiles -> 0; sharp vs blurred -> clearly > 0
    checks.append(("profile_distance(step, step) == 0", profile_distance(centers, p_step, p_step) == 0.0))
    pd_val = profile_distance(centers, p_step, p_blur)
    checks.append(("profile_distance(step, blurred) > 0.1", np.isfinite(pd_val) and pd_val > 0.1))

    # degenerate normalization (I_in == I_out exactly, no noise at all) -> NaN profile, not garbage
    flat_img = np.zeros(shape)
    centers_f, mean_flat, _ = bin_it(flat_img)
    p_flat, _, _ = normalize_profile(centers_f, mean_flat, (-6, -4), (4, 6))
    checks.append(("degenerate (flat) profile is all-NaN, not exploded", np.isnan(p_flat).all()))

    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= bool(passed)
    print(f"\nSANITY {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


# ───────────────────────── per-patient shard computation ─────────────────────────
def patient_rows(pid: str):
    vols_t, label_t = load_patient(pid, "cpu")
    masks = {k: m.numpy() for k, m in region_masks(label_t, vols_t["t1n"]).items()}
    brain = masks["healthy"] | masks["whole_tumor"]
    healthy = masks["healthy"]
    rows = []
    for region in REGIONS:
        region_mask = masks[region]
        if int(region_mask.sum()) == 0:
            continue
        dt_in = distance_transform_edt(region_mask)
        dt_out = distance_transform_edt(~region_mask)
        bin_masks = {}
        for k in range(1, MAX_BIN + 1):
            bin_masks[-k] = region_mask & (dt_in == k)
            bin_masks[k] = healthy & (dt_out == k)
        for c in CONTRAST_SUFFIX:
            img = vols_t[c].numpy().astype(np.float64)
            b = img[brain]
            mu, sigma = float(b.mean()), float(b.std())
            if sigma <= 1e-9:
                continue
            z = (img - mu) / sigma
            for k in BINS:
                m = bin_masks[k]
                n = int(m.sum())
                if n == 0:
                    continue
                zz = z[m]
                rows.append([pid, c, region, k, n, float(zz.sum()), float((zz ** 2).sum())])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit-patients", type=int, default=None)
    p.add_argument("--shard-prefix", default="boundary_profile_shard")
    p.add_argument("--output-dir", type=Path, default=DATA_DIR,
                   help="where shard CSVs land — use a separate dir for --limit-patients smoke runs")
    p.add_argument("--sanity", action="store_true")
    args = p.parse_args()
    if args.sanity:
        sys.exit(run_sanity())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patients = sorted(pd.read_csv(DATA_DIR / "patient_region_deltas.csv")["case"].unique())
    if args.limit_patients:
        patients = patients[: args.limit_patients]
    mine = patients[args.rank::args.world_size]
    shard = args.output_dir / f"{args.shard_prefix}{args.rank}.csv"
    done = set(pd.read_csv(shard)["patient"].unique()) if shard.exists() else set()
    todo = [q for q in mine if q not in done]
    log.info("rank %d/%d: %d assigned, %d done, %d to do", args.rank, args.world_size, len(mine), len(done), len(todo))
    header = not shard.exists()
    n_ok, n_fail = 0, 0
    for i, pid in enumerate(todo):
        try:
            rows = patient_rows(pid)
        except (FileNotFoundError, ValueError) as e:
            n_fail += 1
            log.warning("SKIP %s: %s", pid, e)
            continue
        pd.DataFrame(rows, columns=SHARD_COLS).to_csv(shard, mode="a", header=header, index=False)
        header = False
        n_ok += 1
        if (i + 1) % 5 == 0:
            log.info("  rank %d: %d/%d (%d ok, %d failed)", args.rank, i + 1, len(todo), n_ok, n_fail)
    log.info("rank %d finished: %d ok, %d failed this run", args.rank, n_ok, n_fail)


if __name__ == "__main__":
    main()
