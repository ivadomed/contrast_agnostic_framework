#!/usr/bin/env python
"""
H6 "step vs affine-texture" (Paul's hypothesis, 2026-09-24). BraTS2024-glioma.

The noise-fill ablation rung destroys within-region texture and keeps only each region's
mean-intensity STEP against its surroundings; the real-fill rung additionally keeps the
training contrast's within-region pattern, up to a random per-region affine map (scale,
offset, sign — exactly what real-fill applies, see palette_noisefill.py). Hypothesis: the
noise-fill model's cross-contrast Dice tracks how visible the region's mean STEP is in the
EVAL image (it never learned anything else); the real-fill model's Dice additionally tracks
whether the eval contrast's within-region pattern is an AFFINE function of the training
contrast's own pattern (real-fill's invariance class) — this can help or hurt depending on
direction, unlike the noise-fill floor.

PRE-REGISTERED PREDICTIONS (written before computing outcomes, 2026-09-24):
  P1: noise-fill Dice tracks eval step (M1 step_d, M2 step_auc, M3 edge_salience): tau > 0.
  P2: real-fill Dice tracks affine_r2 (M4, highpass) MORE than noise-fill Dice does.
  P3: gain Delta = real - noise tracks z(M4 highpass) - z(M1) within rows: tau > 0.
  P4 (edema/SNFH, t2w-trained): step on t1n > step on t1c (explains noise-fill t1n .53 vs
      t1c .31 recall reported in the error-analysis prior work), tested on M1 step_d as
      primary (one-sided Wilcoxon signed-rank over patients, alpha=0.05), M2 step_auc and
      M3 edge_salience reported alongside as the same directional claim on the other two
      "step" measures. "Holds" = direction correct (mean t1n > mean t1c) AND p < 0.05
      one-sided; "direction correct but n.s." is reported separately from "does not hold".
  Tested within each training-contrast x region row (12 rows) over its 3 eval contrasts via
  Kendall S / tau pooled with an exact one-sided permutation p (ranking_within_train.kendall_S /
  pooled_p, reused unchanged), plus Spearman across the 36 pooled cells. Robustness: ring
  widths 3 and 8 (primary 5); raw vs highpass for M4 (primary highpass, matches
  region_surround_vs_fill_swap_summary.py's own primary-config choice and rationale), plus a
  "*_core" M4 variant restricted to voxels >3 voxels inside the (non-eroded) region boundary,
  to check affine_r2 isn't just picking up a shared boundary step under the highpass kernel.
  Secondary (between-patient, exploratory power is lower): ONE-SIDED (rho > 0 predicted)
  per-patient Spearman of noise-fill Dice vs eval step_d in cells (t2w->t1n SNFH),
  (t2w->t1c SNFH), (t2f->t1n SNFH), (t1n->t2f SNFH); Holm over those 4 one-sided p-values.

Measures (per patient x contrast x region; regions SNFH, RC, ET, NCR; MIN_VOX = 200 voxels):
  M1 step_d    = |mean(R) - mean(S)| / sqrt((var(R)+var(S))/2)
                 R = region eroded by 1 voxel (EDT > 1, region_parts(("region", nan))),
                 S = ring of brain voxels at Euclidean distance (1, d_out] outside the region
                 (region_parts(("ring", d_out)), d_out in {3, 5, 8}, primary 5).
                 Affine- and sign-invariant (a std-normalized mean gap).
  M2 step_auc  = max(AUC, 1-AUC) of region vs ring intensities, AUC = Mann-Whitney
                 U/(n1*n2) (scipy.stats.mannwhitneyu; monotone- and sign-invariant, no
                 sklearn dependency needed — matches every other script in this dir).
  M3 edge_salience = mean |gradient| (np.gradient, per-contrast z-scored-within-brain image)
                 on the region boundary shell (voxels within 1 voxel of the region/non-region
                 boundary, EDT-based, inside brain) / mean |gradient| over all brain voxels.
                 Independent of d_out (stored once per (patient,contrast,region), duplicated
                 across the d_out rows purely so one join key works for all three measures).
  M4 affine_r2(a,b) = squared Pearson r between contrast a and contrast b intensities over
                 the same eroded-region voxels (co-registered — same voxel indices for every
                 contrast). Computed for every unordered contrast pair, on the raw
                 (un-z-scored) intensities AND on highpass = img - Gaussian(sigma=2)
                 within-brain, mask-normalized (compute_region_surround_texture.highpass,
                 reused unchanged). r^2 is symmetric by construction (a,b vs b,a) — expected,
                 not a bug; the summary script looks it up either way round.

Reuses (imported, not copied): compute_cross_contrast_ngf.{CONTRAST_SUFFIX, load_patient,
region_masks}; compute_region_surround_texture.{region_parts, highpass, D_OUTS}.

Parallel/resumable exactly like compute_region_surround_texture.py: --rank/--world-size
shards, one CSV pair (step + affine) per rank, appended per patient so a killed job only
loses the in-flight patient and a rerun with the same flags skips finished patients.

Usage (CPU-only Vulcan Slurm job, never the login node):
  python compute_step_affine.py --sanity
  python compute_step_affine.py --rank R --world-size W
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import mannwhitneyu, pearsonr

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from compute_cross_contrast_ngf import CONTRAST_SUFFIX, load_patient, region_masks  # noqa: E402
from compute_region_surround_texture import region_parts, highpass, D_OUTS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = THIS_DIR.parent / "outputs" / "data"
REGIONS = ("SNFH", "RC", "ET", "NCR")
MIN_VOX = 200  # same floor as compute_region_surround_texture.py
PATIENTS_CSV = DATA_DIR / "patient_region_deltas.csv"

STEP_COLS = ["patient", "contrast", "region", "d_out", "n_region", "n_ring",
             "step_d", "step_auc", "edge_salience"]
AFFINE_COLS = ["patient", "region", "contrast_a", "contrast_b", "variant", "n_vox", "affine_r2"]


# ───────────────────────── measures ─────────────────────────
def zscore_in_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.zeros_like(img)
    if int(mask.sum()) == 0:
        return z
    m, s = img[mask].mean(), img[mask].std()
    if s > 0:
        z[mask] = (img[mask] - m) / s
    return z


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gz, gy, gx = np.gradient(img)
    return np.sqrt(gz * gz + gy * gy + gx * gx)


def boundary_shell(region: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Voxels within 1 voxel of the region/non-region boundary (either side), inside brain."""
    d_in = distance_transform_edt(region)
    d_out = distance_transform_edt(~region)
    shell = ((d_in > 0) & (d_in <= 1.0)) | ((d_out > 0) & (d_out <= 1.0))
    return shell & brain


def step_d(region_vals: np.ndarray, ring_vals: np.ndarray) -> float:
    if len(region_vals) < 2 or len(ring_vals) < 2:
        return float("nan")
    vr, vs = float(region_vals.var()), float(ring_vals.var())
    denom = np.sqrt((vr + vs) / 2.0)
    if denom <= 0:
        return float("nan")
    return float(abs(region_vals.mean() - ring_vals.mean()) / denom)


def step_auc(region_vals: np.ndarray, ring_vals: np.ndarray) -> float:
    n1, n2 = len(region_vals), len(ring_vals)
    if n1 < 2 or n2 < 2:
        return float("nan")
    try:
        res = mannwhitneyu(region_vals, ring_vals, alternative="two-sided")
    except ValueError:
        return float("nan")
    auc = float(res.statistic) / (n1 * n2)
    return max(auc, 1.0 - auc)


def affine_r2(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    r, _ = pearsonr(a, b)
    return float(r * r)


# ───────────────────────── sanity ─────────────────────────
def run_sanity() -> int:
    rng = np.random.default_rng(0)
    shape = (50, 50, 50)
    brain = np.ones(shape, bool)
    region = np.zeros(shape, bool)
    region[15:35, 15:35, 15:35] = True
    d_out_map = distance_transform_edt(~region)
    ring = (d_out_map > 1.0) & (d_out_map <= 5.0)

    checks = []
    same_r = rng.normal(size=int(region.sum()))
    same_s = rng.normal(size=int(ring.sum()))
    d0, a0 = step_d(same_r, same_s), step_auc(same_r, same_s)
    checks.append(("step_d ~ 0-level for identical distributions (< 0.15)", d0 < 0.15))
    checks.append(("step_auc ~ 0.5-level for identical distributions (< 0.55)", a0 < 0.55))

    shift_r = rng.normal(loc=4.0, size=int(region.sum()))
    d1, a1 = step_d(shift_r, same_s), step_auc(shift_r, same_s)
    checks.append(("step_d large for a shifted distribution (> 1.5)", d1 > 1.5))
    checks.append(("step_auc close to 1 for a shifted distribution (> 0.95)", a1 > 0.95))

    x = rng.normal(size=3000)
    y = -3.0 * x + 5.0
    checks.append(("affine_r2 = 1 for y = -3x+5", abs(affine_r2(x, y) - 1.0) < 1e-9))
    z = rng.normal(size=3000)
    checks.append(("affine_r2 ~ 0 for independent noise (< 0.02)", affine_r2(x, z) < 0.02))

    shell = boundary_shell(region, brain)
    d_in_map = distance_transform_edt(region)
    checks.append(("boundary shell non-empty and stays within 1 voxel of the boundary",
                   int(shell.sum()) > 0 and not bool((shell & (d_in_map > 1.5) & (d_out_map > 1.5)).any())))
    checks.append(("boundary shell disjoint from the eroded region (EDT > 1)",
                   not bool((shell & (d_in_map > 1.0)).any())))

    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= bool(passed)
    print(f"\nSANITY {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


# ───────────────────────── sharded compute ─────────────────────────
def all_patients() -> list:
    if not PATIENTS_CSV.exists():
        sys.exit(f"missing {PATIENTS_CSV} (run compute_cross_contrast_ngf.py --merge first)")
    return sorted(pd.read_csv(PATIENTS_CSV)["case"].unique().tolist())


def step_shard_path(rank: int) -> Path:
    return DATA_DIR / f"step_affine_step_shard{rank}.csv"


def affine_shard_path(rank: int) -> Path:
    return DATA_DIR / f"step_affine_affine_shard{rank}.csv"


def done_patients(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        return set(pd.read_csv(csv_path)["patient"].unique())
    except Exception:  # noqa: BLE001 — corrupt/partial file from a killed job, redo it
        return set()


def compute_shard(rank: int, world_size: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    my_patients = all_patients()[rank::world_size]
    step_csv, affine_csv = step_shard_path(rank), affine_shard_path(rank)
    finished = done_patients(step_csv) & done_patients(affine_csv)
    todo = [p for p in my_patients if p not in finished]
    log.info("rank %d/%d: %d patients assigned, %d already done, %d to do",
              rank, world_size, len(my_patients), len(finished), len(todo))

    write_header_step = not step_csv.exists()
    write_header_affine = not affine_csv.exists()
    n_ok, n_fail = 0, 0
    for i, pid in enumerate(todo):
        try:
            vols, label = load_patient(pid, "cpu")
        except (FileNotFoundError, ValueError) as e:
            n_fail += 1
            log.warning("SKIP %s: %s", pid, e)
            continue

        masks = {k: v.numpy() for k, v in region_masks(label, vols["t1n"]).items()}
        brain, healthy = masks["healthy"] | masks["whole_tumor"], masks["healthy"]
        raw = {c: vols[c].numpy().astype(np.float64) for c in CONTRAST_SUFFIX}
        hp = {c: highpass(raw[c], brain) for c in CONTRAST_SUFFIX}
        gradmag = {c: gradient_magnitude(zscore_in_mask(raw[c], brain)) for c in CONTRAST_SUFFIX}
        brain_gradmean = {c: (float(gradmag[c][brain].mean()) if int(brain.sum()) > 0 else float("nan"))
                           for c in CONTRAST_SUFFIX}

        step_rows, affine_rows = [], []
        for region in REGIONS:
            rmask = masks[region]
            if int(rmask.sum()) < MIN_VOX:
                continue
            parts = region_parts(rmask, brain, healthy)
            Rmask = parts[("region", np.nan)]
            n_R = int(Rmask.sum())
            shell = boundary_shell(rmask, brain)
            n_shell = int(shell.sum())

            for c in CONTRAST_SUFFIX:
                edge_sal = (float(gradmag[c][shell].mean() / brain_gradmean[c])
                            if n_shell > 0 and brain_gradmean[c] and brain_gradmean[c] > 0 else float("nan"))
                for d in D_OUTS:
                    ring = parts[("ring", d)]
                    n_s = int(ring.sum())
                    if n_R < MIN_VOX or n_s < MIN_VOX:
                        sd = sa = float("nan")
                    else:
                        sd = step_d(raw[c][Rmask], raw[c][ring])
                        sa = step_auc(raw[c][Rmask], raw[c][ring])
                    step_rows.append([pid, c, region, d, n_R, n_s, sd, sa, edge_sal])

            if n_R >= MIN_VOX:
                for a, b in itertools.combinations(CONTRAST_SUFFIX, 2):
                    for variant, imgs in (("raw", raw), ("highpass", hp)):
                        r2 = affine_r2(imgs[a][Rmask], imgs[b][Rmask])
                        affine_rows.append([pid, region, a, b, variant, n_R, r2])

            # Robustness variant for M4, added BEFORE any outcome was looked at (advisor
            # review, 2026-09-24): a highpass sigma=2 kernel leaves boundary-step residue
            # inside the eroded region for several voxels, which could inflate affine_r2 via
            # a shared step rather than shared interior texture (r^2 is sign-invariant, so a
            # step present in both contrasts correlates). "*_core" restricts M4 to voxels at
            # least 3 voxels inside the ORIGINAL (non-eroded) region boundary — far enough
            # from the boundary that a shared step can't explain a high r^2 there.
            core_mask = distance_transform_edt(rmask) > 3.0
            n_core = int(core_mask.sum())
            if n_core >= MIN_VOX:
                for a, b in itertools.combinations(CONTRAST_SUFFIX, 2):
                    for variant, imgs in (("raw_core", raw), ("highpass_core", hp)):
                        r2 = affine_r2(imgs[a][core_mask], imgs[b][core_mask])
                        affine_rows.append([pid, region, a, b, variant, n_core, r2])

        pd.DataFrame(step_rows, columns=STEP_COLS).to_csv(
            step_csv, mode="a", header=write_header_step, index=False)
        pd.DataFrame(affine_rows, columns=AFFINE_COLS).to_csv(
            affine_csv, mode="a", header=write_header_affine, index=False)
        write_header_step = write_header_affine = False
        n_ok += 1
        del vols, label, raw, hp, gradmag
        if (i + 1) % 5 == 0:
            log.info("  rank %d: %d/%d done (%d ok, %d skipped)", rank, i + 1, len(todo), n_ok, n_fail)
    log.info("rank %d: finished, %d ok / %d failed this run (shard totals now %d/%d patients)",
              rank, n_ok, n_fail, len(done_patients(step_csv)), len(done_patients(affine_csv)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rank", type=int, default=0, help="shard index, 0-based")
    p.add_argument("--world-size", type=int, default=1, help="number of parallel shards")
    p.add_argument("--sanity", action="store_true")
    args = p.parse_args()
    if args.sanity:
        sys.exit(run_sanity())
    compute_shard(args.rank, args.world_size)


if __name__ == "__main__":
    main()
