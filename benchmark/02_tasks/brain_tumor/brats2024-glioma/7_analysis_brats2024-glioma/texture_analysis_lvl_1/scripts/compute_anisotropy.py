#!/usr/bin/env python
"""
Through-plane / fine-texture anisotropy per patient x contrast x region — BraTS2024-glioma.

Hypothesis (Paul, 2026-09-24): a t2w/t2f-trained segmentation model gets WORSE on t1n test
images with real-fill (rung4->5), but BETTER on t1c and every other eval direction. Pairwise
texture-similarity measures (NGF, correlation ratio) between contrasts failed to explain the
t1n-specific asymmetry. New hypothesis: this is not about how similar t1n's texture is to
another contrast, it's a property of t1n itself — native T1 in this cohort is often a 2D
thick-slice acquisition upsampled to 1mm isotropic, so it is missing fine texture, especially
through the (thick) slice axis, whereas t1c (post-gad, usually acquired as a 3D volumetric
sequence for lesion detection) is not. This script measures per-volume "fine-texture energy"
along each of the 3 array axes and an overall anisotropy ratio, independent of any cross-
contrast comparison.

Method: z-score each contrast's intensities using mean/std over a per-patient brain mask
(healthy | whole_tumor, i.e. all foreground voxels used elsewhere in this analysis). For each
array axis ax in {0,1,2} and each region mask (brain, healthy, SNFH, RC, ET, NCR — regions
with <200 voxels skipped), compute E_ax = mean of squared forward first differences along ax
over voxel PAIRS where both voxels of the pair lie in the mask (mask[:-1] & mask[1:] along ax
via slicing, no interpolation). E_mean = mean(E_0,E_1,E_2) ("fine-texture energy", rotation-
robust up to axis choice). anisotropy = min(E)/max(E) in [0,1]: 1 = isotropic fine texture,
->0 = one axis is much smoother (e.g. the thick-slice axis of a 2D acquisition, possibly after
BraTS's own resampling/registration partially mixes axes — see NOTE below).

NOTE on interpretation: BraTS2024 volumes are registered to a common SRI24 template (rotation
possible), so a native acquisition's true thick-slice axis need not align with an output array
axis. Per-axis anisotropy/argmin can therefore UNDERESTIMATE a real anisotropy that got rotated
across two array axes; E_mean (rotation-mixing-robust magnitude) is reported alongside as the
less axis-committal comparison. Both are reported; do not over-read argmin_axis as literal
scanner geometry without corroboration.

Usage (Vulcan CPU job only, see run_anisotropy.sh):
  python compute_anisotropy.py --sanity
  python compute_anisotropy.py --axcodes-only
  python compute_anisotropy.py --output-dir <dir> [--limit-patients N]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.resolve().parents[6]
DS_ROOT = PROJECT_ROOT / "benchmark" / "02_tasks" / "brain_tumor" / "brats2024-glioma"
BIDS_ROOT = DS_ROOT / "1_BIDS_brats2024-glioma" / "glioma-brain-brats2024"
OUT_DIR_DEFAULT = DS_ROOT / "7_analysis_brats2024-glioma" / "texture_analysis_lvl_1" / "outputs"

sys.path.insert(0, str(THIS_DIR))
from compute_cross_contrast_ngf import load_patient, region_masks, CONTRAST_SUFFIX  # noqa: E402

REGIONS = ("brain", "healthy", "SNFH", "RC", "ET", "NCR")
MIN_REGION_VOX = 200


def per_axis_energy(z: np.ndarray, mask: np.ndarray):
    """Returns (E_0, E_1, E_2, n_pairs_per_axis) — mean squared forward diff over voxel PAIRS
    both inside `mask`, along each of the 3 array axes. No interpolation, no wraparound."""
    E = []
    for ax in range(3):
        d = np.diff(z, axis=ax)
        # pair mask: True where BOTH voxels of the forward-diff pair are in `mask`
        sl_a = [slice(None)] * 3
        sl_a[ax] = slice(0, -1)
        sl_b = [slice(None)] * 3
        sl_b[ax] = slice(1, None)
        pair_mask = mask[tuple(sl_a)] & mask[tuple(sl_b)]
        n = int(pair_mask.sum())
        e = float(np.mean(d[pair_mask] ** 2)) if n > 0 else float("nan")
        E.append((e, n))
    return E


def anisotropy_row(z: np.ndarray, mask: np.ndarray, patient: str, contrast: str, region: str):
    n_vox = int(mask.sum())
    if n_vox < MIN_REGION_VOX:
        return None
    E = per_axis_energy(z, mask)
    e_vals = [e for e, _ in E]
    if any(not np.isfinite(e) for e in e_vals) or min(e_vals) == 0 and max(e_vals) == 0:
        return None
    e_mean = float(np.mean(e_vals))
    anis = float(min(e_vals) / max(e_vals)) if max(e_vals) > 0 else float("nan")
    argmin_axis = int(np.argmin(e_vals))
    return dict(patient=patient, contrast=contrast, region=region, n_vox=n_vox,
                E_0=e_vals[0], E_1=e_vals[1], E_2=e_vals[2], E_mean=e_mean,
                anisotropy=anis, argmin_axis=argmin_axis)


def zscore(vol: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    m = vol[brain_mask]
    mu, sd = float(m.mean()), float(m.std())
    if sd < 1e-8:
        sd = 1.0
    return (vol - mu) / sd


# ───────────────────────── sanity ─────────────────────────
def run_sanity() -> int:
    rng = np.random.default_rng(0)
    D = 48
    ok = True

    # Isotropic random field, smoothed slightly (so it has genuine local structure at all).
    base = rng.standard_normal((D, D, D)).astype(np.float32)
    iso = gaussian_filter1d(base, sigma=1.2, axis=0)
    iso = gaussian_filter1d(iso, sigma=1.2, axis=1)
    iso = gaussian_filter1d(iso, sigma=1.2, axis=2)
    mask = np.ones((D, D, D), dtype=bool)
    mask[:4] = mask[-4:] = mask[:, :4] = mask[:, -4:] = mask[:, :, :4] = mask[:, :, -4:] = False

    print(f"{'case':22s} {'E_0':>10s} {'E_1':>10s} {'E_2':>10s} {'anisotropy':>11s} {'argmin':>7s}")

    def report(name, vol):
        z = zscore(vol, mask)
        row = anisotropy_row(z, mask, "sanity", name, "brain")
        print(f"{name:22s} {row['E_0']:10.4f} {row['E_1']:10.4f} {row['E_2']:10.4f} "
              f"{row['anisotropy']:11.3f} {row['argmin_axis']:7d}")
        return row

    r_iso = report("isotropic", iso)
    ok &= r_iso["anisotropy"] > 0.9

    for blur_axis in range(3):
        blurred = gaussian_filter1d(iso, sigma=2.0, axis=blur_axis)
        r = report(f"blur_axis{blur_axis}", blurred)
        ok &= r["anisotropy"] < 0.7
        ok &= r["argmin_axis"] == blur_axis

    # Thick-slice simulation: box-average (through-plane averaging of a thick slice profile)
    # then subsample every 5th slice along axis 0, then LINEARLY INTERPOLATE back to full
    # resolution (mimics a thick 2D acquisition upsampled to isotropic — smooth loss of
    # high-frequency through-plane info, not a blocky nearest-hold which would spuriously
    # ADD high-frequency step edges at block boundaries and inflate E_0 instead of shrinking it).
    thick_blur = uniform_filter1d(iso, size=5, axis=0, mode="nearest")
    sub_idx = np.arange(0, D, 5)
    sub = thick_blur[sub_idx]
    f = interp1d(sub_idx, sub, axis=0, kind="linear", fill_value="extrapolate")
    thick = f(np.arange(D))
    r_thick = report("thickslice_axis0", thick)
    ok &= r_thick["anisotropy"] < 0.7
    ok &= r_thick["argmin_axis"] == 0

    print(f"\nSANITY {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def run_axcodes_only():
    # Any one patient's T1w BIDS file, as instructed.
    anat_dirs = sorted((BIDS_ROOT).glob("sub-*/anat"))
    if not anat_dirs:
        sys.exit(f"No sub-*/anat dirs under {BIDS_ROOT}")
    anat_dir = anat_dirs[0]
    t1w_candidates = sorted(anat_dir.glob("*_T1w.nii*"))
    t1w_candidates = [f for f in t1w_candidates if "ce-gadolinium" not in f.name]
    if not t1w_candidates:
        sys.exit(f"No plain T1w file under {anat_dir}")
    f = t1w_candidates[0]
    img = nib.load(str(f))
    codes = nib.aff2axcodes(img.affine)
    print(f"File: {f}")
    print(f"Shape: {img.shape}")
    print(f"Affine:\n{img.affine}")
    print(f"aff2axcodes: {codes}  (array axis 0,1,2 -> {codes[0]},{codes[1]},{codes[2]})")
    # Also check for a JSON sidecar with slice-thickness info, if any exists.
    sidecars = sorted(anat_dir.glob("*.json"))
    if sidecars:
        print(f"Sidecar JSON(s) found: {[s.name for s in sidecars]}")
        for s in sidecars:
            print(f"--- {s.name} ---")
            print(s.read_text()[:2000])
    else:
        print("No JSON sidecars found in this anat/ dir (expected for BraTS-derived BIDS).")


COLS = ["patient", "contrast", "region", "n_vox", "E_0", "E_1", "E_2", "E_mean",
        "anisotropy", "argmin_axis"]


def shard_csv_path(data_dir: Path, rank: int) -> Path:
    return data_dir / f"anisotropy_shard{rank}.csv"


def already_done(shard_csv: Path) -> set:
    if not shard_csv.exists():
        return set()
    try:
        return set(pd.read_csv(shard_csv)["patient"].unique())
    except Exception:  # noqa: BLE001 — corrupt/partial file from a killed job
        return set()


def compute_shard(patients, rank, world_size, data_dir):
    my_patients = patients[rank::world_size]
    shard_csv = shard_csv_path(data_dir, rank)
    done = already_done(shard_csv)
    todo = [p for p in my_patients if p not in done]
    log.info("rank %d/%d: %d assigned, %d already done, %d to do",
              rank, world_size, len(my_patients), len(done), len(todo))
    write_header = not shard_csv.exists()
    n_ok, n_fail = 0, 0
    for i, patient_id in enumerate(todo):
        try:
            vols, label = load_patient(patient_id, "cpu")
        except (FileNotFoundError, ValueError) as e:
            n_fail += 1
            log.warning("SKIP %s: %s", patient_id, e)
            continue
        t1n_np = vols["t1n"].numpy()
        masks = region_masks(label, vols["t1n"])
        masks_np = {k: v.numpy() for k, v in masks.items()}
        brain_mask = masks_np["healthy"] | masks_np["whole_tumor"]
        masks_np["brain"] = brain_mask

        rows = []
        for contrast in CONTRAST_SUFFIX.keys():
            vol_np = vols[contrast].numpy()
            z = zscore(vol_np, brain_mask)
            for region in REGIONS:
                m = masks_np[region]
                row = anisotropy_row(z, m, patient_id, contrast, region)
                if row is not None:
                    rows.append(row)
        pd.DataFrame(rows, columns=COLS).to_csv(shard_csv, mode="a", header=write_header, index=False)
        write_header = False
        n_ok += 1
        del vols, label, t1n_np, masks, masks_np
        if (i + 1) % 10 == 0:
            log.info("  rank %d: %d/%d done (%d ok, %d skipped)", rank, i + 1, len(todo), n_ok, n_fail)
    log.info("rank %d: finished, %d ok / %d failed this run", rank, n_ok, n_fail)


def merge(data_dir: Path):
    shard_files = sorted(data_dir.glob("anisotropy_shard*.csv"))
    if not shard_files:
        sys.exit(f"No anisotropy_shard*.csv found in {data_dir} — run shards first")
    df = pd.concat([pd.read_csv(f) for f in shard_files], ignore_index=True)
    out_csv = data_dir / "anisotropy_per_patient.csv"
    df.to_csv(out_csv, index=False)
    log.info("Merged %d shard file(s) -> %d rows -> %s", len(shard_files), len(df), out_csv)
    print(df.groupby(["contrast", "region"])["anisotropy"].median().to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--limit-patients", type=int, default=None)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--axcodes-only", action="store_true")
    args = p.parse_args()

    if args.sanity:
        sys.exit(run_sanity())
    if args.axcodes_only:
        run_axcodes_only()
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge(data_dir)
        return

    deltas_csv = data_dir / "patient_region_deltas.csv"
    patients = sorted(pd.read_csv(deltas_csv)["case"].unique())
    if args.limit_patients:
        patients = patients[: args.limit_patients]
    log.info("%d unique patients", len(patients))
    compute_shard(patients, args.rank, args.world_size, data_dir)


if __name__ == "__main__":
    main()
