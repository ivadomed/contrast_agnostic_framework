#!/usr/bin/env python
"""
Provenance check for the dwi_ap training arm (2026-09-21): is dwi_ap volume 0 (a b~0
EPI readout, NOT a diffusion-weighted image) appearance-adjacent to contrasts already
in the on-harmony roster, relative to how adjacent the roster's existing contrasts are
to EACH OTHER?

A single number (e.g. r(dwi,epi)=0.34) is uninterpretable without a within-roster
baseline, so this scores every pair of the six contrasts on the same sessions:
T1w, T2w, bold, dwi_ap(vol0), epi_ap, gre_echo1_mag.

Metric: |Spearman rho| over brain voxels (SynthSeg mask of the BASE image), i.e. the
rank-based, contrast- and inversion-invariant statistic this project already uses for
texture comparison (a plain Pearson r reads T1w<->T2w as ~0 or negative because of
contrast inversion, which is not "dissimilar appearance" in the sense that matters).
Pearson r is printed alongside for reference. Same session => same head, so the
statistic reflects appearance, not anatomy. The second image is resampled onto the
first's grid (linear), the mask is the first image's own SynthSeg mask.

Usage: python check_dwi_vs_epi_similarity.py   (run via run_job, not on a login node)
"""
import itertools
import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.stats import spearmanr

BIDS = Path(os.environ["BIDS_ROOT"])
N_SESSIONS = int(os.environ.get("N_SESSIONS", "16"))
random.seed(0)

NAMES = ["T1w", "T2w", "bold", "dwi_ap", "epi_ap", "gre"]


def paths(s, e):
    se = BIDS / s / e
    gre = sorted((se / "swi").glob("*echo-1*part-mag*GRE.nii.gz")) if (se / "swi").exists() else []
    return {
        "T1w": se / "anat" / f"{s}_{e}_T1w.nii.gz",
        "T2w": se / "anat" / f"{s}_{e}_T2w.nii.gz",
        "bold": se / "func" / f"{s}_{e}_task-rest_bold.nii.gz",
        "dwi_ap": se / "dwi" / f"{s}_{e}_dir-AP_dwi.nii.gz",
        "epi_ap": se / "fmap" / f"{s}_{e}_dir-AP_epi.nii.gz",
        "gre": gre[0] if gre else se / "swi" / "MISSING",
    }


def mask_of(p):
    rel = p.relative_to(BIDS)
    return BIDS / "derivatives" / "labels" / rel.parent / (rel.name[:-len(".nii.gz")] + "_label-synthseg_dseg.nii.gz")


def load3d(name, p):
    n = nib.load(str(p))
    if n.ndim == 4:                       # same 4D handling as the eval/training pipeline
        a = np.asarray(n.dataobj)
        a = a.mean(-1) if name == "bold" else a[..., 0]
        return nib.Nifti1Image(a.astype(np.float32), n.affine)
    return n


def pair_stats(base, other, base_mask):
    o = resample_from_to(other, base, order=1)
    m = np.asarray(resample_from_to(base_mask, base, order=0).dataobj) > 0
    a = np.asarray(base.dataobj)[m].astype(np.float64)
    b = np.asarray(o.dataobj)[m].astype(np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 500:
        return np.nan, np.nan
    return abs(spearmanr(a, b).correlation), float(np.corrcoef(a, b)[0, 1])


sessions = []
for sd in sorted(BIDS.glob("sub-*")):
    for se in sorted(sd.glob("ses-*")):
        P = paths(sd.name, se.name)
        if all(P[k].exists() and mask_of(P[k]).exists() for k in NAMES):
            sessions.append((sd.name, se.name, P))
print(f"sessions with all 6 contrasts + masks: {len(sessions)}")
random.shuffle(sessions)
sessions = sessions[:N_SESSIONS]
print(f"sampled {len(sessions)} sessions across {len({s for s, _, _ in sessions})} subjects")

res_s = {}   # (A,B) -> [|rho|]
res_p = {}
for s, e, P in sessions:
    img = {k: load3d(k, P[k]) for k in NAMES}
    msk = {k: nib.load(str(mask_of(P[k]))) for k in NAMES}
    for a, b in itertools.combinations(NAMES, 2):
        rho, r = pair_stats(img[a], img[b], msk[a])       # base = a
        res_s.setdefault((a, b), []).append(rho)
        res_p.setdefault((a, b), []).append(r)

print("\nmedian |Spearman rho| between contrasts (same session, brain voxels)")
print("            " + "".join(f"{n:>8s}" for n in NAMES))
M = {}
for a in NAMES:
    row = []
    for b in NAMES:
        if a == b:
            row.append("    -   ")
            continue
        key = (a, b) if (a, b) in res_s else (b, a)
        v = float(np.nanmedian(res_s[key]))
        M[(a, b)] = v
        row.append(f"{v:8.2f}")
    print(f"{a:>11s} " + "".join(row))

print("\nmedian Pearson r (reference only; sign flips under contrast inversion)")
for (a, b), v in sorted(res_p.items()):
    print(f"  r({a},{b}) = {np.nanmedian(v):+.2f}")

others = [n for n in NAMES if n != "dwi_ap"]
roster = [M[(a, b)] for a, b in itertools.combinations(others, 2)]
dwi = {n: M[("dwi_ap", n)] for n in others}
print(f"\nWITHIN-ROSTER baseline (pairs among {others}): "
      f"median {np.median(roster):.2f}, min {min(roster):.2f}, max {max(roster):.2f}")
print("dwi_ap vs each roster contrast: " + ", ".join(f"{n}={v:.2f}" for n, v in dwi.items()))
top = max(dwi, key=dwi.get)
print(f"dwi_ap's CLOSEST roster contrast: {top} ({dwi[top]:.2f}); "
      f"rank of that pair among all {len(roster) + len(dwi)} pairs: "
      f"{1 + sum(v > dwi[top] for v in roster + list(dwi.values()))}")
