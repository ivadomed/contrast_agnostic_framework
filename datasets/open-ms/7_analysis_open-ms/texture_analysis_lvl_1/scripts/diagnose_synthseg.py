#!/usr/bin/env python
"""
Diagnostic: why does synthseg_em/noem score above the noisefill_v2 floor on census_r1?
Two hypotheses tested on already-generated noblur volumes (FLAIR source, all 30 subjects):
  H1 (blur/smoothing): synthseg's output is locally smoother than noisefill_v2's (evidence
      of an internal blur/PSF/resample step not controlled by our blur_sigmas=0 knob).
  H2 (boundary-sharing, not texture): the excess census concentrates in a thin shell near
      the ROI boundary (shared anatomical geometry from the same lesion mask), not spread
      uniformly through the eroded interior (which would suggest genuine texture leakage).
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

REPO = Path("/project/6102268/paulh/mri_synthesis_project")
sys.path.insert(0, str(REPO / "datasets/open-ms/7_analysis_open-ms/texture_analysis_lvl_1/scripts"))
from compute_texture_metrics_openms import (
    rank_transform, abscorr, erode, list_source_keys, foreground_mask, RANK_RADII,
)

GENERATED = REPO / "datasets/open-ms/7_analysis_open-ms/data/generated_noblur"
METHODS = ["palette", "synthseg_em", "synthseg_noem", "v26_6_2_noisefill_v2", "auglab_default"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load(p):
    arr = np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[..., 0]
    return torch.from_numpy(arr).to(device)

def local_smoothness(x, m):
    """Mean |x - mean_of_6_neighbors| within mask m — lower = smoother (blurrier)."""
    xp = x[None, None]
    kernel = torch.zeros(1, 1, 3, 3, 3, device=x.device)
    kernel[0, 0, 1, 1, 0] = kernel[0, 0, 1, 1, 2] = 1
    kernel[0, 0, 1, 0, 1] = kernel[0, 0, 1, 2, 1] = 1
    kernel[0, 0, 0, 1, 1] = kernel[0, 0, 2, 1, 1] = 1
    kernel /= 6.0
    nbr_mean = F.conv3d(xp, kernel, padding=1)[0, 0]
    return float((x - nbr_mean).abs()[m].mean().item())

source_keys = list_source_keys("FLAIR")
print(f"{len(source_keys)} FLAIR subjects\n")

results = {m: {"smooth_lesion": [], "smooth_fg": [],
               "census_interior": [], "census_shell": [], "n_interior": [], "n_shell": []}
           for m in METHODS}

for i, (key, (img_path, les_path)) in enumerate(sorted(source_keys.items())):
    source = load(img_path)
    from compute_texture_metrics_openms import _load_lesion
    lesion = _load_lesion(les_path, source.shape, device)
    fg = foreground_mask(source)
    src_ranks = {r: rank_transform(source, r) for r in RANK_RADII}

    # interior (eroded 3x) vs shell (original minus eroded) of the FOREGROUND roi
    # (lesion is too small/thin to erode 3x without vanishing for most subjects)
    interior = erode(fg, 3)
    shell = fg & (~interior)

    for m in METHODS:
        mdir = GENERATED / m / key
        runs = sorted(mdir.glob(f"{key}_run-*.nii.gz"))
        if not runs:
            continue
        synth = load(runs[0])  # run-00 only, enough for a diagnostic
        if synth.shape != source.shape:
            continue
        results[m]["smooth_lesion"].append(local_smoothness(synth, lesion))
        results[m]["smooth_fg"].append(local_smoothness(synth, fg))

        syn_ranks = {r: rank_transform(synth, r) for r in RANK_RADII}
        n_int, n_sh = int(interior.sum()), int(shell.sum())
        if n_int > 200:
            results[m]["census_interior"].append(abscorr(src_ranks[1], syn_ranks[1], interior))
            results[m]["n_interior"].append(n_int)
        if n_sh > 200:
            results[m]["census_shell"].append(abscorr(src_ranks[1], syn_ranks[1], shell))
            results[m]["n_shell"].append(n_sh)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(source_keys)} done", flush=True)

# also compute source's own smoothness as a reference point
src_smooth_lesion, src_smooth_fg = [], []
for key, (img_path, les_path) in sorted(source_keys.items()):
    source = load(img_path)
    from compute_texture_metrics_openms import _load_lesion
    lesion = _load_lesion(les_path, source.shape, device)
    fg = foreground_mask(source)
    src_smooth_lesion.append(local_smoothness(source, lesion))
    src_smooth_fg.append(local_smoothness(source, fg))

print("\n=== H1: local smoothness (mean |x - local 6-neighbor mean|; LOWER = smoother/blurrier) ===")
print(f"{'method':22s} {'lesion':>10s} {'foreground':>10s}")
print(f"{'SOURCE (real)':22s} {np.mean(src_smooth_lesion):10.4f} {np.mean(src_smooth_fg):10.4f}")
for m in METHODS:
    sl, sf = results[m]["smooth_lesion"], results[m]["smooth_fg"]
    if sl:
        print(f"{m:22s} {np.mean(sl):10.4f} {np.mean(sf):10.4f}")

print("\n=== H2: census_r1, foreground ROI, interior (eroded x3) vs boundary shell ===")
print(f"{'method':22s} {'interior':>10s} {'shell':>10s} {'shell-interior':>15s} {'n_int(med)':>10s} {'n_shell(med)':>12s}")
for m in METHODS:
    ci, cs = results[m]["census_interior"], results[m]["census_shell"]
    if ci and cs:
        print(f"{m:22s} {np.mean(ci):10.4f} {np.mean(cs):10.4f} {np.mean(cs)-np.mean(ci):15.4f} "
              f"{np.median(results[m]['n_interior']):10.0f} {np.median(results[m]['n_shell']):12.0f}")
