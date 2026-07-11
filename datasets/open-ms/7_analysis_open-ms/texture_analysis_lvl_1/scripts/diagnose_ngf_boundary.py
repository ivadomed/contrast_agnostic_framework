#!/usr/bin/env python
"""
Diagnostic: is noisefill_v2 / synthseg_noem's above-floor NGF (0.5-0.65, vs the 0.333 chance
floor) genuine interior texture leakage, or a shared-ROI-mask-boundary artifact (every method's
fill respects the same lesion/foreground mask edge, so the mask's own boundary contributes a
"free" gradient-orientation match to ALL methods, textured or not)?

Test: interior (eroded away from the ROI mask edge) vs boundary-shell NGF, mirroring the
interior/shell decomposition already used to diagnose SynthSeg's blur artifact
(diagnose_synthseg.py). If noisefill_v2/synthseg_noem's excess sits in the SHELL and their
INTERIOR already equals the chance floor, this is the same class of "shared macro boundary"
confound, not new information.
"""
import sys
from pathlib import Path
import torch
import nibabel as nib
import numpy as np

REPO = Path("/project/6102268/paulh/mri_synthesis_project")
SCRIPTS = REPO / "datasets/open-ms/7_analysis_open-ms/texture_analysis_lvl_1/scripts"
sys.path.insert(0, str(SCRIPTS))
from compute_texture_metrics_openms import list_source_keys, foreground_mask, _load_lesion, erode
from compute_ngf_texture import ngf_scores

GENERATED = REPO / "datasets/open-ms/7_analysis_open-ms/data/generated_noblur"
METHODS = ["palette", "auglab_default", "synthseg_em", "synthseg_noem", "v26_6_2_noisefill_v2"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(p):
    arr = np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[..., 0]
    return torch.from_numpy(arr).to(device)


source_keys = list_source_keys("FLAIR")
print(f"{len(source_keys)} FLAIR subjects\n", flush=True)

agg = {m: {"foreground_interior": [], "foreground_shell": [],
           "lesion_interior": [], "lesion_shell": []} for m in METHODS}

for i, (key, (img_path, les_path)) in enumerate(sorted(source_keys.items())):
    source = load(img_path)
    lesion = _load_lesion(les_path, source.shape, device)
    fg = foreground_mask(source)
    rois = {"foreground": fg, "lesion": lesion}

    for m in METHODS:
        runs = sorted((GENERATED / m / key).glob(f"{key}_run-*.nii.gz"))
        if not runs:
            continue
        synth = load(runs[0])
        if synth.shape != source.shape:
            continue
        for rn, mask in rois.items():
            interior = erode(mask, 3)
            shell = mask & (~interior)
            if int(interior.sum()) < 200 or int(shell.sum()) < 200:
                continue
            ia, _, _ = ngf_scores(source, synth, interior)
            sa, _, _ = ngf_scores(source, synth, shell)
            agg[m][f"{rn}_interior"].append(ia)
            agg[m][f"{rn}_shell"].append(sa)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(source_keys)}", flush=True)


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


print("\n=== NGF: interior (eroded x3) vs boundary-shell of the ROI mask ===")
print(f"{'method':22s} {'fg_interior':>12s} {'fg_shell':>10s} {'les_interior':>13s} {'les_shell':>10s}")
for m in METHODS:
    print(f"{m:22s} {mean(agg[m]['foreground_interior']):12.3f} {mean(agg[m]['foreground_shell']):10.3f} "
          f"{mean(agg[m]['lesion_interior']):13.3f} {mean(agg[m]['lesion_shell']):10.3f}")
print("\nchance floor (independent gradients, 3-D) = 0.333")
print("If interior ≈ 0.333 and shell ≫ 0.333: excess is the shared-ROI-mask-boundary artifact.")
print("If interior ≫ 0.333: genuine — the method's fill correlates with source structure away from any edge.")
