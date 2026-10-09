#!/usr/bin/env python3
"""
Materialize on-harmony's shared cross-contrast test set as nnU-Net predict inputs.

Extracted verbatim (same numpy/nibabel logic, unchanged) from the python heredoc that used
to live inline in 06_evaluate/06_01_evaluate_testset.sh's LAUNCHER section — moved here so
prediction follows the standard 3-tier pattern (05_00_build_test_inputs -> predict_common.sh)
instead of being rebuilt inline on every eval run. Idempotent: built once, reused by every
run/model (the test set is identical across all training contrasts/methods).

For each of the 6 held-out test contrasts (T1w, T2w, bold, dwi_ap, epi_ap, gre_echo1_mag):
  - 4D volumes are reduced to 3D: bold is averaged across the time axis, dwi_ap/epi_ap take
    the first volume (b~0 EPI framework, NOT diffusion-weighted -- see env_dwi.sh), GRE
    multi-echo series select echo-1 magnitude via find_gre().
  - The reduced native-space image is RAS-reoriented (nib.as_closest_canonical) before being
    used as the prediction INPUT -- nnU-Net does not reorient, training data is RAS, and
    feeding native LAS (bold/dwi/epi) would mirror the brain for the network (only
    mirror-augmented models would cope -- an artifact that inflates v26_6_2 over the rest).
  - GT (31-class SynthSeg parcellation) is remapped from raw FreeSurfer label IDs to the
    project's compact 1..31 label space.

Writes THREE things per contrast:
  TESTSET/<contrast>/images_native/<cid>_0000.nii.gz   -- native geometry (resample ref, matches GT)
  TESTSET/<contrast>/images_ras/<cid>_0000.nii.gz       -- RAS-canonical (prediction input)
  TESTSET/<contrast>/gt_native/<cid>.nii.gz             -- remapped GT, native geometry
  <nnUNet_raw>/<TARGET_DS>/imagesTs_<contrast>/<cid>_0000.nii.gz  -- hardlink of images_ras,
      so predict_common.sh's standard flat-input contract (own mode, ONE fixed
      PREDICT_DATASET_ID_DEFAULT hosting every contrast regardless of which per-contrast
      Dataset actually trained a given model -- same convention as totalseg-pelvic's
      05_00_build_test_inputs.py) is satisfied without duplicating these files' content.

TESTSET = PREDICTIONS_ROOT/<MODEL_TYPE>/_test_set (env var TESTSET, set by the .sh wrapper).
Run via 05_00_build_test_inputs.sh.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib

BIDS = Path(os.environ["BIDS_ROOT"])
TESTSET = Path(os.environ["TESTSET"])
NNUNET_RAW = Path(os.environ["nnUNet_raw"])
TEST_CASES = Path(os.environ["TEST_CASES"])

# Canonical dataset dir that hosts imagesTs_<contrast> for ALL contrasts, regardless of
# which per-contrast Dataset (031/032/033) actually trained a given model -- same "one
# consolidated target dataset" convention predict_common.sh's own-mode expects (see
# totalseg-pelvic's 05_00_build_test_inputs.py docstring for the shared rationale).
TARGET_DATASET = os.environ.get("PREDICT_TARGET_DATASET", "Dataset031_OnHarmonyT1w31")

_FS_IDS_31 = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28,
              41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]
_MAXFS = max(_FS_IDS_31)
_LUT = np.zeros(_MAXFS + 2, np.uint8)
for i, fs in enumerate(_FS_IDS_31):
    _LUT[fs] = i + 1


def remap(nib_img):
    a = np.asarray(nib_img.dataobj).astype(np.int32)
    return nib.Nifti1Image(_LUT[np.clip(a, 0, _MAXFS + 1)].astype(np.uint8), nib_img.affine, nib_img.header)


def find_gre(sd):
    swi = sd / "swi"
    if not swi.exists():
        return None
    c = sorted(swi.glob("*echo-1*part-mag*GRE.nii.gz"))
    return c[0] if c else None


CONTRASTS = {
    "T1w": lambda s, e, b: b / s / e / "anat" / f"{s}_{e}_T1w.nii.gz",
    "T2w": lambda s, e, b: b / s / e / "anat" / f"{s}_{e}_T2w.nii.gz",
    "bold": lambda s, e, b: b / s / e / "func" / f"{s}_{e}_task-rest_bold.nii.gz",
    "dwi_ap": lambda s, e, b: b / s / e / "dwi" / f"{s}_{e}_dir-AP_dwi.nii.gz",
    "epi_ap": lambda s, e, b: b / s / e / "fmap" / f"{s}_{e}_dir-AP_epi.nii.gz",
    "gre_echo1_mag": lambda s, e, b: find_gre(b / s / e),
}


def _link_into_nnunet_raw(contrast: str, ras_dir: Path) -> None:
    dst_dir = NNUNET_RAW / TARGET_DATASET / f"imagesTs_{contrast}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(ras_dir.glob("*.nii.gz")):
        dst = dst_dir / src.name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        n += 1
    print(f"[build_test_inputs] {TARGET_DATASET}/imagesTs_{contrast}: {n} new links (of {len(list(ras_dir.glob('*.nii.gz')))} total)")


def main() -> None:
    test_cases = json.loads(TEST_CASES.read_text())
    found = 0
    for contrast, fn in CONTRASTS.items():
        idir = TESTSET / contrast / "images_native"; idir.mkdir(parents=True, exist_ok=True)
        rdir = TESTSET / contrast / "images_ras";    rdir.mkdir(parents=True, exist_ok=True)
        gdir = TESTSET / contrast / "gt_native";     gdir.mkdir(parents=True, exist_ok=True)
        for tc in test_cases:
            s, e = tc["subject"], tc["session"]
            src = fn(s, e, BIDS)
            if src is None or not src.exists():
                continue
            cid = f"{s}_{e}_{contrast}"
            out = idir / f"{cid}_0000.nii.gz"
            if not out.exists():
                n = nib.load(str(src))
                if n.ndim > 3:
                    arr = np.asarray(n.dataobj)
                    v = arr.mean(axis=-1) if contrast == "bold" else arr[..., 0]
                    nib.save(nib.Nifti1Image(v.astype(np.float32), n.affine, n.header), str(out))
                else:
                    shutil.copy2(src, out)
            # RAS-canonical copy = the PREDICTION input. See module docstring for why.
            rout = rdir / f"{cid}_0000.nii.gz"
            if not rout.exists():
                nib.save(nib.as_closest_canonical(nib.load(str(out))), str(rout))
            found += 1
            rel = src.relative_to(BIDS)
            m = BIDS / "derivatives" / "labels" / rel.parent / (rel.name[:-len(".nii.gz")] + "_label-synthseg_dseg.nii.gz")
            gout = gdir / f"{cid}.nii.gz"
            if m.exists() and not gout.exists():
                nib.save(remap(nib.load(str(m))), str(gout))
        _link_into_nnunet_raw(contrast, rdir)
    print(f"shared test set ready ({found} image refs).")


if __name__ == "__main__":
    main()
