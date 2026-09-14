#!/usr/bin/env python3
"""
Package the Duke-Breast-Cancer-MRI (MAMA-MIA expert-mask subset) cohort as a
PRECONTRAST-only, TEST-ONLY nnU-Net dataset for cross-dataset evaluation of
I-SPY2-trained models -- a third test contrast alongside t1wce/t2w. Modeled
directly on 02_01_convert_test_t1wce.py (see that file for the fuller
docstring); differs only in which DCE channel is copied as the image.

Source (already staged on Vulcan's $SCRATCH from the same Synapse download
used for t1wce -- no new download needed):
  <RAW_STAGE>/DUKE_XXX/DUKE_XXX_000{0..4}.nii.gz   (5-channel MAMA-MIA nnU-Net
                                                     preprocessing: pre-contrast
                                                     + 4 post-contrast DCE phases)
  <RAW_STAGE>/DUKE_XXX/DUKE_XXX_mask.nii.gz         (expert tumour segmentation)

Why this matters: I-SPY2's own BIDS README confirms `t1wce` training data is
UNIFORMLY the first post-contrast temporal position of a standardized 7-phase
VOLSER DCE stack (0=pre-contrast, 1-6=post), applied consistently across the
whole 22-site trial cohort -- not a mix of phases. So testing our t1wce/t2w-
trained models against Duke's channel _0000 (true pre-contrast, no Gd
enhancement at all) is a clean, single-variable new-contrast test: same
patients/masks as the t1wce test set, only the contrast timing differs.

Channel selection verified the same way as 02_01 (empirical enhancement
check, not assumed from documentation): _0000 = pre-contrast, _0001 = first
post-contrast (used only for the enhancement sanity check here, not copied).

Output layout (eval-only convention, same as duke's t1wce test set):
  2_nnUNet_duke-breast-mri/raw/imagesTs_precontrast/duke_XXX_0000.nii.gz
  2_nnUNet_duke-breast-mri/raw/labelsTs_precontrast/duke_XXX.nii.gz
  2_nnUNet_duke-breast-mri/raw/duke_test_precontrast.json

Run:  bash 02_nnunet/02_02_convert_test_precontrast.sh
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import nibabel as nib

DATASET_ROOT = Path(__file__).resolve().parents[2]           # datasets/duke-breast-mri
REPO_ROOT = DATASET_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "datasets" / "00_commun_scripts" / "00_00_utils"))
from orient import TARGET_AXCODES, reorient_file  # noqa: E402

RAW_STAGE = Path("/scratch/paulh/duke_download/raw")   # login-node download staging area
OUT = DATASET_ROOT / "2_nnUNet_duke-breast-mri" / "raw"
MANIFEST_IN = Path("/scratch/paulh/duke_download/duke_full_manifest.json")

PRECONTRAST_CHAN = "_0000.nii.gz"
POSTCONTRAST_CHAN = "_0001.nii.gz"   # used only for the enhancement sanity check


def duke_cases() -> list[str]:
    full = json.load(open(MANIFEST_IN))
    ok = []
    for pid, entry in full.items():
        chans = entry.get("channels", {})
        have_pre = chans.get(f"{pid}{PRECONTRAST_CHAN}") in ("ok", "cached")
        have_post = chans.get(f"{pid}{POSTCONTRAST_CHAN}") in ("ok", "cached")
        have_mask = entry.get("mask") in ("ok", "cached")
        if have_pre and have_post and have_mask:
            ok.append(pid)
    return sorted(ok)


def axcodes_of(path: Path) -> str:
    return "".join(nib.aff2axcodes(nib.load(str(path)).affine))


def main() -> None:
    pids = duke_cases()
    if not pids:
        raise SystemExit(f"No complete Duke cases found via {MANIFEST_IN} / {RAW_STAGE} -- "
                          f"has the download finished?")

    (OUT / "imagesTs_precontrast").mkdir(parents=True, exist_ok=True)
    (OUT / "labelsTs_precontrast").mkdir(parents=True, exist_ok=True)

    manifest = {"cases": [], "modality": "precontrast"}
    raw_axcodes_img = Counter()
    raw_axcodes_lab = Counter()
    reoriented_img, reoriented_lab = 0, 0
    n_empty_mask, n_enhancement_fail = 0, 0

    for pid in pids:
        case_id = pid.lower()
        pdir = RAW_STAGE / pid
        pre_src = pdir / f"{pid}{PRECONTRAST_CHAN}"
        post_src = pdir / f"{pid}{POSTCONTRAST_CHAN}"
        mask_src = pdir / f"{pid}_mask.nii.gz"

        img_dst = OUT / "imagesTs_precontrast" / f"{case_id}_0000.nii.gz"
        lab_dst = OUT / "labelsTs_precontrast" / f"{case_id}.nii.gz"

        # record RAW axcodes before any reorientation (this is the actual evidence
        # the orientation-distribution report is built from)
        raw_img_ax = axcodes_of(pre_src)
        raw_lab_ax = axcodes_of(mask_src)
        raw_axcodes_img[raw_img_ax] += 1
        raw_axcodes_lab[raw_lab_ax] += 1

        shutil.copyfile(pre_src, img_dst)
        shutil.copyfile(mask_src, lab_dst)

        img_status = reorient_file(img_dst, TARGET_AXCODES)
        lab_status = reorient_file(lab_dst, TARGET_AXCODES)
        if img_status == "fixed":
            reoriented_img += 1
        if lab_status == "fixed":
            reoriented_lab += 1

        img_nib = nib.load(str(img_dst))
        lab_nib = nib.load(str(lab_dst))
        arr = np.asanyarray(lab_nib.dataobj)
        vals = set(np.unique(arr).astype(int).tolist())
        if not vals <= {0, 1}:
            raise RuntimeError(f"{case_id}: unexpected label values {sorted(vals)} "
                               f"(need binary {{0,1}})")
        n_vox = int((arr == 1).sum())
        if n_vox == 0:
            n_empty_mask += 1

        if img_nib.shape != lab_nib.shape:
            raise RuntimeError(f"{case_id}: image/label shape mismatch "
                               f"{img_nib.shape} vs {lab_nib.shape}")

        # empirical channel-order sanity check: post-contrast should enhance
        # within the tumour ROI relative to pre-contrast (confirms _0000 really
        # is the pre-contrast channel we intend to test against)
        pre_arr = np.asanyarray(nib.load(str(pre_src)).dataobj).astype(np.float64)
        post_arr = np.asanyarray(nib.load(str(post_src)).dataobj).astype(np.float64)
        enhancement_ok = None
        if n_vox > 0:
            mask_bool = arr == 1
            if pre_arr.shape == mask_bool.shape:
                mean_pre = float(pre_arr[mask_bool].mean())
                mean_post = float(post_arr[mask_bool].mean())
                enhancement_ok = mean_post > mean_pre
                if not enhancement_ok:
                    n_enhancement_fail += 1

        manifest["cases"].append({
            "case_id": case_id,
            "source_id": pid,
            "label_voxels": n_vox,
            "image_shape": list(img_nib.shape),
            "image_zooms": [round(float(z), 3) for z in img_nib.header.get_zooms()],
            "raw_image_axcodes": raw_img_ax,
            "raw_label_axcodes": raw_lab_ax,
            "final_axcodes": "".join(TARGET_AXCODES),
            "image_reoriented": img_status == "fixed",
            "label_reoriented": lab_status == "fixed",
            "enhancement_check_post_gt_pre": enhancement_ok,
        })

    manifest["n_cases"] = len(manifest["cases"])
    manifest["orientation_audit"] = {
        "raw_image_axcodes_distribution": dict(raw_axcodes_img),
        "raw_label_axcodes_distribution": dict(raw_axcodes_lab),
        "n_images_reoriented_to_LPS": reoriented_img,
        "n_labels_reoriented_to_LPS": reoriented_lab,
        "canonical_target": "".join(TARGET_AXCODES),
    }
    manifest["qa"] = {
        "n_empty_masks": n_empty_mask,
        "n_enhancement_check_failed": n_enhancement_fail,
        "channel_convention": f"{PRECONTRAST_CHAN} = pre-contrast (copied as image), "
                              f"{POSTCONTRAST_CHAN} = first post-contrast (used only for the "
                              f"enhancement sanity check, verified via intra-tumour enhancement, "
                              f"not assumed from documentation)",
    }
    manifest["provenance"] = {
        "role": "test-only external evaluation set (models trained on ispy2, pre-contrast "
                "NATIVE T1w -- genuinely different contrast from t1wce/t2w training "
                "modalities, no Gd enhancement)",
        "source": "MAMA-MIA (Garrucho et al. 2025, Synapse syn60868042), Duke-Breast-Cancer-MRI "
                  "subset only -- ISPY1/ISPY2/NACT-Pilot cases explicitly excluded",
        "labels": {"background": 0, "tumour": 1},
        "licence": "CC BY-NC 4.0 (inherited from TCIA Duke-Breast-Cancer-MRI; NOT eligible for "
                   "this project's public CC-BY-only git-annex redistribution list)",
        "reference": "Garrucho et al., Scientific Data 2025, "
                     "https://doi.org/10.1038/s41597-025-04707-4; "
                     "https://doi.org/10.7937/C7X1-YN57 (Duke-Breast-Cancer-MRI, TCIA)",
    }
    (OUT / "duke_test_precontrast.json").write_text(json.dumps(manifest, indent=2))

    print(f"Duke-Breast-MRI PRECONTRAST TEST set written -> {OUT}")
    print(f"  cases: {manifest['n_cases']}")
    print(f"  raw image axcodes distribution: {dict(raw_axcodes_img)}")
    print(f"  raw label axcodes distribution: {dict(raw_axcodes_lab)}")
    print(f"  images reoriented to {''.join(TARGET_AXCODES)}: {reoriented_img}/{manifest['n_cases']}")
    print(f"  labels reoriented to {''.join(TARGET_AXCODES)}: {reoriented_lab}/{manifest['n_cases']}")
    print(f"  empty masks: {n_empty_mask}  enhancement-check failures: {n_enhancement_fail}")


if __name__ == "__main__":
    main()
