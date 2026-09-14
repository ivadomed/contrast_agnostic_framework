#!/usr/bin/env python3
"""
Materialize the PSMA cohort (LMU Munich, cross-institution/cross-tracer, EVAL-ONLY —
never trained on) as nnU-Net predict inputs under BOTH per-modality training datasets, so
the FDG-trained checkpoints can be scored on it directly with the same predict_common.sh
driver used for the FDG own-test set:

    <nnUNet_raw>/Dataset120_AutoPET_CT/imagesTs_psma_ct/<case>_0000.nii.gz
    <nnUNet_raw>/Dataset120_AutoPET_CT/labelsTs_psma_ct/<case>.nii.gz
    <nnUNet_raw>/Dataset120_AutoPET_CT/imagesTs_psma_pet/<case>_0000.nii.gz
    <nnUNet_raw>/Dataset120_AutoPET_CT/labelsTs_psma_pet/<case>.nii.gz

ALL consolidated under Dataset120_AutoPET_CT's own raw tree — same convention as
05_00_build_test_inputs.py (predict_common.sh's "own" mode resolves imagesTs_<item>
under ONE fixed PREDICT_DATASET_ID_DEFAULT regardless of which per-modality Dataset
actually trained the model or which Dataset an item's real pixel data belongs to).

Unlike 05_00_build_test_inputs.py, these cases were NEVER extracted into imagesTr/ at all
(PSMA never trains — see 00_utils/env.sh) — this script extracts them straight out of the
archive zip, POSITIVE (lesion-present) studies only (539 of 597 — ground truth from
01_00_scan_labels.py's direct per-case label scan, run 2026-09-13; negative-control PSMA
studies are a separate optional false-positive-rate check, not part of the headline eval
pool). Positivity comes from the scan manifest, NOT from psma_metadata.csv — verified
2026-09-13 that file has no diagnosis/lesion column at all (just Subject ID, Study Date,
age, manufacturer_model_name, pet_radionuclide, ct_contrast_agent), so any attempt to
guess one would have failed loudly (by design) rather than silently miscounted.

Run via 05_00b_build_psma_test_inputs.sh — ONCE, after Dataset120 exists and
01_create_splits/01_01_create_splits.py has produced case_id_map.json.
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", str(DATASET_ROOT / "2_nnUNet_autopet" / "raw")))
SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", str(DATASET_ROOT / "4_splits_autopet")))
ARCHIVE_ZIP = Path(os.environ.get(
    "AUTOPET_ARCHIVE_ZIP", "/scratch/paulh/pet_task_staging/autopet/psma-fdg-pet-ct-lesions_v2.zip"))

# (item_name, channel_suffix) — channel_suffix matches the archive's own
# imagesTr/<case>_000{0,1}.nii.gz convention (0000=CT, 0001=PET). Both items land under
# the SAME target dataset (Dataset120) — see module docstring.
TARGET_DATASET = "Dataset120_AutoPET_CT"
TARGETS = {
    "ct": ("psma_ct", "0000"),
    "pet": ("psma_pet", "0001"),
}


def _positive_psma_clean_ids() -> list[str]:
    scan = json.loads((SPLITS_DIR / "label_scan_manifest.json").read_text())
    case_id_map: dict[str, str] = json.loads((SPLITS_DIR / "case_id_map.json").read_text())
    archive_to_clean = {v: k for k, v in case_id_map.items()}
    return sorted(
        archive_to_clean[m["case_id"]]
        for m in scan["manifest"]
        if m["tracer"] == "psma" and m["n_positive_voxels"] > 0
    )


def main() -> None:
    if not ARCHIVE_ZIP.exists():
        raise FileNotFoundError(f"{ARCHIVE_ZIP} not found — is the download finished?")

    positive_clean_ids = _positive_psma_clean_ids()
    case_id_map: dict[str, str] = json.loads((SPLITS_DIR / "case_id_map.json").read_text())
    print(f"[psma_test_inputs] {len(positive_clean_ids)} positive PSMA studies "
          f"(ground truth from label_scan_manifest.json, not the metadata CSV)")

    with zipfile.ZipFile(ARCHIVE_ZIP) as zf:
        names = zf.namelist()
        images_index = {n.split("imagesTr/")[-1]: n for n in names if "imagesTr/" in n}
        labels_index = {n.split("labelsTr/")[-1]: n for n in names if "labelsTr/" in n}

        ds = NNUNET_RAW / TARGET_DATASET
        for _contrast, (item, channel) in TARGETS.items():
            img_ts, lab_ts = ds / f"imagesTs_{item}", ds / f"labelsTs_{item}"
            img_ts.mkdir(parents=True, exist_ok=True)
            lab_ts.mkdir(parents=True, exist_ok=True)

            n_done, missing = 0, []
            for clean_case in positive_clean_ids:
                archive_case = case_id_map[clean_case]
                img_member = images_index.get(f"{archive_case}_{channel}.nii.gz")
                lbl_member = labels_index.get(f"{archive_case}.nii.gz")
                if img_member is None or lbl_member is None:
                    missing.append(clean_case)
                    continue
                # Output files use the CLEAN id, same rule as 02_00_convert.py.
                img_dst = img_ts / f"{clean_case}_0000.nii.gz"
                lbl_dst = lab_ts / f"{clean_case}.nii.gz"
                if not img_dst.exists():
                    with zf.open(img_member) as src, open(img_dst, "wb") as dst:
                        dst.write(src.read())
                if not lbl_dst.exists():
                    with zf.open(lbl_member) as src, open(lbl_dst, "wb") as dst:
                        dst.write(src.read())
                n_done += 1

            print(f"[psma_test_inputs] {TARGET_DATASET}/imagesTs_{item}: {n_done} cases "
                  f"written, {len(missing)} missing")


if __name__ == "__main__":
    main()
