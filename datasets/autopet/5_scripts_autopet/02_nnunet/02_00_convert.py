#!/usr/bin/env python3
"""
Split the AutoPET archive's 2-channel (CT+PET) imagesTr into TWO single-channel nnU-Net
raw datasets — Dataset120_AutoPET_CT (channel 0000, CT resampled to PET grid) and
Dataset121_AutoPET_PET (channel 0001, PET in SUV) — using ONLY this project's own split
(4_splits_autopet/{splits_final.json,test_cases.json}, built by
01_create_splits/01_01_create_splits.py), NOT the archive's own reference split.

Only the lesion-POSITIVE FDG cases that ended up in our split (train+val+test union) are
extracted from the (huge, 140GB) zip — negative controls and the PSMA cohort are read
directly out of the zip on demand by 05_predict's build-test-inputs step instead of being
extracted here, so this step touches only the ~1038-case positive-FDG subset, not the
full 1611-study archive (avoids doubling disk usage on $SCRATCH for data we don't use in
this step, and avoids dumping tens of thousands of negative-control files onto a
networked filesystem for nothing — see CLAUDE.md "many small files hammer networked
filesystems").

Per-modality nnU-Net raw layout produced:
  2_nnUNet_autopet/raw/Dataset120_AutoPET_CT/imagesTr/<case>_0000.nii.gz
  2_nnUNet_autopet/raw/Dataset120_AutoPET_CT/labelsTr/<case>.nii.gz
  2_nnUNet_autopet/raw/Dataset120_AutoPET_CT/dataset.json
  (mirrored for Dataset121_AutoPET_PET, channel 0001)

Usage:
  source ../00_utils/env.sh
  .venv/bin/python 02_00_convert.py
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

DATASET_ROOT = Path(os.environ["DATASET_ROOT"]) if "DATASET_ROOT" in os.environ else \
    Path(__file__).resolve().parents[2]
SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", str(DATASET_ROOT / "4_splits_autopet")))
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", str(DATASET_ROOT / "2_nnUNet_autopet" / "raw")))
ARCHIVE_ZIP = Path(os.environ.get(
    "AUTOPET_ARCHIVE_ZIP", "/scratch/paulh/pet_task_staging/autopet/psma-fdg-pet-ct-lesions_v2.zip"))

# (output_dataset_dir, channel_suffix) — channel_suffix matches the archive's own
# imagesTr/<case>_000{0,1}.nii.gz convention (0000=CT, 0001=PET; see FDAT record
# "Structure and usage" section, verified 2026-09-13).
DATASETS = {
    "ct": ("Dataset120_AutoPET_CT", "0000"),
    "pet": ("Dataset121_AutoPET_PET", "0001"),
}


def _all_split_cases() -> list[str]:
    folds = json.loads((SPLITS_DIR / "splits_final.json").read_text())
    test = json.loads((SPLITS_DIR / "test_cases.json").read_text())["test"]
    cases: set[str] = set(test)
    for f in folds:
        cases.update(f["train"])
        cases.update(f["val"])
    return sorted(cases)


def main() -> None:
    cases = _all_split_cases()  # these are CLEAN ids (fdg_<hex10>_<yyyymmdd>)
    print(f"[convert] {len(cases)} positive-FDG cases in this project's split")

    case_id_map: dict[str, str] = json.loads((SPLITS_DIR / "case_id_map.json").read_text())
    # Reverse-lookup: clean id -> the archive's own (possibly messy, space-containing)
    # case id, needed to find the actual zip members — see 01_01_create_splits.py header.

    if not ARCHIVE_ZIP.exists():
        raise FileNotFoundError(f"{ARCHIVE_ZIP} not found — is the download finished?")

    with zipfile.ZipFile(ARCHIVE_ZIP) as zf:
        names = zf.namelist()
        # Build a suffix -> full-member-name index once (avoids an O(cases * names) scan).
        images_index = {n.split("imagesTr/")[-1]: n for n in names if "imagesTr/" in n}
        labels_index = {n.split("labelsTr/")[-1]: n for n in names if "labelsTr/" in n}
        dataset_json_name = next((n for n in names if n.endswith("dataset.json")), None)
        if dataset_json_name is None:
            raise FileNotFoundError("dataset.json not found inside the archive")
        with zf.open(dataset_json_name) as fh:
            archive_dataset_json = json.load(fh)

        for contrast, (out_name, channel) in DATASETS.items():
            out_root = NNUNET_RAW / out_name
            (out_root / "imagesTr").mkdir(parents=True, exist_ok=True)
            (out_root / "labelsTr").mkdir(parents=True, exist_ok=True)

            n_done = 0
            for clean_case in cases:
                archive_case = case_id_map.get(clean_case)
                if archive_case is None:
                    print(f"[convert][WARN] clean id '{clean_case}' not found in "
                          f"case_id_map.json — skipping")
                    continue
                img_suffix = f"{archive_case}_{channel}.nii.gz"
                lbl_suffix = f"{archive_case}.nii.gz"
                img_member = images_index.get(img_suffix)
                lbl_member = labels_index.get(lbl_suffix)
                if img_member is None or lbl_member is None:
                    print(f"[convert][WARN] missing image or label for case '{clean_case}' "
                          f"(archive id '{archive_case}', {contrast}) — "
                          f"img={img_member} lbl={lbl_member}, skipping")
                    continue

                # Output files use the CLEAN id — never the archive's messy, space-
                # containing native name (fragile for shell scripts building paths).
                img_dst = out_root / "imagesTr" / f"{clean_case}_0000.nii.gz"
                lbl_dst = out_root / "labelsTr" / f"{clean_case}.nii.gz"
                if not img_dst.exists():
                    with zf.open(img_member) as src, open(img_dst, "wb") as dst:
                        dst.write(src.read())
                if not lbl_dst.exists():
                    with zf.open(lbl_member) as src, open(lbl_dst, "wb") as dst:
                        dst.write(src.read())
                n_done += 1

            dataset_json = {
                "channel_names": {"0": contrast.upper() if contrast == "ct" else "PET_SUV"},
                "labels": archive_dataset_json.get("labels", {"background": 0, "lesion": 1}),
                "numTraining": n_done,
                "file_ending": ".nii.gz",
            }
            (out_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
            print(f"[convert] {out_name}: {n_done}/{len(cases)} cases written -> {out_root}")


if __name__ == "__main__":
    main()
