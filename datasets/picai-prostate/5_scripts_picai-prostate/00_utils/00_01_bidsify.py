#!/usr/bin/env python3
"""
BIDSify PI-CAI: 0_raw_picai-prostate (MHA, three native grids) → 1_BIDS_picai-prostate
(NIfTI, ONE common grid per study).

WHY A COMMON GRID (the one non-obvious step in this dataset)
------------------------------------------------------------
PI-CAI ships each study as three series acquired in one session but reconstructed onto
three different grids: axial T2W at ~0.3x0.3x3.0 mm, and ADC/HBV at ~2x2x3.0 mm. This
project's cross-contrast protocol needs the SAME lesion mask to be valid for every test
contrast (open-ms gets this for free — its T2W/T1W are already co-registered to FLAIR).
So we resample all three series, plus the label, onto one reference grid per study:

    spacing (0.5, 0.5, 3.0) mm,  size (256, 256, 24) voxels,  T2W's direction cosines,
    centred on the prostate (AI whole-gland mask centroid; image centre as fallback).

That is deliberately the geometry the PI-CAI challenge's own baseline preprocessing uses
(picai_prep's PreprocessingSettings matrix_size=[20,256,256], spacing=[3.0,0.5,0.5]) — we
only widen 20→24 slices for a little more cranio-caudal context. Because the three series
share a scanner frame of reference, resampling by physical coordinates co-registers them;
no image-based registration is performed or needed.

Resampling to 0.5 mm also downsamples T2W from ~0.3 mm, which is what makes nnU-Net's
3d_fullres patches affordable here — at native 0.3 mm the in-plane matrix is ~2x larger in
each axis for no added anatomical information at lesion scale.

CASE SELECTION
--------------
Only studies whose HUMAN-EXPERT csPCa delineation is non-empty are kept (~425 of 1500).
The other ~1075 studies are csPCa-negative, i.e. all-zero masks: keeping them would make
per-case Dice undefined for 70% of the test set and would silently dominate the headline
numbers. So picai-prostate is "segment the lesion in a known-positive exam", stated
explicitly in datasets/picai-prostate/README.md. The AI-derived (Bosma22a) lesion
delineations are NOT used — human expert only.

Output (case id = "<patient_id>_<study_id>", e.g. 10417_1000424):
  <BIDS_ROOT>/sub-<pid>/ses-<sid>/anat/sub-<pid>_ses-<sid>_{T2w,ADC,HBV}.nii.gz
  <BIDS_ROOT>/derivatives/manual_masks/sub-<pid>/ses-<sid>/anat/sub-<pid>_ses-<sid>_dseg.nii.gz
  <BIDS_ROOT>/dataset_description.json
  <BIDS_ROOT>/cases.json          manifest: case id -> {patient, study, n_lesion_voxels}

Run:  bash 00_01_bidsify.sh          (dispatches through run_job — CPU-only, parallel)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ── Reference grid (see module docstring) ────────────────────────────────────────────
REF_SPACING = (0.5, 0.5, 3.0)      # SimpleITK (x, y, z) mm
REF_SIZE = (256, 256, 24)          # SimpleITK (x, y, z) voxels
CONTRASTS = {"t2w": "T2w", "adc": "ADC", "hbv": "HBV"}   # raw mha suffix -> BIDS suffix


def _find_series(images_root: Path) -> dict[str, dict[str, Path]]:
    """case_id -> {'t2w': path, 'adc': path, 'hbv': path} from the extracted MHA tree.

    Globs recursively so it works whether the zips extracted to
    <root>/picai_public_images_fold0/<pid>/... or straight to <root>/<pid>/...
    """
    series: dict[str, dict[str, Path]] = {}
    for mha in images_root.rglob("*.mha"):
        stem = mha.name[: -len(".mha")]
        for suffix in CONTRASTS:
            tail = f"_{suffix}"
            if stem.endswith(tail):
                case_id = stem[: -len(tail)]
                series.setdefault(case_id, {})[suffix] = mha
                break
    return series


def _reference_grid(t2w: sitk.Image, gland: sitk.Image | None) -> sitk.Image:
    """Build the per-study reference grid: fixed spacing/size + T2W direction, centred on
    the prostate centroid (gland mask) or, failing that, on the T2W image centre."""
    direction = t2w.GetDirection()

    centre = None
    if gland is not None:
        arr = sitk.GetArrayFromImage(gland)          # (z, y, x)
        idx = np.argwhere(arr > 0)
        if idx.size:
            # mean index in (z,y,x) -> SimpleITK (x,y,z) continuous index -> physical point
            zyx = idx.mean(axis=0)
            centre = np.array(gland.TransformContinuousIndexToPhysicalPoint(
                (float(zyx[2]), float(zyx[1]), float(zyx[0]))))
    if centre is None:
        size = np.array(t2w.GetSize(), dtype=float)
        centre = np.array(t2w.TransformContinuousIndexToPhysicalPoint(
            tuple((size - 1) / 2.0)))

    # origin = centre - R @ (spacing * (size-1)/2)
    rot = np.array(direction, dtype=float).reshape(3, 3)
    half = np.array(REF_SPACING) * (np.array(REF_SIZE) - 1) / 2.0
    origin = centre - rot @ half

    ref = sitk.Image(list(REF_SIZE), sitk.sitkFloat32)
    ref.SetSpacing(REF_SPACING)
    ref.SetDirection(direction)
    ref.SetOrigin(tuple(float(v) for v in origin))
    return ref


def _resample(img: sitk.Image, ref: sitk.Image, *, label: bool) -> sitk.Image:
    return sitk.Resample(
        img, ref,
        sitk.Transform(),                                   # identity: shared scanner frame
        sitk.sitkNearestNeighbor if label else sitk.sitkBSpline,
        0.0,
        sitk.sitkUInt8 if label else sitk.sitkFloat32,
    )


def _process(case_id: str, paths: dict[str, Path], lesion_p: Path, gland_p: Path | None,
             bids_root: Path) -> tuple[str, str, int]:
    """Returns (case_id, status, n_lesion_voxels). status: 'ok' | 'skip:<why>' | 'error:<why>'."""
    try:
        missing = [c for c in CONTRASTS if c not in paths]
        if missing:
            return case_id, f"skip:missing_contrast({','.join(missing)})", 0
        if not lesion_p.is_file():
            return case_id, "skip:no_human_expert_label", 0

        lesion_native = sitk.ReadImage(str(lesion_p))
        n_native = int((sitk.GetArrayFromImage(lesion_native) > 0).sum())
        if n_native == 0:
            return case_id, "skip:csPCa_negative", 0

        t2w = sitk.ReadImage(str(paths["t2w"]))
        gland = sitk.ReadImage(str(gland_p)) if gland_p and gland_p.is_file() else None
        ref = _reference_grid(t2w, gland)

        lesion = _resample(lesion_native, ref, label=True)
        n_vox = int((sitk.GetArrayFromImage(lesion) > 0).sum())
        if n_vox == 0:
            # The lesion fell outside the 128x128x72 mm prostate-centred slab — that would
            # be an empty ground truth in a "known-positive" dataset, so drop the case
            # rather than ship a degenerate one.
            return case_id, "skip:lesion_outside_reference_grid", 0

        pid, sid = case_id.split("_", 1)
        anat = bids_root / f"sub-{pid}" / f"ses-{sid}" / "anat"
        anat.mkdir(parents=True, exist_ok=True)
        for suffix, bids_suffix in CONTRASTS.items():
            out = _resample(sitk.ReadImage(str(paths[suffix])), ref, label=False)
            sitk.WriteImage(out, str(anat / f"sub-{pid}_ses-{sid}_{bids_suffix}.nii.gz"), True)

        deriv = bids_root / "derivatives" / "manual_masks" / f"sub-{pid}" / f"ses-{sid}" / "anat"
        deriv.mkdir(parents=True, exist_ok=True)
        # Binarise: PI-CAI encodes one integer per distinct lesion; the segmentation task
        # here is a single csPCa class (as open-ms does for MS lesions).
        lesion = sitk.Cast(lesion > 0, sitk.sitkUInt8)
        sitk.WriteImage(lesion, str(deriv / f"sub-{pid}_ses-{sid}_dseg.nii.gz"), True)
        return case_id, "ok", n_vox
    except Exception as exc:  # noqa: BLE001 — one bad study must not kill the whole run
        return case_id, f"error:{type(exc).__name__}: {exc}\n{traceback.format_exc()}", 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("BIDSIFY_WORKERS", "16")))
    ap.add_argument("--limit", type=int, default=0, help="debug: only process the first N studies")
    args = ap.parse_args()

    raw_root = Path(os.environ["RAW_ROOT"]).resolve()
    bids_root = Path(os.environ["BIDS_ROOT"]).resolve()
    bids_root.mkdir(parents=True, exist_ok=True)

    images_root = raw_root / "images"
    labels_root = raw_root / "picai_labels"
    lesion_dir = labels_root / "csPCa_lesion_delineations" / "human_expert" / "resampled"
    gland_dir = labels_root / "anatomical_delineations" / "whole_gland" / "AI" / "Bosma22b"
    for p in (images_root, lesion_dir):
        if not p.is_dir():
            sys.exit(f"ERROR: expected directory missing: {p} (run 00_00_download.sh first)")
    if not gland_dir.is_dir():
        print(f"WARNING: whole-gland masks not at {gland_dir} — falling back to image-centre "
              f"cropping for every study.", flush=True)
        gland_dir = None

    series = _find_series(images_root)
    case_ids = sorted(series)
    if args.limit:
        case_ids = case_ids[: args.limit]
    print(f"found {len(case_ids)} studies under {images_root}", flush=True)

    results: dict[str, tuple[str, int]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                _process, cid, series[cid],
                lesion_dir / f"{cid}.nii.gz",
                (gland_dir / f"{cid}.nii.gz") if gland_dir else None,
                bids_root,
            ): cid
            for cid in case_ids
        }
        for n, fut in enumerate(as_completed(futs), 1):
            cid, status, n_vox = fut.result()
            results[cid] = (status, n_vox)
            if n % 100 == 0:
                print(f"  {n}/{len(case_ids)} processed", flush=True)

    kept = {c: v[1] for c, v in results.items() if v[0] == "ok"}
    errors = {c: v[0] for c, v in results.items() if v[0].startswith("error")}
    skips: dict[str, int] = {}
    for c, v in results.items():
        if v[0].startswith("skip:"):
            skips[v[0]] = skips.get(v[0], 0) + 1

    manifest = {
        cid: {"patient": cid.split("_", 1)[0], "study": cid.split("_", 1)[1],
              "n_lesion_voxels": kept[cid]}
        for cid in sorted(kept)
    }
    (bids_root / "cases.json").write_text(json.dumps(manifest, indent=2))
    (bids_root / "dataset_description.json").write_text(json.dumps({
        "Name": "picai-prostate (PI-CAI public training/development, csPCa-positive subset)",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
        "License": "CC-BY-NC-4.0",
        "Acknowledgements": "PI-CAI Challenge (Saha et al. 2022); lesion delineations from "
                            "github.com/DIAGNijmegen/picai_labels (human expert).",
        "ReferencesAndLinks": ["https://doi.org/10.5281/zenodo.6624726",
                               "https://pi-cai.grand-challenge.org/"],
    }, indent=2))

    print(f"\nBIDS → {bids_root}")
    print(f"  kept (csPCa-positive, all 3 contrasts): {len(kept)}")
    for reason, n in sorted(skips.items()):
        print(f"  {reason}: {n}")
    if errors:
        print(f"  ERRORS: {len(errors)}")
        for cid, msg in list(errors.items())[:5]:
            print(f"    {cid}: {msg.splitlines()[0]}")
        sys.exit(f"ERROR: {len(errors)} studies failed — see above")
    print(f"  patients: {len({c.split('_', 1)[0] for c in kept})}")


if __name__ == "__main__":
    main()
