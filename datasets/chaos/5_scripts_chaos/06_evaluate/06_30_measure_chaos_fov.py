#!/usr/bin/env python3
"""
Measure the CHAOS field-of-view (FOV) relative to the kidneys (and liver).

CHAOS has a restricted axial FOV — models trained on it never saw anatomy outside a
~2-3 dm superior-inferior slab of the upper abdomen (it even truncates the liver dome
at its top edge). To score chaos-trained models FAIRLY on full-torso test volumes
(AMOS / SLIVER07 / TRUSTED), we restrict evaluation to the CHAOS-equivalent slab,
anchored on a landmark organ present in the test GT (kidneys; liver where there are
no kidneys). This script quantifies that slab from the CHAOS training GT.

For every CHAOS training label volume, for each anchor organ, we take the organ's
S-I bounding box and measure, in mm:
  * sup_mm — distance from the organ's superior edge to the image's superior FOV edge
  * inf_mm — distance from the organ's inferior edge to the image's inferior FOV edge
(S-I axis + direction resolved per-volume from the image geometry, so this is robust
to orientation — see 00_commun_scripts/00_00_utils/fov.py.)

We report the full per-volume distribution and take the **median** (the chosen crop
statistic — the typical CHAOS slab) per (contrast, anchor). Downstream evaluators
expand the test-volume anchor bbox by these medians to build the keep-slab.

Outputs (both regenerable):
  <this dir>/chaos_fov_margins.json                        machine contract (read by eval run scripts)
  8_results_chaos/03_aggregated_results/chaos_fov_margins.md   human-readable distribution report

    bash 06_30_measure_chaos_fov.sh          # (preferred — routes through run_job)
    python 06_30_measure_chaos_fov.py
"""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_00_utils"))
import fov  # noqa: E402

DATASET_ROOT = Path(__file__).resolve().parents[2]
RAW = DATASET_ROOT / "2_nnUNet_chaos" / "raw"
OUT_JSON = Path(__file__).resolve().parent / "chaos_fov_margins.json"
OUT_MD = DATASET_ROOT / "8_results_chaos" / "03_aggregated_results" / "chaos_fov_margins.md"

# contrast -> nnUNet dataset dir holding labelsTr
CONTRAST_DS = {
    "t1in":   "Dataset060_CHAOS_MR_T1in",
    "t2spir": "Dataset061_CHAOS_MR_T2spir",
}
# anchor organ -> chaos GT label id(s)
ANCHORS = {
    "kidney": [2, 3],   # right_kidney + left_kidney
    "liver":  [1],
}


def measure_volume(path, anchor_ids):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    axis, sup_dir, sp = fov.si_axis_sign(img)
    mask = np.isin(arr, anchor_ids)
    bb = fov.bbox_si(mask, axis)
    if bb is None:
        return None
    return fov.edge_margins_mm(arr.shape, axis, sup_dir, sp, bb[0], bb[1])


def main():
    result = {"_unit": "mm",
              "_definition": "sup_mm/inf_mm = median CHAOS distance from the anchor "
                             "organ's S-I bbox to the image FOV edge; the crop statistic is the median.",
              "_generated_from": {c: ds for c, ds in CONTRAST_DS.items()}}
    md = ["# CHAOS FOV margins relative to anchor organs", "",
          "Per-volume distance (mm) from each anchor organ's superior/inferior "
          "bounding-box edge to the image FOV edge, over the CHAOS training GT. "
          "The evaluators use the **median** (`sup_mm`,`inf_mm`) to build the keep-slab.", ""]

    for contrast, ds in CONTRAST_DS.items():
        labels_dir = RAW / ds / "labelsTr"
        vols = sorted(labels_dir.glob("*.nii.gz"))
        result[contrast] = {}
        md += [f"## {contrast}  ({ds}, {len(vols)} volumes)", ""]
        for anchor, ids in ANCHORS.items():
            sups, infs = [], []
            for v in vols:
                m = measure_volume(v, ids)
                if m is not None:
                    sups.append(m[0]); infs.append(m[1])
            if not sups:
                md += [f"- **{anchor}**: not present in any volume", ""]
                continue
            sups, infs = np.array(sups), np.array(infs)
            entry = {
                "sup_mm": round(float(np.median(sups)), 1),
                "inf_mm": round(float(np.median(infs)), 1),
                "n": len(sups),
                "sup_mm_stats": {"min": round(float(sups.min()), 1),
                                 "median": round(float(np.median(sups)), 1),
                                 "mean": round(float(sups.mean()), 1),
                                 "max": round(float(sups.max()), 1)},
                "inf_mm_stats": {"min": round(float(infs.min()), 1),
                                 "median": round(float(np.median(infs)), 1),
                                 "mean": round(float(infs.mean()), 1),
                                 "max": round(float(infs.max()), 1)},
            }
            result[contrast][anchor] = entry
            md += [f"### anchor = {anchor}  (n={len(sups)})", "",
                   f"| margin | min | median (used) | mean | max |",
                   f"|---|---|---|---|---|",
                   f"| superior | {sups.min():.1f} | **{np.median(sups):.1f}** | {sups.mean():.1f} | {sups.max():.1f} |",
                   f"| inferior | {infs.min():.1f} | **{np.median(infs):.1f}** | {infs.mean():.1f} | {infs.max():.1f} |",
                   ""]

    OUT_JSON.write_text(json.dumps(result, indent=2))
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n→ {OUT_JSON}")
    print(f"→ {OUT_MD}")


if __name__ == "__main__":
    main()
