#!/usr/bin/env python3
"""
Derive the BILATERAL / UNILATERAL field-of-view variants for every I-SPY2 patient.

CONTEXT (2026-09-04 pivot)
--------------------------
ispy2 is being converted from a cross-eval-only dataset into the PRIMARY TRAINING
set for the breast-cancer task (replacing ambl, 99 cases, which overfit). Because
I-SPY2's DCE (T1wce) series come in two incompatible field-of-view flavours, the
training set is built with BOTH FOVs present as explicit, separately-named cases,
so the model sees the bilateral and unilateral acquisition geometries as two
augmentation-relevant variants of the same anatomy rather than as an uncontrolled
protocol confound.

VERIFIED FACTS THIS SCRIPT BUILDS ON (do not re-litigate)
---------------------------------------------------------
* I-SPY2 has NO native unilateral acquisition. 443 of 561 manifest patients carry
  DICOM SeriesDescription "ISPY2: VOLSER: uni-lateral cropped: original DCE" — a
  *derived half-FOV crop* of an originally bilateral scan. There is no bilateral
  DCE series for those patients anywhere in TCIA (checked via the TCIA API), so
  bilateral T1wce is obtainable ONLY for the natively-bilateral patients.
* T2w is ALWAYS bilateral, for every patient.
* Laterality classification is NOT taken from the manifest alone: it is taken from
  4_splits_ispy2/unilateral_fov_exclusions.json, the prior manual audit which
  reclassified 4 "uni-lateral cropped"-tagged patients as genuinely bilateral after
  visual inspection, and records 1 manifest patient (ISPY2-335091) with no BIDS
  conversion. Net over the 560 patients that actually have BIDS data:
  438 unilateral / 122 bilateral (NOT the manifest's raw 443/118).

CROP-BOX DERIVATION
-------------------
All 560 x 4 volumes (T1wce, T2w and both label files) are LPS
(nib.aff2axcodes == ('L','P','S')), verified over the whole cohort — axis 0 = L
(low index = patient RIGHT), axis 1 = P (low index = ANTERIOR), axis 2 = S.

The VOLSER "uni-lateral cropped" reconstruction was measured against the patients'
own bilateral T2w by mapping the T1w voxel-grid corners into world space and then
into T2w index space (affines only — never voxel indices, the two grids differ).
Over the 438 unilateral patients the crop is an exactly-half IN-PLANE box:
    axis 0 span = 0.500 of the T2 axis-0 size (median), sitting either at
                  [0.001, 0.519] (n=230, patient RIGHT breast) or
                  [0.487, 0.993] (n=208, patient LEFT breast)
    axis 1 span = 0.500 (median), window [0.127, 0.654] (median lo/hi) — i.e. the
                  anterior ~half, which is where the breasts are
    axis 2 (S-I) is NOT halved (median T2/T1 ratio 1.18, range 0.80-2.74) — the
                  slice slab simply differs between the DCE and T2 sequences.
So the unilateral FOV is a 2-D in-plane (axes 0 and 1) crop. This script therefore
crops ONLY axes 0 and 1 and leaves axis 2 at each volume's native extent —
cropping axis 2 as well would silently fold "the two sequences cover different
slabs" into the FOV variable we are trying to isolate.

Two derivation paths:

(A) NATIVELY-UNILATERAL patients (438) — the unilateral box is KNOWN exactly: it is
    that patient's own T1wce world extent. It is mapped into the T2w grid via
    inv(A_T2) @ A_T1 corners, giving an anatomically exact, side-correct
    unilateral T2w. Their T1wce needs no derivation (it IS the unilateral variant),
    and NO bilateral T1wce is fabricated for them — it does not exist.

(B) NATIVELY-BILATERAL patients (122) — there is no unilateral T1wce to read a box
    off, so the box is synthesised on the T2w grid to match the measured VOLSER
    geometry above:
      axis 0: the exact half of the volume containing the LESION CENTROID
              (T2w-space mask) — [0, n0//2) or [n0//2, n0). Side selection is by
              lesion laterality so the derived unilateral scan is clinically
              meaningful (a real unilateral breast MRI images the affected breast).
      axis 1: the empirical median VOLSER window [0.127*n1, 0.654*n1).
      axis 2: full.
    SAFETY RULE: the window is then minimally shifted (and, only if the lesion is
    genuinely wider than the window, expanded) so that it fully contains the
    lesion bounding box. Every case that needed a shift or an expansion, and every
    case whose lesion crosses the axis-0 midline, is recorded in the report — a
    crop that clips the tumour is a silent disaster, so it is never allowed to
    happen quietly. The resulting box is converted to world coordinates and applied
    to BOTH that patient's T2w and T1wce grids, so their unilateral variants cover
    the same anatomy.

Boxes are clipped to the target grid. For 41 of the 438 unilateral patients the T1w
in-plane extent is not fully inside the T2w grid (min containment 0.84); those T2w
crops are clipped and the containment fraction is recorded per case.

OUTPUTS (nothing existing is modified or deleted)
-------------------------------------------------
New BIDS files alongside the native ones (derived variants only — a variant that IS
the native file is not duplicated, it is referenced in the manifest):
    1_BIDS_ispy2/breast-ispy2/sub-ispy2<PID>/anat/
        sub-ispy2<PID>_acq-firstpostuni_T1w.nii.gz    (bilateral patients only)
        sub-ispy2<PID>_acq-uni_T2w.nii.gz             (all 560)
    .../derivatives/labels/sub-ispy2<PID>/anat/
        sub-ispy2<PID>_acq-firstpostuni_T1w_label-lesion_seg.nii.gz
        sub-ispy2<PID>_acq-uni_desc-sharedframe_T2w_label-lesion_seg.nii.gz
    4_splits_ispy2/fov_variants.json    <- the manifest: for every
        (patient, contrast, fov) it records the image + label path actually to be
        used (native or derived), the crop box, the side, orientation checks and
        the lesion-voxel survival counts. This is the SINGLE source of truth read
        by 01_03_create_splits.py and by both 02_nnunet converters — nothing
        downstream re-derives the case list or re-guesses which file is which FOV.
    4_splits_ispy2/fov_variants_report.md   <- human-readable verification summary.

PREREQUISITE — RUN 01_01_rederive_t2w_labels.sh FIRST
-----------------------------------------------------
The T2w-space labels this script reads are the CORRECTED, shared-DICOM-frame masks
(`sub-*_desc-sharedframe_T2w_label-lesion_seg.nii.gz`), not the original
`sub-*_T2w_label-lesion_seg.nii.gz`. The originals were produced by rigid Mattes-MI
registration and 01_00_audit_t2w_label_registration.py showed them to be in the wrong
place for 91.8% of the unilateral and 27.9% of the bilateral patients (median centroid
error 101 mm on the unilateral group, 44% of them in the CONTRALATERAL breast). Using
them here would have made every derived unilateral T2w crop look like it had "lost the
lesion". See 4_splits_ispy2/t2w_label_registration_audit.md.

Run:  bash 01_create_splits/01_02_derive_fov_variants.sh   (submits via run_job —
      ~1250 volumes are decompressed, cropped and re-gzipped, far over the
      login-node ~10 CPU-min allowance).
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib

DATASET_ROOT = Path(__file__).resolve().parents[2]              # datasets/ispy2
BIDS_ROOT = DATASET_ROOT / "1_BIDS_ispy2" / "breast-ispy2"
DERIV_ROOT = BIDS_ROOT / "derivatives" / "labels"
SPLITS = DATASET_ROOT / "4_splits_ispy2"
EXCLUSIONS = SPLITS / "unilateral_fov_exclusions.json"

# Native BIDS suffixes — IMAGES.
NATIVE_SUFFIX = {"t1wce": "acq-firstpost_T1w", "t2w": "T2w"}
# Native BIDS suffixes — LABELS. NOTE the T2w label is NOT `T2w_label-lesion_seg`:
# that file is the old registration-derived mask, which 01_00_audit_t2w_label_
# registration.py showed to be in the wrong place for 91.8% of unilateral and 27.9%
# of bilateral patients. This pipeline consumes ONLY the corrected shared-DICOM-frame
# mask written by 01_01_rederive_t2w_labels.py. Run that step first.
NATIVE_LABEL_SUFFIX = {"t1wce": "acq-firstpost_T1w", "t2w": "desc-sharedframe_T2w"}
# Derived (unilateral) BIDS suffixes.
UNI_SUFFIX = {"t1wce": "acq-firstpostuni_T1w", "t2w": "acq-uni_T2w"}
UNI_LABEL_SUFFIX = {"t1wce": "acq-firstpostuni_T1w", "t2w": "acq-uni_desc-sharedframe_T2w"}

EXPECTED_AXCODES = ("L", "P", "S")

# Empirical median VOLSER axis-1 (anterior-posterior) window, measured over the 438
# natively-unilateral patients (see module docstring). Used ONLY for path (B).
AX1_LO_FRAC, AX1_HI_FRAC = 0.127, 0.654

N_WORKERS = int(os.environ.get("ISPY2_FOV_WORKERS", "16"))


# ── path helpers ────────────────────────────────────────────────────────────────
def img_path(sub: str, suffix: str) -> Path:
    return BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.nii.gz"


def lbl_path(sub: str, suffix: str) -> Path:
    return DERIV_ROOT / sub / "anat" / f"{sub}_{suffix}_label-lesion_seg.nii.gz"


# ── geometry ────────────────────────────────────────────────────────────────────
def _corners_world(aff: np.ndarray, shape) -> np.ndarray:
    """World coords of the 8 outer VOXEL CORNERS (index -0.5 .. n-0.5)."""
    idx = np.array([[i, j, k]
                    for i in (-0.5, shape[0] - 0.5)
                    for j in (-0.5, shape[1] - 0.5)
                    for k in (-0.5, shape[2] - 0.5)])
    return (aff @ np.c_[idx, np.ones(8)].T).T[:, :3]


def world_box_to_index(aff_target: np.ndarray, corners: np.ndarray):
    """Axis-aligned bounding box, in the TARGET grid's index space, of a world-space
    box given by its 8 corners. Handles (small) obliquity differences between the
    two grids — which is exactly why this goes through world coordinates rather
    than comparing voxel indices."""
    inv = np.linalg.inv(aff_target)
    idx = (inv @ np.c_[corners, np.ones(len(corners))].T).T[:, :3]
    return idx.min(0), idx.max(0)


def slice_from_corner_range(lo: float, hi: float, n: int):
    """Voxel index range [start, stop) whose voxel CENTRES lie inside the corner
    range [lo, hi], clipped to [0, n)."""
    start = int(max(0, np.ceil(lo - 1e-6)))
    stop = int(min(n, np.floor(hi + 1e-6) + 1))
    if stop <= start:                       # degenerate — keep at least one voxel
        start = int(np.clip(round((lo + hi) / 2), 0, n - 1)); stop = start + 1
    return start, stop


def crop(img: nib.Nifti1Image, box):
    """box = ((s0,e0),(s1,e1),(s2,e2)) in index space. Returns a new image with the
    affine translated so world coordinates of every retained voxel are unchanged."""
    arr = np.asanyarray(img.dataobj)
    sl = tuple(slice(s, e) for s, e in box)
    sub = arr[sl]
    aff = np.asarray(img.affine).copy()
    aff[:3, 3] = aff[:3, 3] + aff[:3, :3] @ np.array([b[0] for b in box], dtype=float)
    out = nib.Nifti1Image(np.ascontiguousarray(sub), aff)
    out.header.set_zooms(img.header.get_zooms()[:3])
    out.set_data_dtype(img.get_data_dtype())
    return out


def bbox_1d(mask: np.ndarray, axis: int):
    """(lo, hi_exclusive) of nonzero extent along `axis`, or None if empty."""
    other = tuple(a for a in range(mask.ndim) if a != axis)
    prof = mask.any(axis=other)
    nz = np.flatnonzero(prof)
    return (int(nz[0]), int(nz[-1]) + 1) if nz.size else None


# ── per-patient work ────────────────────────────────────────────────────────────
def process(sub: str, native_fov: str) -> dict:
    """native_fov: 'uni' or 'bil' — the FOV of this patient's NATIVE T1wce."""
    rec: dict = {"sub": sub, "native_t1wce_fov": native_fov, "warnings": []}

    t1 = nib.load(str(img_path(sub, NATIVE_SUFFIX["t1wce"])))
    t2 = nib.load(str(img_path(sub, NATIVE_SUFFIX["t2w"])))
    l1 = nib.load(str(lbl_path(sub, NATIVE_LABEL_SUFFIX["t1wce"])))
    l2 = nib.load(str(lbl_path(sub, NATIVE_LABEL_SUFFIX["t2w"])))

    for tag, im in (("t1wce", t1), ("t2w", t2), ("t1wce_label", l1), ("t2w_label", l2)):
        ax = nib.aff2axcodes(im.affine)
        rec[f"axcodes_native_{tag}"] = "".join(ax)
        if ax != EXPECTED_AXCODES:
            rec["warnings"].append(f"native {tag} axcodes {ax} != {EXPECTED_AXCODES}")

    m1 = np.asanyarray(l1.dataobj) > 0
    m2 = np.asanyarray(l2.dataobj) > 0
    rec["lesion_vox_native_t1wce"] = int(m1.sum())
    rec["lesion_vox_native_t2w"] = int(m2.sum())
    if rec["lesion_vox_native_t2w"] == 0:
        rec["warnings"].append("native T2w label is EMPTY")
    if rec["lesion_vox_native_t1wce"] == 0:
        rec["warnings"].append("native T1wce label is EMPTY")

    n2 = t2.shape

    if native_fov == "uni":
        # (A) box = this patient's own T1wce world extent, in-plane only.
        corners = _corners_world(np.asarray(t1.affine), t1.shape)
        lo, hi = world_box_to_index(np.asarray(t2.affine), corners)
        box2 = [slice_from_corner_range(lo[k], hi[k], n2[k]) for k in (0, 1)]
        box2.append((0, n2[2]))
        # how much of the T1 in-plane extent actually fits inside the T2 grid
        cont = []
        for k in (0, 1):
            span = hi[k] - lo[k]
            clipped = min(hi[k], n2[k] - 0.5) - max(lo[k], -0.5)
            cont.append(round(float(max(0.0, clipped) / span), 4))
        rec["t2_box_containment_inplane"] = cont
        if min(cont) < 0.98:
            rec["warnings"].append(f"T1 in-plane extent only {cont} contained in T2 grid (crop clipped)")
        rec["side_rule"] = "derived_from_native_unilateral_T1wce_world_extent"
    else:
        # (B) synthesise the VOLSER-style box on the T2 grid; side by lesion centroid.
        if m2.any():
            cen0 = float(np.argwhere(m2)[:, 0].mean())
            lb0 = bbox_1d(m2, 0); lb1 = bbox_1d(m2, 1)
        elif m1.any():
            # fall back to the T1wce mask mapped into T2 index space
            idx1 = np.argwhere(m1).astype(float)
            w = (np.asarray(l1.affine) @ np.c_[idx1, np.ones(len(idx1))].T).T[:, :3]
            j = (np.linalg.inv(np.asarray(t2.affine)) @ np.c_[w, np.ones(len(w))].T).T[:, :3]
            cen0 = float(j[:, 0].mean())
            lb0 = (int(np.floor(j[:, 0].min())), int(np.ceil(j[:, 0].max())) + 1)
            lb1 = (int(np.floor(j[:, 1].min())), int(np.ceil(j[:, 1].max())) + 1)
            rec["warnings"].append("side chosen from T1wce-space mask (T2w mask empty)")
        else:
            cen0, lb0, lb1 = n2[0] / 2.0, None, None
            rec["warnings"].append("NO lesion in either mask — side fell back to axis-0 low half")

        half = n2[0] // 2
        if cen0 < n2[0] / 2.0:
            s0, e0 = 0, half
            side = "right"          # LPS: low axis-0 index = patient RIGHT
        else:
            s0, e0 = n2[0] - half, n2[0]
            side = "left"
        if lb0 is not None and not (s0 <= lb0[0] and lb0[1] <= e0):
            rec["warnings"].append(
                f"lesion axis-0 bbox {lb0} crosses the chosen {side}-half [{s0},{e0}) — half expanded to contain it")
            s0, e0 = min(s0, lb0[0]), max(e0, lb0[1])

        s1 = int(round(AX1_LO_FRAC * n2[1])); e1 = int(round(AX1_HI_FRAC * n2[1]))
        s1, e1 = max(0, s1), min(n2[1], e1)
        if lb1 is not None:
            width = e1 - s1
            if lb1[1] - lb1[0] > width:
                rec["warnings"].append(
                    f"lesion axis-1 extent {lb1} wider than the VOLSER window ({width}) — window expanded")
                s1, e1 = min(s1, lb1[0]), max(e1, lb1[1])
            else:
                shift = 0
                if lb1[0] < s1:
                    shift = lb1[0] - s1
                elif lb1[1] > e1:
                    shift = lb1[1] - e1
                if shift:
                    s1 = int(np.clip(s1 + shift, 0, n2[1] - width))
                    e1 = s1 + width
                    rec["warnings"].append(f"axis-1 window shifted by {shift} voxels to contain the lesion")
        box2 = [(s0, e0), (s1, e1), (0, n2[2])]
        rec["side"] = side
        rec["side_rule"] = "lesion_centroid_half (T2w-space mask, LPS axis 0)"

    rec["box_t2_index"] = [list(b) for b in box2]

    # ---- write the derived unilateral T2w + its label -------------------------
    out_t2 = crop(t2, box2)
    out_l2 = crop(l2, box2)
    rec["axcodes_uni_t2w"] = "".join(nib.aff2axcodes(out_t2.affine))
    rec["axcodes_uni_t2w_label"] = "".join(nib.aff2axcodes(out_l2.affine))
    kept2 = int((np.asanyarray(out_l2.dataobj) > 0).sum())
    rec["lesion_vox_uni_t2w"] = kept2
    rec["lesion_survival_t2w"] = (round(kept2 / rec["lesion_vox_native_t2w"], 4)
                                  if rec["lesion_vox_native_t2w"] else None)
    if rec["lesion_vox_native_t2w"] and kept2 == 0:
        rec["warnings"].append("CROP LOST THE LESION ENTIRELY on T2w")
    elif rec["lesion_survival_t2w"] is not None and rec["lesion_survival_t2w"] < 1.0:
        rec["warnings"].append(f"T2w crop clipped the lesion (survival {rec['lesion_survival_t2w']})")
    rec["shape_uni_t2w"] = list(out_t2.shape)
    _save(out_t2, img_path(sub, UNI_SUFFIX["t2w"]))
    _save(out_l2, lbl_path(sub, UNI_LABEL_SUFFIX["t2w"]), as_uint8=True)

    # ---- bilateral patients also get a derived unilateral T1wce ---------------
    if native_fov == "bil":
        # exact world box of the T2 crop -> T1 index space, so the two unilateral
        # variants of this patient cover the same anatomy.
        corners = _corners_world(np.asarray(out_t2.affine), out_t2.shape)
        lo, hi = world_box_to_index(np.asarray(t1.affine), corners)
        n1 = t1.shape
        box1 = [slice_from_corner_range(lo[k], hi[k], n1[k]) for k in (0, 1)]
        box1.append((0, n1[2]))
        rec["box_t1_index"] = [list(b) for b in box1]
        out_t1 = crop(t1, box1)
        out_l1 = crop(l1, box1)
        rec["axcodes_uni_t1wce"] = "".join(nib.aff2axcodes(out_t1.affine))
        rec["axcodes_uni_t1wce_label"] = "".join(nib.aff2axcodes(out_l1.affine))
        kept1 = int((np.asanyarray(out_l1.dataobj) > 0).sum())
        rec["lesion_vox_uni_t1wce"] = kept1
        rec["lesion_survival_t1wce"] = (round(kept1 / rec["lesion_vox_native_t1wce"], 4)
                                        if rec["lesion_vox_native_t1wce"] else None)
        if rec["lesion_vox_native_t1wce"] and kept1 == 0:
            rec["warnings"].append("CROP LOST THE LESION ENTIRELY on T1wce")
        elif rec["lesion_survival_t1wce"] is not None and rec["lesion_survival_t1wce"] < 1.0:
            rec["warnings"].append(f"T1wce crop clipped the lesion (survival {rec['lesion_survival_t1wce']})")
        rec["shape_uni_t1wce"] = list(out_t1.shape)
        _save(out_t1, img_path(sub, UNI_SUFFIX["t1wce"]))
        _save(out_l1, lbl_path(sub, UNI_LABEL_SUFFIX["t1wce"]), as_uint8=True)

    return rec


def _save(img: nib.Nifti1Image, dst: Path, as_uint8: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if as_uint8:
        arr = np.asanyarray(img.dataobj).astype(np.uint8)
        img = nib.Nifti1Image(arr, img.affine)
        img.set_data_dtype(np.uint8)
    nib.save(img, str(dst))


# ── driver ──────────────────────────────────────────────────────────────────────
def load_laterality() -> dict:
    """sub -> 'uni' | 'bil', from the prior MANUAL AUDIT (not the raw manifest tag —
    it reclassified 4 mis-tagged patients; see module docstring)."""
    excl = json.loads(EXCLUSIONS.read_text())
    lat = {c["sub"]: "uni" for c in excl["excluded_unilateral_cases"]}
    for c in excl["manual_overrides_reclassified_as_bilateral"]:
        lat[c["sub"]] = "bil"
    subs = sorted(p.name for p in BIDS_ROOT.glob("sub-ispy2*") if p.is_dir())
    for s in subs:
        lat.setdefault(s, "bil")        # bilateral-tagged patients aren't listed in the file
    return {s: lat[s] for s in subs}


def _worker(args):
    sub, fov = args
    try:
        return process(sub, fov)
    except Exception as exc:                       # noqa: BLE001 — record, never hide
        return {"sub": sub, "native_t1wce_fov": fov, "failed": repr(exc), "warnings": ["FAILED"]}


def main() -> None:
    lat = load_laterality()
    print(f"{len(lat)} patients with BIDS data: "
          f"{sum(v == 'uni' for v in lat.values())} unilateral / "
          f"{sum(v == 'bil' for v in lat.values())} bilateral", flush=True)

    items = sorted(lat.items())
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        recs = list(ex.map(_worker, items, chunksize=4))
    recs.sort(key=lambda r: r["sub"])

    failed = [r for r in recs if "failed" in r]
    if failed:
        print(f"!! {len(failed)} patients FAILED:", flush=True)
        for r in failed[:20]:
            print("   ", r["sub"], r["failed"], flush=True)

    # ---- build the (patient, contrast, fov) -> file manifest ------------------
    variants: dict[str, dict] = {}
    for r in recs:
        if "failed" in r:
            continue
        sub = r["sub"]
        pid = sub.removeprefix("sub-ispy2")
        v: dict[str, dict] = {}
        # T2w: native file IS the bilateral variant; the derived crop is unilateral.
        v["t2w/bil"] = {"image": f"{sub}/anat/{sub}_{NATIVE_SUFFIX['t2w']}.nii.gz",
                        "label": f"{sub}/anat/{sub}_{NATIVE_LABEL_SUFFIX['t2w']}_label-lesion_seg.nii.gz",
                        "source": "native"}
        v["t2w/uni"] = {"image": f"{sub}/anat/{sub}_{UNI_SUFFIX['t2w']}.nii.gz",
                        "label": f"{sub}/anat/{sub}_{UNI_LABEL_SUFFIX['t2w']}_label-lesion_seg.nii.gz",
                        "source": "derived"}
        if r["native_t1wce_fov"] == "uni":
            v["t1wce/uni"] = {"image": f"{sub}/anat/{sub}_{NATIVE_SUFFIX['t1wce']}.nii.gz",
                              "label": f"{sub}/anat/{sub}_{NATIVE_LABEL_SUFFIX['t1wce']}_label-lesion_seg.nii.gz",
                              "source": "native"}
            # NO t1wce/bil — a bilateral DCE does not exist for these patients.
        else:
            v["t1wce/bil"] = {"image": f"{sub}/anat/{sub}_{NATIVE_SUFFIX['t1wce']}.nii.gz",
                              "label": f"{sub}/anat/{sub}_{NATIVE_LABEL_SUFFIX['t1wce']}_label-lesion_seg.nii.gz",
                              "source": "native"}
            v["t1wce/uni"] = {"image": f"{sub}/anat/{sub}_{UNI_SUFFIX['t1wce']}.nii.gz",
                              "label": f"{sub}/anat/{sub}_{UNI_LABEL_SUFFIX['t1wce']}_label-lesion_seg.nii.gz",
                              "source": "derived"}
        variants[sub] = {"pid": pid, "native_t1wce_fov": r["native_t1wce_fov"],
                         "side": r.get("side"), "side_rule": r.get("side_rule"),
                         "variants": v}

    SPLITS.mkdir(parents=True, exist_ok=True)
    (SPLITS / "fov_variants.json").write_text(json.dumps(
        {"dataset": "ispy2",
         "generated_by": "5_scripts_ispy2/01_create_splits/01_00_derive_fov_variants.py",
         "bids_root": str(BIDS_ROOT.relative_to(DATASET_ROOT)),
         "deriv_root": str(DERIV_ROOT.relative_to(DATASET_ROOT)),
         "note": "Paths are relative to bids_root (images) / deriv_root (labels). "
                 "'source':'native' means the existing untouched BIDS file already IS "
                 "that FOV variant; 'derived' means this script wrote it. Patients "
                 "whose native T1wce is unilateral have NO t1wce/bil entry — a "
                 "bilateral DCE series does not exist for them in TCIA.",
         "patients": variants,
         "per_patient_verification": {r["sub"]: r for r in recs}}, indent=1))

    _write_report(recs, variants)
    print(f"-> {SPLITS/'fov_variants.json'}")
    print(f"-> {SPLITS/'fov_variants_report.md'}")


def _write_report(recs: list, variants: dict) -> None:
    ok = [r for r in recs if "failed" not in r]
    uni = [r for r in ok if r["native_t1wce_fov"] == "uni"]
    bil = [r for r in ok if r["native_t1wce_fov"] == "bil"]
    lost, clipped, warned = [], [], []
    for r in ok:
        for c in ("t2w", "t1wce"):
            s = r.get(f"lesion_survival_{c}")
            if s is None:
                continue
            if s == 0.0:
                lost.append((r["sub"], c, r.get(f"lesion_vox_native_{c}")))
            elif s < 1.0:
                clipped.append((r["sub"], c, s))
        if r["warnings"]:
            warned.append(r)
    bad_ax = [r["sub"] for r in ok
              for k, v in r.items() if k.startswith("axcodes_") and v != "LPS"]
    L = []
    L.append("# ispy2 FOV-variant derivation — verification report\n")
    L.append(f"Patients processed: **{len(ok)}** ({len(uni)} natively unilateral, "
             f"{len(bil)} natively bilateral); failures: {len(recs)-len(ok)}\n")
    L.append("## Case counts per contrast per FOV\n")
    n_t2_bil = sum(1 for v in variants.values() if "t2w/bil" in v["variants"])
    n_t2_uni = sum(1 for v in variants.values() if "t2w/uni" in v["variants"])
    n_t1_bil = sum(1 for v in variants.values() if "t1wce/bil" in v["variants"])
    n_t1_uni = sum(1 for v in variants.values() if "t1wce/uni" in v["variants"])
    L.append("| contrast | bilateral | unilateral | total |")
    L.append("|---|---|---|---|")
    L.append(f"| t1wce | {n_t1_bil} | {n_t1_uni} | {n_t1_bil+n_t1_uni} |")
    L.append(f"| t2w | {n_t2_bil} | {n_t2_uni} | {n_t2_bil+n_t2_uni} |\n")
    sides = {}
    for v in variants.values():
        if v["side"]:
            sides[v["side"]] = sides.get(v["side"], 0) + 1
    L.append(f"Derived-side selection (bilateral patients, lesion-centroid half): {sides}\n")
    L.append("## Orientation\n")
    L.append(f"Volumes with axcodes != LPS: **{len(bad_ax)}**"
             + (f" — {sorted(set(bad_ax))}" if bad_ax else " (all native and all derived "
                "images and masks are LPS)") + "\n")
    L.append("## Lesion survival through the crop\n")
    L.append(f"Crops that LOST the lesion entirely: **{len(lost)}**"
             + (f"\n\n{chr(10).join('- ' + s + ' / ' + c + f' (native {n} vox)' for s, c, n in lost)}"
                if lost else " — none.") + "\n")
    L.append(f"Crops that partially clipped the lesion: **{len(clipped)}**"
             + (f"\n\n{chr(10).join('- ' + s + ' / ' + c + f' (survival {v:.4f})' for s, c, v in clipped)}"
                if clipped else " — none.") + "\n")
    L.append(f"## All warnings ({len(warned)} patients)\n")
    for r in warned:
        L.append(f"- `{r['sub']}` ({r['native_t1wce_fov']}): " + "; ".join(r["warnings"]))
    (SPLITS / "fov_variants_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:24]))


if __name__ == "__main__":
    sys.exit(main())
