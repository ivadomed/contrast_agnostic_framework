#!/usr/bin/env python3
"""
Shared voxel-orientation core for all datasets' 03_preprocess steps.

The project's canonical voxel-array convention is LPS ``('L','P','S')`` — the
orientation CHAOS and SLIVER07 ship in. Any dataset whose stored orientation
differs (AMOS shipped CT as LAS, MRI as RAS) must be reoriented to LPS so that
steps comparing in array space (and visual QA) line up. The reorientation is a
pure axis permute/flip (nibabel ``as_reoriented``): NO interpolation, NO
resampling — lossless and label-safe, so image and mask reorient identically.

This module is the single source of truth for that operation. Dataset-specific
03_preprocess scripts supply only the *list of files* (their BIDS / nnUNet trees)
and call :func:`run`; the per-file logic lives here once. A dataset already in LPS
(e.g. TRUSTED) uses the very same code path as an idempotent CHECK — every file
reports ``ok`` and nothing is written.

Writes are temp-file + atomic ``os.replace`` so that, when BIDS files are
hard-linked from a pristine ``0_raw`` tree, the replace breaks the hard-link and
leaves the raw inode untouched.
"""
import os
from pathlib import Path

import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

TARGET_AXCODES = ("L", "P", "S")


def reorient_file(path: Path, target_axcodes=TARGET_AXCODES, dry_run: bool = False) -> str:
    """Reorient one .nii.gz to ``target_axcodes`` in place (lossless permute/flip).

    Returns one of: ``'ok'`` (already at target), ``'fixed'`` (reoriented), or
    ``'would-fix'`` (dry-run and it differs). Idempotent — already-target files are
    never rewritten, so this doubles as an orientation *check* when all return 'ok'.
    """
    img = nib.load(str(path))
    if nib.aff2axcodes(img.affine) == tuple(target_axcodes):
        return "ok"
    if dry_run:
        return "would-fix"
    transform = ornt_transform(io_orientation(img.affine), axcodes2ornt(target_axcodes))
    out = img.as_reoriented(transform)                     # lossless axis permute/flip
    tmp = path.with_name(path.name + ".reorient.tmp.nii.gz")
    nib.save(out, str(tmp))
    os.replace(str(tmp), str(path))                        # breaks hard-link to 0_raw
    return "fixed"


def run(files, target_axcodes=TARGET_AXCODES, dry_run: bool = False,
        title: str = "orientation", rel_to: Path | None = None) -> dict:
    """Apply :func:`reorient_file` to every path in ``files``; print a report.

    ``rel_to`` shortens printed paths. Returns the status→count dict. Raises
    SystemExit if ``files`` is empty (nothing to do is almost always a mistake).
    """
    files = [Path(f) for f in files]
    tgt = "".join(target_axcodes)
    print("=" * 64)
    print(f"{title} → {tgt}{'  (DRY RUN)' if dry_run else ''}")
    print("=" * 64)
    if not files:
        raise SystemExit("ERROR: no .nii.gz files found — nothing to do.")

    counts = {"ok": 0, "fixed": 0, "would-fix": 0}
    for f in files:
        status = reorient_file(f, target_axcodes, dry_run)
        counts[status] += 1
        if status != "ok":
            shown = f.relative_to(rel_to) if rel_to else f
            print(f"  {status:9s} {shown}")

    print("-" * 64)
    print(f"  already {tgt} : {counts['ok']}")
    if dry_run:
        print(f"  would fix    : {counts['would-fix']}")
    else:
        print(f"  reoriented   : {counts['fixed']}")
    print(f"  total        : {len(files)}")
    return counts
