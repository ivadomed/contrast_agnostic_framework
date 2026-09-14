# ispy2 — T2w lesion masks re-derived through the shared DICOM frame

Written as `sub-*_desc-sharedframe_T2w_label-lesion_seg.nii.gz`. The old, registration-derived `sub-*_T2w_label-lesion_seg.nii.gz` files are left untouched on disk.

> **The old T2w masks are wrong for most patients** (see `t2w_label_registration_audit.md`: 91.8% zero overlap on the 438 unilateral patients, 27.9% on the 122 bilateral ones). Any existing ambl->ispy2 cross-evaluation **T2w** result computed against them is invalid and needs re-running against these corrected masks. T1wce results are unaffected — the T1wce mask is the collection's own DICOM-SEG and was never registered.

Patients: 560 written, 0 failed

- masks empty after resampling (lesion outside the T2w slab): **0**
- orientation != LPS: **0**
- lesion VOLUME preserved (mm^3 ratio new-T2w / T1wce): median 1.000, p5 0.965, p95 1.035
- volume ratio outside [0.5, 2.0]: **0** (expected to be non-zero: the T2w slab is much coarser than the DCE's, so a small lesion can be under- or over-represented by whole-voxel quantisation)

## Patients with warnings (0)

