# ispy2 — audit of the registration-derived T2w lesion masks

Ground truth for this audit is the shared DICOM FrameOfReferenceUID: the DCE
and T2w series of a patient are in ONE scanner frame, so the T1wce mask mapped
into T2w space by the IDENTITY world transform is where the lesion really is.

Patients audited: 560 (of 560; 0 skipped/empty/failed)

## Natively-unilateral patients (n=438)

- centroid distance T1wce-mask vs T2w-mask (mm): median **101.2**, p90 135.0, max 194.3
- Dice(T2w mask, identity-resampled T1wce mask): median **0.000**, mean 0.002
- masks on OPPOSITE sides of the midline: **193/438** (44.1%)
- Dice exactly 0 (no overlap at all): **402/438** (91.8%)
- centroid distance > 20 mm: **438/438** (100.0%)

## Natively-bilateral patients (n=122)

- centroid distance T1wce-mask vs T2w-mask (mm): median **6.3**, p90 164.8, max 261.0
- Dice(T2w mask, identity-resampled T1wce mask): median **0.323**, mean 0.351
- masks on OPPOSITE sides of the midline: **16/122** (13.1%)
- Dice exactly 0 (no overlap at all): **34/122** (27.9%)
- centroid distance > 20 mm: **42/122** (34.4%)

