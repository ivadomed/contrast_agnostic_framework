# ispy2 splits — patient-level, FOV-stratified

Patients: 476 train pool + 84 held-out internal test = 560

Laterality class balance (the 122 bilateral patients are the ONLY source of `t1wce/bil` cases, so they are stratified across every split):

| split | patients | natively bilateral | natively unilateral |
|---|---|---|---|
| test | 84 | 18 | 66 |
| fold0 val | 159 | 35 | 124 |
| fold1 val | 159 | 35 | 124 |
| fold2 val | 158 | 34 | 124 |

## Dataset100_ISPY2T1wce (t1wce) — nnU-Net cases

| split | cases | bilateral-FOV | unilateral-FOV |
|---|---|---|---|
| test | 102 | 18 | 84 |
| fold0 train | 386 | 69 | 317 |
| fold0 val | 194 | 35 | 159 |
| fold1 train | 386 | 69 | 317 |
| fold1 val | 194 | 35 | 159 |
| fold2 train | 388 | 70 | 318 |
| fold2 val | 192 | 34 | 158 |

## Dataset101_ISPY2T2w (t2w) — nnU-Net cases

| split | cases | bilateral-FOV | unilateral-FOV |
|---|---|---|---|
| test | 168 | 84 | 84 |
| fold0 train | 634 | 317 | 317 |
| fold0 val | 318 | 159 | 159 |
| fold1 train | 634 | 317 | 317 |
| fold1 val | 318 | 159 | 159 |
| fold2 train | 636 | 318 | 318 |
| fold2 val | 316 | 158 | 158 |

## Tumour-burden balance (T1wce DICOM-SEG lesion volume, mm^3)

- test: median 15875, p10 3847, p90 73721
- train pool: median 15978, p10 3953, p90 68234

## Straddle check

Every patient's FOV variants (and both contrasts) are assigned as a unit; asserted from the emitted case lists for every fold and for train-pool vs test in both nnU-Net datasets. **No patient straddles a split.**

